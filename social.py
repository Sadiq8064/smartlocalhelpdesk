# social.py
"""
Social media module with in-process TensorFlow prediction.

Features:
- /social/posts/create  (accepts file or image_url, will run TF model if available)
- /social/providers/{email}/posts
- /social/feed  (personalized feed by lat/lon/state/city)
- comments/likes/push endpoints
- Model auto-download from Google Drive (if ML_MODEL_PATH not present and MODEL_GDRIVE_ID provided)
- Model loaded once (lazy) in a threadpool to avoid blocking event loop
- Helper init function: init_social_routes(app, db)
"""

import os
import uuid
import math
import asyncio
import logging
import base64
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import EmailStr, BaseModel

from io import BytesIO
from PIL import Image
import numpy as np
import requests

logger = logging.getLogger("social")
logging.basicConfig(level=logging.INFO)

# ---------------- Config & Constants ----------------
USERS_COLLECTION = "users"
POSTS_COLLECTION = "posts"
SERVICES_COLLECTION = "services"
PROVIDERS_COLLECTION = "providers"

# Similarity API (optional)
SIMILARITY_API_BASE = os.getenv("SIMILARITY_API_BASE", None)

# Duplicate thresholds (env override possible)
DISTANCE_THRESHOLD_METERS = int(os.getenv("SOCIAL_DISTANCE_THRESHOLD_METERS", "100"))
TIME_THRESHOLD_DAYS = int(os.getenv("SOCIAL_TIME_THRESHOLD_DAYS", "4"))
CAPTION_SIMILARITY_THRESHOLD = float(os.getenv("SOCIAL_CAPTION_SIM_THRESHOLD", "0.80"))

# Feed radius for priority (km)
FEED_PRIORITY_KM = float(os.getenv("FEED_PRIORITY_KM", "3.0"))

# ImageKit config
IMAGEKIT_PUBLIC_KEY = os.getenv("IMAGEKIT_PUBLIC_KEY")
IMAGEKIT_PRIVATE_KEY = os.getenv("IMAGEKIT_PRIVATE_KEY")
IMAGEKIT_URL_ENDPOINT = os.getenv("IMAGEKIT_URL_ENDPOINT")

# Model storage path (Railway volume)
# Use environment override if present, otherwise default to /mnt/data/models
MODEL_DIR = os.getenv("MODEL_DIR", "/mnt/data/models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Model file path and GDrive id (expect ML_MODEL_PATH or MODEL_GDRIVE_ID in .env)
ML_MODEL_PATH = os.path.join(MODEL_DIR, "model.keras")
MODEL_GDRIVE_ID = os.getenv("MODEL_GDRIVE_ID") or "1GWw7-JpXXo72yo43jhUTuogrSGHCY_tv"  # fallback to the KERAS id you provided

# Model input size and classes
IMG_SIZE = (299, 299)
CLASS_NAMES = [
    'Damaged_highway', 'buses', 'drainage', 'dustbin',
    'electric_pole', 'fallen_tree', 'fire', 'footpath',
    'pothole_image_data', 'street_light_not_working',
    'street_light_working', 'traffic_signals', 'trash'
]

# ---------------- Module state ----------------
router = APIRouter()
_db = None

_model = None      # loaded TF model
_tf = None         # tensorflow module reference
_model_lock = asyncio.Lock()
_model_loaded_event = asyncio.Event()

# ---------------- Utilities ----------------
def haversine_meters(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

def normalize_department_for_compare(dept: Optional[str]) -> Optional[str]:
    if not dept:
        return None
    return dept.strip().replace(" ", "_").lower()

# ---------------- Google Drive download helper ----------------
def _gdrive_download_url(file_id: str):
    return "https://docs.google.com/uc?export=download&id=" + file_id

def download_file_from_gdrive(file_id: str, dest_path: str, chunk_size: int = 32768):
    """
    Synchronous download from Google Drive using confirm token flow.
    Intended to be called via run_in_executor.
    """
    session = requests.Session()
    url = _gdrive_download_url(file_id)

    resp = session.get(url, stream=True)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to start Google Drive download: HTTP {resp.status_code}")

    # Check cookie token
    token = None
    for k, v in session.cookies.items():
        if k.startswith("download_warning"):
            token = v
            break

    if token:
        resp = session.get(url, params={"confirm": token}, stream=True)
    else:
        # try to parse confirm token inside HTML for large files
        try:
            content = resp.content.decode("utf-8", errors="ignore")
            import re
            m = re.search(r"confirm=([0-9A-Za-z_-]+)&", content)
            if m:
                token = m.group(1)
                resp = session.get(url, params={"confirm": token}, stream=True)
        except Exception:
            pass

    total = 0
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                total += len(chunk)
    return total

# ---------------- Model loading & prediction ----------------
def _preprocess_image_bytes_sync(img_bytes: bytes):
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, 0)
    return arr

async def ensure_model_loaded():
    """
    Ensure model file exists and model loaded into memory.
    Downloads from Drive if missing (tries KERAS model path).
    """
    global _model, _tf
    if _model is not None:
        return

    async with _model_lock:
        if _model is not None:
            return

        try:
            # Download if missing
            if not os.path.exists(ML_MODEL_PATH):
                if MODEL_GDRIVE_ID:
                    logger.info("Model not found on disk; downloading from Google Drive (%s) to %s", MODEL_GDRIVE_ID, ML_MODEL_PATH)
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, download_file_from_gdrive, MODEL_GDRIVE_ID, ML_MODEL_PATH)
                    logger.info("Model downloaded to %s", ML_MODEL_PATH)
                else:
                    logger.warning("No MODEL_GDRIVE_ID configured; model won't be available.")
                    _model_loaded_event.set()
                    return

            # Load TF model in executor (avoid blocking event loop)
            loop = asyncio.get_running_loop()
            def _sync_load():
                import tensorflow as tf_local
                global _tf
                _tf = tf_local
                m = tf_local.keras.models.load_model(ML_MODEL_PATH)
                return m

            _model = await loop.run_in_executor(None, _sync_load)
            logger.info("Model loaded into memory from %s", ML_MODEL_PATH)
        except Exception as e:
            logger.exception("Failed to ensure/load model: %s", e)
            _model = None
            # still set the event so awaiting tasks don't hang
            _model_loaded_event.set()
            raise
        finally:
            _model_loaded_event.set()

async def predict_image_from_bytes(img_bytes: bytes):
    await ensure_model_loaded()
    global _model, _tf
    if _model is None:
        raise RuntimeError("Model not available")

    loop = asyncio.get_running_loop()
    arr = await loop.run_in_executor(None, _preprocess_image_bytes_sync, img_bytes)

    def _sync_predict(a):
        # 🔥 prevent recursive graph execution
        _tf.config.run_functions_eagerly(True)

        # 🔥 run model in eager mode
        preds = _model(a, training=False).numpy().flatten()

        idx = int(np.argmax(preds))
        conf = float(preds[idx])
        return idx, conf, preds.tolist()

    idx, conf, raw = await loop.run_in_executor(None, _sync_predict, arr)

    cls = CLASS_NAMES[idx] if 0 <= idx < len(CLASS_NAMES) else str(idx)
    return cls, conf, raw


# ---------------- Duplicate detection ----------------
def _call_caption_similarity_api_sync(text1: str, text2: str) -> Optional[float]:
    if not SIMILARITY_API_BASE:
        return None
    try:
        r = requests.get(SIMILARITY_API_BASE, params={"text1": text1, "text2": text2}, timeout=6)
        if r.status_code != 200:
            return None
        j = r.json()
        # try to extract first numeric value
        if isinstance(j, dict):
            for v in j.values():
                if isinstance(v, (int, float)):
                    return float(v)
            return None
        if isinstance(j, (float, int)):
            return float(j)
        return float(r.text.strip())
    except Exception:
        return None

async def _call_caption_similarity_api(text1: str, text2: str) -> Optional[float]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _call_caption_similarity_api_sync, text1, text2)

async def _find_duplicate(user_doc: dict, predicted_class: str, lat: float, lon: float, caption: str):
    posts_col = _db[POSTS_COLLECTION]
    state = user_doc.get("state")
    city = user_doc.get("city")
    if not state or not city:
        return False, None, None

    q = {"state": state, "city": city, "predicted_class": predicted_class}
    candidates = await posts_col.find(q).to_list(length=None)
    if not candidates:
        return False, None, None

    close = []
    for p in candidates:
        try:
            plat = float(p.get("latitude", 0))
            plon = float(p.get("longitude", 0))
        except Exception:
            continue
        dist = haversine_meters(lat, lon, plat, plon)
        if dist <= DISTANCE_THRESHOLD_METERS:
            close.append((p, dist))
    if not close:
        return False, None, None

    now = datetime.utcnow()
    recent = []
    for p, dist in close:
        created_at = p.get("created_at")
        if isinstance(created_at, str):
            try:
                created_dt = datetime.fromisoformat(created_at.replace("Z", ""))
            except Exception:
                created_dt = None
        elif isinstance(created_at, datetime):
            created_dt = created_at
        else:
            created_dt = None
        if created_dt and (now - created_dt) <= timedelta(days=TIME_THRESHOLD_DAYS):
            recent.append((p, dist))
    if not recent:
        return False, None, None

    best = None
    best_score = -1.0
    for p, dist in recent:
        other_caption = p.get("caption", "")
        sim = await _call_caption_similarity_api(caption, other_caption)
        if sim is None:
            # fallback heuristic: exact match or token overlap Jaccard
            if caption.strip().lower() == other_caption.strip().lower():
                sim = 1.0
            else:
                a_tokens = set(caption.lower().split())
                b_tokens = set(other_caption.lower().split())
                if not a_tokens or not b_tokens:
                    sim = 0.0
                else:
                    sim = len(a_tokens & b_tokens) / len(a_tokens | b_tokens)
        if sim is None:
            continue
        if sim > best_score:
            best_score = sim
            best = p

    if best_score >= CAPTION_SIMILARITY_THRESHOLD and best is not None:
        return True, best, best_score

    return False, None, best_score if best_score >= 0 else None

# ---------------- Pydantic models ----------------
class CreatePostResponse(BaseModel):
    success: bool
    message: str
    post_id: Optional[str] = None
    existing_post_id: Optional[str] = None
    duplicate_similarity: Optional[float] = None

class CommentCreateRequest(BaseModel):
    email: EmailStr
    post_id: str
    text: str

# ---------------- Core endpoints ----------------
@router.post("/posts/create", response_model=CreatePostResponse)
async def create_post(
    email: str = Form(...),
    caption: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    predicted_class: Optional[str] = Form(None),
    predicted_confidence: Optional[float] = Form(None)
):
    posts_col = _db[POSTS_COLLECTION]
    users_col = _db[USERS_COLLECTION]

    user_doc = await users_col.find_one({"email": email})
    if not user_doc:
        raise HTTPException(status_code=404, detail="User not found")
    if not user_doc.get("state") or not user_doc.get("city"):
        raise HTTPException(status_code=400, detail="User must have state and city set")
    if not caption or caption.strip() == "":
        raise HTTPException(status_code=400, detail="Caption required")

    # Get image bytes
    img_bytes = None
    if file:
        img_bytes = await file.read()
    elif image_url:
        try:
            r = requests.get(image_url, timeout=10)
            if r.status_code != 200:
                raise HTTPException(status_code=400, detail="Failed to fetch remote image URL")
            img_bytes = r.content
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch remote image: {e}")
    else:
        raise HTTPException(status_code=400, detail="Either file upload or image_url must be provided")

    # Run prediction if not provided
    if not predicted_class:
        try:
            await ensure_model_loaded()
            if _model is None:
                raise HTTPException(status_code=400, detail="Model not available on server; please provide predicted_class")
            predicted_class, conf, raw = await predict_image_from_bytes(img_bytes)
            predicted_confidence = float(conf)
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Prediction failed: %s", e)
            raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    # Map department (stored format)
    department_norm = predicted_class.strip().replace(" ", "_")
    department_stored = "_".join([part.capitalize() for part in department_norm.split("_")])

    # Duplicate check
    dup, matched_doc, sim_score = await _find_duplicate(user_doc, predicted_class, latitude, longitude, caption)
    if dup:
        return CreatePostResponse(
            success=False,
            message="The same post already exists. You can comment or push it.",
            existing_post_id=matched_doc.get("post_id"),
            duplicate_similarity=sim_score
        )

    # Upload to ImageKit using executor (blocking SDK)
    if not IMAGEKIT_PRIVATE_KEY or not IMAGEKIT_PUBLIC_KEY or not IMAGEKIT_URL_ENDPOINT:
        raise HTTPException(status_code=500, detail="ImageKit configuration missing on server")

    loop = asyncio.get_running_loop()
    def _sync_upload_to_imagekit(bytes_data, filename):
        from imagekitio import ImageKit
        ik = ImageKit(public_key=IMAGEKIT_PUBLIC_KEY, private_key=IMAGEKIT_PRIVATE_KEY, url_endpoint=IMAGEKIT_URL_ENDPOINT)
        b64 = base64.b64encode(bytes_data).decode()
        res = ik.upload_file(file=b64, file_name=filename)
        # extract URL robustly
        try:
            if getattr(res, "response_metadata", None):
                raw = getattr(res.response_metadata, "raw", None)
                if isinstance(raw, dict) and raw.get("url"):
                    return {"url": raw.get("url")}
        except Exception:
            pass
        if hasattr(res, "url"):
            return {"url": res.url}
        try:
            return dict(res)
        except Exception:
            return {"url": None}

    fname = (file.filename if file else f"remote_{uuid.uuid4().hex}.jpg")
    try:
        upload_res = await loop.run_in_executor(None, _sync_upload_to_imagekit, img_bytes, fname)
    except Exception as e:
        logger.exception("ImageKit upload failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Image upload failed: {e}")

    url = upload_res.get("url")
    if not url:
        raise HTTPException(status_code=500, detail="Image upload succeeded but no URL returned")

    # Save post
    post_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat() + "Z"
    user_name = user_doc.get("name") or user_doc.get("username") or user_doc.get("full_name") or ""

    post_doc = {
        "post_id": post_id,
        "email": email,
        "user_name": user_name,
        "image_url": url,
        "caption": caption,
        "latitude": float(latitude),
        "longitude": float(longitude),
        "state": user_doc["state"],
        "city": user_doc["city"],
        "predicted_class": predicted_class,
        "predicted_confidence": float(predicted_confidence) if predicted_confidence is not None else None,
        "department": department_stored,
        "likes": [],
        "pushes": [],
        "likes_count": 0,
        "push_count": 0,
        "comments": [],
        "created_at": created_at
    }

    await posts_col.insert_one(post_doc)
    return CreatePostResponse(success=True, message="Post created", post_id=post_id)

# ---------------- Provider endpoint ----------------
@router.get("/providers/{email}/posts")
async def provider_get_posts(email: str, limit: int = 50, skip: int = 0):
    providers_col = _db[PROVIDERS_COLLECTION]
    posts_col = _db[POSTS_COLLECTION]
    services_col = _db[SERVICES_COLLECTION]

    provider = await providers_col.find_one({"email": email})
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")

    service = await services_col.find_one({"provider_email": email})
    if service:
        prov_state = service.get("state") or provider.get("state")
        prov_city = service.get("city") or provider.get("city")
        prov_dept = service.get("department") or provider.get("department")
    else:
        prov_state = provider.get("state")
        prov_city = provider.get("city")
        prov_dept = provider.get("department")

    if not prov_state or not prov_city or not prov_dept:
        raise HTTPException(status_code=400, detail="Provider must have state/city/department registered")

    prov_dept_norm = normalize_department_for_compare(prov_dept)

    cursor = posts_col.find({"state": prov_state, "city": prov_city})
    docs = await cursor.sort("created_at", -1).skip(skip).limit(limit).to_list(length=None)

    filtered = []
    for d in docs:
        if normalize_department_for_compare(d.get("department")) == prov_dept_norm:
            doc = dict(d)
            if isinstance(doc.get("created_at"), datetime):
                doc["created_at"] = doc["created_at"].isoformat()
            if "_id" in doc:
                doc["_id"] = str(doc["_id"])
            filtered.append(doc)

    return JSONResponse(content={
        "success": True,
        "provider_email": email,
        "department": prov_dept,
        "count": len(filtered),
        "posts": filtered
    })

# ---------------- Feed API ----------------
@router.post("/feed")
async def get_personalized_feed(
    latitude: float = Form(...),
    longitude: float = Form(...),
    state: str = Form(...),
    city: str = Form(...),
    limit: int = Form(50),
    skip: int = Form(0)
):
    posts_col = _db[POSTS_COLLECTION]
    cursor = posts_col.find({"state": state, "city": city})
    docs = await cursor.to_list(length=None)

    enriched = []
    for d in docs:
        try:
            plat = float(d.get("latitude", 0))
            plon = float(d.get("longitude", 0))
        except Exception:
            continue
        dist_m = haversine_meters(latitude, longitude, plat, plon)
        created_at_raw = d.get("created_at")
        if isinstance(created_at_raw, str):
            try:
                created_dt = datetime.fromisoformat(created_at_raw.replace("Z", ""))
            except Exception:
                created_dt = None
        elif isinstance(created_at_raw, datetime):
            created_dt = created_at_raw
        else:
            created_dt = None
        enriched.append({
            "doc": d,
            "distance_m": dist_m,
            "created_at": created_dt,
            "push_count": int(d.get("push_count", 0) or 0)
        })

    priority_m = FEED_PRIORITY_KM * 1000.0
    within = [e for e in enriched if e["distance_m"] <= priority_m]
    beyond = [e for e in enriched if e["distance_m"] > priority_m]

    # sort rules: nearest first, then higher pushes, then recent
    within.sort(key=lambda x: (x["distance_m"], -x["push_count"], -(x["created_at"].timestamp() if x["created_at"] else 0)))
    beyond.sort(key=lambda x: (x["distance_m"], -(x["created_at"].timestamp() if x["created_at"] else 0)))

    ranked = within + beyond
    sliced = ranked[skip: skip + limit]

    def _serialize(e):
        d = dict(e["doc"])
        if "_id" in d:
            d["_id"] = str(d["_id"])
        if isinstance(d.get("created_at"), datetime):
            d["created_at"] = d["created_at"].isoformat()
        return {
            "post_id": d.get("post_id"),
            "caption": d.get("caption"),
            "image_url": d.get("image_url"),
            "created_at": d.get("created_at"),
            "latitude": d.get("latitude"),
            "longitude": d.get("longitude"),
            "state": d.get("state"),
            "city": d.get("city"),
            "department": d.get("department"),
            "likes_count": d.get("likes_count", 0),
            "push_count": d.get("push_count", 0),
            "comments_count": len(d.get("comments", []) or []),
            "distance_m": round(e["distance_m"], 2)
        }

    feed = [_serialize(e) for e in sliced]
    return {"success": True, "count": len(feed), "feed": feed}

# ---------------- Comments / likes / push endpoints ----------------
@router.post("/comments/create")
async def create_comment(req: CommentCreateRequest):
    posts_col = _db[POSTS_COLLECTION]
    users_col = _db[USERS_COLLECTION]

    user = await users_col.find_one({"email": req.email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    post = await posts_col.find_one({"post_id": req.post_id})
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    comment_id = str(uuid.uuid4())
    comment = {
        "comment_id": comment_id,
        "email": req.email,
        "user_name": user.get("name") or user.get("username") or "Unknown",
        "text": req.text,
        "state": user.get("state"),
        "city": user.get("city"),
        "created_at": datetime.utcnow().isoformat() + "Z"
    }
    await posts_col.update_one({"post_id": req.post_id}, {"$push": {"comments": comment}})
    return {"success": True, "comment_id": comment_id}

@router.get("/comments/user/{email}")
async def list_comments_by_user(email: str):
    posts_col = _db[POSTS_COLLECTION]
    cursor = posts_col.find({"comments.email": email}, {"comments.$": 1, "post_id": 1})
    results = []
    async for p in cursor:
        for c in p.get("comments", []):
            if c.get("email") == email:
                entry = {
                    "post_id": p.get("post_id"),
                    "comment_id": c.get("comment_id"),
                    "text": c.get("text"),
                    "created_at": c.get("created_at")
                }
                results.append(entry)
    return {"count": len(results), "comments": results}

@router.delete("/comments/delete")
async def delete_comment(email: str = Form(...), comment_id: str = Form(...)):
    posts_col = _db[POSTS_COLLECTION]
    res = await posts_col.update_one(
        {"comments.comment_id": comment_id, "comments.email": email},
        {"$pull": {"comments": {"comment_id": comment_id, "email": email}}}
    )
    if res.modified_count == 0:
        raise HTTPException(status_code=404, detail="Comment not found or not owned by user")
    return {"success": True, "message": "Comment deleted"}

@router.post("/posts/like")
async def like_post(email: str = Form(...), post_id: str = Form(...), action: int = Form(...)):
    posts_col = _db[POSTS_COLLECTION]
    p = await posts_col.find_one({"post_id": post_id})
    if not p:
        raise HTTPException(status_code=404, detail="Post not found")

    if action == 1:
        if email in (p.get("likes") or []):
            return {"success": True, "message": "Already liked"}
        await posts_col.update_one({"post_id": post_id}, {"$push": {"likes": email}, "$inc": {"likes_count": 1}})
        return {"success": True, "message": "Liked"}
    else:
        if email not in (p.get("likes") or []):
            return {"success": True, "message": "Not previously liked"}
        await posts_col.update_one({"post_id": post_id}, {"$pull": {"likes": email}, "$inc": {"likes_count": -1}})
        return {"success": True, "message": "Unliked"}

@router.post("/posts/push")
async def push_post(email: str = Form(...), post_id: str = Form(...), action: int = Form(...)):
    posts_col = _db[POSTS_COLLECTION]
    p = await posts_col.find_one({"post_id": post_id})
    if not p:
        raise HTTPException(status_code=404, detail="Post not found")

    if action == 1:
        if email in (p.get("pushes") or []):
            return {"success": True, "message": "Already pushed"}
        await posts_col.update_one({"post_id": post_id}, {"$push": {"pushes": email}, "$inc": {"push_count": 1}})
        return {"success": True, "message": "Pushed/upvoted"}
    else:
        if email not in (p.get("pushes") or []):
            return {"success": True, "message": "Not previously pushed"}
        await posts_col.update_one({"post_id": post_id}, {"$pull": {"pushes": email}, "$inc": {"push_count": -1}})
        return {"success": True, "message": "Push removed"}

@router.get("/posts/user/{email}")
async def get_posts_by_user(email: str):
    posts_col = _db[POSTS_COLLECTION]
    docs = await posts_col.find({"email": email}).sort("created_at", -1).to_list(length=None)
    out = []
    for d in docs:
        doc = dict(d)
        if isinstance(doc.get("created_at"), datetime):
            doc["created_at"] = doc["created_at"].isoformat()
        if "_id" in doc:
            doc["_id"] = str(doc["_id"])
        out.append(doc)
    return JSONResponse(content={"count": len(out), "posts": out})

@router.delete("/posts/delete")
async def delete_post(email: str = Form(...), post_id: str = Form(...)):
    posts_col = _db[POSTS_COLLECTION]
    res = await posts_col.delete_one({"post_id": post_id, "email": email})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Post not found or not owned by user")
    return {"success": True, "message": "Post deleted"}

# ---------------- EXTRA ADMIN / SEARCH APIs ----------------
@router.get("/posts/all")
async def get_all_posts(limit: int = 200, skip: int = 0):
    posts_col = _db[POSTS_COLLECTION]
    docs = await posts_col.find().sort("created_at", -1).skip(skip).limit(limit).to_list(length=None)
    out = []
    for d in docs:
        doc = dict(d)
        if "_id" in doc:
            doc["_id"] = str(doc["_id"])
        if isinstance(doc.get("created_at"), datetime):
            doc["created_at"] = doc["created_at"].isoformat()
        out.append(doc)
    return {"success": True, "count": len(out), "posts": out}

@router.get("/posts/details/{post_id}")
async def get_post_details(post_id: str):
    posts_col = _db[POSTS_COLLECTION]
    post = await posts_col.find_one({"post_id": post_id})
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    if "_id" in post:
        post["_id"] = str(post["_id"])
    if isinstance(post.get("created_at"), datetime):
        post["created_at"] = post["created_at"].isoformat()
    return {"success": True, "post": post}

@router.get("/search")
async def search_posts(query: str, limit: int = 100):
    posts_col = _db[POSTS_COLLECTION]
    docs = await posts_col.find({"caption": {"$regex": query, "$options": "i"}}).limit(limit).to_list(length=None)
    out = []
    for d in docs:
        doc = dict(d)
        if "_id" in doc:
            doc["_id"] = str(doc["_id"])
        if isinstance(doc.get("created_at"), datetime):
            doc["created_at"] = doc["created_at"].isoformat()
        out.append(doc)
    return {"success": True, "count": len(out), "posts": out}

@router.post("/posts/location")
async def get_posts_by_location(
    latitude: float = Form(...),
    longitude: float = Form(...),
    radius_km: float = Form(2.0)
):
    posts_col = _db[POSTS_COLLECTION]
    docs = await posts_col.find().to_list(length=None)
    results = []
    for p in docs:
        try:
            plat = float(p.get("latitude", 0))
            plon = float(p.get("longitude", 0))
        except:
            continue
        dist_m = haversine_meters(latitude, longitude, plat, plon)
        if dist_m <= radius_km * 1000:
            p_copy = dict(p)
            p_copy["distance_m"] = round(dist_m, 2)
            if "_id" in p_copy:
                p_copy["_id"] = str(p_copy["_id"])
            results.append(p_copy)
    return {"success": True, "count": len(results), "posts": results}

@router.get("/posts/class/{predicted_class}")
async def get_posts_by_class(predicted_class: str, limit: int = 100):
    posts_col = _db[POSTS_COLLECTION]
    docs = await posts_col.find({"predicted_class": predicted_class}).limit(limit).to_list(length=None)
    out = []
    for d in docs:
        doc = dict(d)
        if "_id" in doc:
            doc["_id"] = str(doc["_id"])
        if isinstance(doc.get("created_at"), datetime):
            doc["created_at"] = doc["created_at"].isoformat()
        out.append(doc)
    return {"success": True, "count": len(out), "posts": out}

# ---------------- Module init ----------------
def init_social_routes(app, db, prefix: str = "/social"):
    """
    Initialize the social routes.

    Parameters:
    - app: FastAPI instance
    - db: AsyncIOMotorDatabase instance (db = client[DB_NAME])
    - prefix: route prefix (default '/social')

    This sets internal _db, mounts router at `prefix`, and
    schedules a background model warmup (non-blocking).
    """
    global _db
    _db = db
    app.include_router(router, prefix=prefix, tags=["social"])
    # start background model warmup but don't block startup
    try:
        asyncio.create_task(_background_model_warmup())
    except Exception:
        # if called outside of running loop, swallow (app startup will still call warmup)
        pass

async def _background_model_warmup():
    try:
        await ensure_model_loaded()
    except Exception as e:
        logger.warning("Background model warmup failed: %s", e)

# End of social.py
