# social.py
"""
Social media module using Gemini for image-to-department classification (no TensorFlow).

Key behavior:
- Uses Gemini (model hard-coded to "gemini-2.5-flash") to classify an uploaded image into
  exactly one department (from the services collection) or "none".
- Sends a strict prompt + base64 JPEG image to Gemini and expects EXACT JSON:
    {"department": "<department-name-or-none>"}
  with no extra text.
- All other endpoints and logic (uploads, duplicate detection, feed, etc.) remain the same.
- Gemini API key is read from the environment variable GEMINI_API_KEY.
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

# Gemini client
from google import genai
from google.genai import types

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

# Gemini config (hard-coded model, API key from env)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"

# Model storage path placeholder (kept for compatibility; not used for TF)
MODEL_DIR = os.getenv("MODEL_DIR", "/mnt/data/models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- Module state ----------------
router = APIRouter()
_db = None

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

# ---------------- Gemini helper ----------------
def _make_gemini_prompt(departments: List[str], b64_image: str) -> str:
    """
    Build a strict prompt asking Gemini to return EXACT JSON:
    {"department":"<one of the departments or none>"}
    """
    # clean department names and join with commas in a single-line list for clarity
    clean_depts = [d.strip() for d in departments if d and isinstance(d, str)]
    # Build a short instruction. We DO NOT include long lists printed as classes in prompt.
    # We'll pass the department list explicitly.
    dept_list_text = ", ".join(clean_depts) if clean_depts else ""
    prompt = (
        "You are an image-to-department classifier. "
        "You will be given a base64-encoded JPEG image and a comma-separated list of department names.\n\n"
        f"Departments available: {dept_list_text}\n\n"
        "Task: Inspect the image and choose EXACTLY ONE department name from the list that this image belongs to. "
        "If the image does NOT match any department, return `none`.\n\n"
        "Return ONLY valid JSON in the EXACT format below and NOTHING else (no explanation, no whitespace padding):\n"
        '{"department":"<one-department-name-or-none>"}\n\n'
        "Important: department name must exactly match one of the provided department names (case-sensitive match is not required). "
        "If uncertain, pick the best match or return \"none\". Do NOT include any other keys or fields.\n\n"
        "Base64Image:\n" + b64_image + "\n\n"
    )
    return prompt

async def _classify_image_with_gemini(img_bytes: bytes, departments: List[str]) -> str:
    """
    Calls Gemini to classify image into one department (or 'none').
    Returns the department string (or 'none').
    Raises RuntimeError on Gemini failures.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set in environment")

    # convert to base64 (JPEG)
    try:
        # ensure it's JPEG bytes: if input is other format PIL will convert
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        buf = BytesIO()
        img.save(buf, format="JPEG")
        jpeg_bytes = buf.getvalue()
    except Exception as e:
        raise RuntimeError(f"Failed to normalize image to JPEG: {e}") from e

    b64_image = base64.b64encode(jpeg_bytes).decode("ascii")

    prompt = _make_gemini_prompt(departments, b64_image)

    loop = asyncio.get_running_loop()

    def _sync_call():
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=256)
            )
            # response.text should contain the assistant's output
            return response.text
        except Exception as e:
            raise RuntimeError(f"Gemini request failed: {e}") from e

    try:
        result_text = await loop.run_in_executor(None, _sync_call)
    except Exception as e:
        logger.exception("Gemini classification failed: %s", e)
        raise RuntimeError(f"Gemini classification failed: {e}") from e

    # Parse result_text: expect exact JSON like {"department":"name"} with no extra text
    result_text_stripped = result_text.strip()
    try:
        parsed = None
        # Try simple JSON parse
        import json
        parsed = json.loads(result_text_stripped)
        if isinstance(parsed, dict) and "department" in parsed:
            dept = parsed["department"]
            # Normalize to None or exact string (if user wanted exact DB department, we will attempt best match)
            if isinstance(dept, str):
                dept_clean = dept.strip()
                if dept_clean.lower() == "none":
                    return "none"
                # Attempt to find the exact department from provided list (case-insensitive match)
                for d in departments:
                    if d and d.strip().lower() == dept_clean.lower():
                        return d  # return DB department exact string
                # If not matched, still return the raw string (caller may accept it)
                return dept_clean
        # If JSON not as expected, fall through to fallback parsing below
    except Exception:
        # not valid JSON — fall through
        pass

    # Fallback heuristics: try to extract department value using simple parsing
    # Look for pattern like {"department":"..."} anywhere in the text
    import re, json
    m = re.search(r'\"department\"\s*:\s*\"([^\"]+)\"', result_text_stripped)
    if m:
        dept_raw = m.group(1).strip()
        if dept_raw.lower() == "none":
            return "none"
        for d in departments:
            if d and d.strip().lower() == dept_raw.lower():
                return d
        return dept_raw

    # If still nothing, try to find any token from departments in the text
    txt_lower = result_text_stripped.lower()
    for d in departments:
        if d and d.strip().lower() in txt_lower:
            return d

    # Last resort: return 'none'
    logger.warning("Gemini returned unexpected output while classifying image: %r", result_text_stripped)
    return "none"

# ---------------- Duplicate detection helpers (unchanged) ----------------
def _call_caption_similarity_api_sync(text1: str, text2: str) -> Optional[float]:
    if not SIMILARITY_API_BASE:
        return None
    try:
        r = requests.get(SIMILARITY_API_BASE, params={"text1": text1, "text2": text2}, timeout=6)
        if r.status_code != 200:
            return None
        j = r.json()
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

# ---------------- Core endpoints (mostly unchanged) ----------------
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
    services_col = _db[SERVICES_COLLECTION]

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
            # Fetch departments from services collection
            departments = await services_col.distinct("department")
            # classify with Gemini
            predicted_department = await _classify_image_with_gemini(img_bytes, departments)
            if not predicted_department:
                predicted_department = "none"
            if predicted_department == "none":
                # We keep the old predicted_class semantics: a label (we will store "none" or leave to client)
                predicted_class = "none"
                predicted_confidence = 0.0
            else:
                predicted_class = predicted_department
                predicted_confidence = None  # Gemini prompt does not return confidence; leave as None
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
    """
    global _db
    _db = db
    app.include_router(router, prefix=prefix, tags=["social"])
    # No TensorFlow warmup. Gemini calls are done per-request.

# End of file
