# service_provider.py — Production-ready, normalized locations, Brevo SDK integrated
import os
import random
import string
import time
import asyncio
from datetime import datetime, timedelta
from bson import ObjectId
import aiohttp
import uuid
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Dict, Any, List
import shutil
from pathlib import Path
import base64
import logging

# ImageKit SDK is synchronous; we'll call it in a threadpool
try:
    from imagekitio import ImageKit
    from imagekitio.models.UploadFileRequestOptions import UploadFileRequestOptions
except Exception:
    ImageKit = None
    UploadFileRequestOptions = None

# Brevo SDK
import sib_api_v3_sdk
from sib_api_v3_sdk.rest import ApiException

GKEY = os.getenv("GFAPI_KEY")
UPLOAD_BASE_URL = os.getenv("GFAPI_BASE_URL")
BREVO_API_KEY = os.getenv("BREVO_API_KEY")

IMAGEKIT_CONFIG = {
    "public_key": os.getenv("IMAGEKIT_PUBLIC_KEY"),
    "private_key": os.getenv("IMAGEKIT_PRIVATE_KEY"),
    "url_endpoint": os.getenv("IMAGEKIT_URL_ENDPOINT")
}

# Add validation after these lines:
if not GKEY:
    raise ValueError("GFAPI_KEY environment variable is required")
if not UPLOAD_BASE_URL:
    raise ValueError("GFAPI_BASE_URL environment variable is required")
if not BREVO_API_KEY:
    raise ValueError("BREVO_API_KEY environment variable is required")
if not IMAGEKIT_CONFIG["public_key"]:
    raise ValueError("IMAGEKIT_PUBLIC_KEY environment variable is required")
if not IMAGEKIT_CONFIG["private_key"]:
    raise ValueError("IMAGEKIT_PRIVATE_KEY environment variable is required")
if not IMAGEKIT_CONFIG["url_endpoint"]:
    raise ValueError("IMAGEKIT_URL_ENDPOINT environment variable is required")

# OTP TTL seconds
_otps_ttl_seconds = int(os.getenv("OTP_TTL_SECONDS", "300"))

# ------------------------------
# Setup
# ------------------------------
router = APIRouter()
_db = None

# Upload temp dir
UPLOAD_DIR = Path("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {'.pdf', '.jpg', '.jpeg', '.png', '.txt', '.xlsx', '.xls'}
UPLOAD_TYPES = ['notice', 'frequently_asked', 'important_data']

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("service_provider")

# Initialize ImageKit client if available
imagekit = None
if ImageKit is not None:
    try:
        imagekit = ImageKit(
            public_key=IMAGEKIT_CONFIG["public_key"],
            private_key=IMAGEKIT_CONFIG["private_key"],
            url_endpoint=IMAGEKIT_CONFIG["url_endpoint"]
        )
    except Exception as e:
        logger.warning("ImageKit init failed: %s", e)
        imagekit = None

# ------------------------------
# DB init & indexes
# ------------------------------
def init_service_routes(app, db):
    """
    Called from app.py to initialize DB reference and start background tasks (indexes).
    """
    global _db
    _db = db
    # schedule index creation
    asyncio.create_task(_ensure_indexes())

async def _ensure_indexes():
    try:
        await _db.otps.create_index("created_at", expireAfterSeconds=_otps_ttl_seconds)
        await _db.services.create_index([("department", 1), ("city", 1), ("state", 1)], unique=True)
        await _db.tickets.create_index("ticket_id", unique=True)
        await _db.tickets.create_index("user_email")
        await _db.tickets.create_index("provider_email")
        await _db.uploads.create_index("delete_at")
        await _db.uploads.create_index([("provider_email", 1), ("document_id", 1)])
        logger.info("Indexes ensured")
    except Exception as e:
        logger.exception("Error ensuring indexes: %s", e)

# ------------------------------
# Small helpers
# ------------------------------
def _generate_otp() -> str:
    return "".join(random.choices(string.digits, k=6))

def _generate_ticket_id() -> str:
    characters = string.ascii_uppercase + string.digits
    return "".join(random.choices(characters, k=6))

def _normalize_location(value: Optional[str]) -> Optional[str]:
    """
    Normalize location strings to Title Case and remove extra spaces.
    Matches user.py normalization.
    """
    if value is None:
        return None
    return " ".join(part.capitalize() for part in value.strip().split())

# ------------------------------
# Brevo email sender (uses official SDK in a threadpool)
# ------------------------------
async def _send_brevo_email(to_email: str, subject: str, html_content: str):
    """
    Send email using sib_api_v3_sdk. The SDK is blocking, so we run it inside
    a threadpool using run_in_executor to avoid blocking the event loop.
    Returns the Brevo message id on success.
    Raises RuntimeError on failure.
    """
    def _send_sync():
        configuration = sib_api_v3_sdk.Configuration()
        configuration.api_key['api-key'] = BREVO_API_KEY
        api_instance = sib_api_v3_sdk.TransactionalEmailsApi(sib_api_v3_sdk.ApiClient(configuration))

        email_data = sib_api_v3_sdk.SendSmtpEmail(
            to=[{"email": to_email}],
            sender={"name": "Smart Local Helpdesk", "email": "mrsadiq471@gmail.com"},
            subject=subject,
            html_content=html_content,
            text_content="This is an automated email from Smart Local Helpdesk."
        )
        try:
            result = api_instance.send_transac_email(email_data)
            # result.message_id exists when successful
            return {"success": True, "brevo_id": getattr(result, "message_id", None)}
        except ApiException as e:
            # ApiException has body/message
            raise RuntimeError(f"Brevo ApiException: {e}")
        except Exception as e:
            raise RuntimeError(f"Brevo send error: {e}")

    loop = asyncio.get_running_loop()
    try:
        res = await loop.run_in_executor(None, _send_sync)
        return res
    except Exception as e:
        logger.exception("Brevo send failed: %s", e)
        raise

# ------------------------------
# Update users when a new service is created
# ------------------------------
async def _update_users_for_new_service(service_state: str, service_city: str, store_name: str):
    """
    Normalize and update all users in same state+city to include the store.
    """
    try:
        state_norm = _normalize_location(service_state)
        city_norm = _normalize_location(service_city)

        users = await _db.users.find({
            "state": state_norm,
            "city": city_norm
        }).to_list(length=None)

        for user in users:
            if "available_service_stores" not in user:
                user["available_service_stores"] = []
            if store_name not in user["available_service_stores"]:
                updated_stores = user["available_service_stores"] + [store_name]
                await _db.users.update_one(
                    {"_id": user["_id"]},
                    {"$set": {"available_service_stores": updated_stores}}
                )
        logger.info("Updated %d users for new service %s in %s, %s", len(users), store_name, city_norm, state_norm)
    except Exception as e:
        logger.exception("Error updating users for new service: %s", e)

# ------------------------------
# Ticket counters
# ------------------------------
async def _update_ticket_counts(provider_email: str):
    try:
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = today_start + timedelta(days=1)

        today_count = await _db.tickets.count_documents({
            "provider_email": provider_email,
            "created_at": {"$gte": today_start, "$lt": today_end}
        })

        pending_count = await _db.tickets.count_documents({
            "provider_email": provider_email,
            "status": "pending"
        })

        completed_count = await _db.tickets.count_documents({
            "provider_email": provider_email,
            "status": "completed"
        })

        await _db.services.update_one(
            {"provider_email": provider_email},
            {"$set": {
                "ticket_counts.today_ticket_count": today_count,
                "ticket_counts.pending_ticket_count": pending_count,
                "ticket_counts.completed_ticket_count": completed_count
            }}
        )
    except Exception as e:
        logger.exception("Error updating ticket counts: %s", e)

# ------------------------------
# ImageKit upload/delete (sync SDK run in threadpool)
# ------------------------------
def _upload_imagekit_sync(file_path: str, filename: str, folder: str = "service_documents"):
    if imagekit is None:
        raise RuntimeError("ImageKit SDK not configured or unavailable.")
    with open(file_path, "rb") as f:
        file_content = f.read()
    upload_response = imagekit.upload_file(
        file=base64.b64encode(file_content).decode(),
        file_name=filename,
        options=UploadFileRequestOptions(
            use_unique_file_name=True,
            folder=folder,
            is_private_file=False
        )
    )
    return upload_response

async def _upload_to_imagekit(file_path: Path, filename: str, folder: str = "service_documents"):
    try:
        loop = asyncio.get_running_loop()
        upload_response = await loop.run_in_executor(None, _upload_imagekit_sync, str(file_path), filename, folder)
        # Normalize SDK response
        try:
            meta = upload_response
            status_code = getattr(getattr(meta, "response_metadata", None), "http_status_code", None)
            if status_code is None:
                status_code = getattr(meta, "status", None) or 200
            if int(status_code) in (200, 201):
                return {
                    'success': True,
                    'file_id': getattr(meta, "file_id", None) or getattr(meta, "fileId", None),
                    'url': getattr(meta, "url", None),
                    'thumbnail_url': getattr(meta, "thumbnail_url", None) or getattr(meta, "thumbnailUrl", None),
                    'file_type': getattr(meta, "file_type", None),
                    'size': getattr(meta, "size", None)
                }
            else:
                return {'success': False, 'error': f"ImageKit upload failed, status {status_code}"}
        except Exception as e:
            return {'success': False, 'error': f"ImageKit parse error: {e}"}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def _delete_imagekit_sync(file_id: str):
    if imagekit is None:
        raise RuntimeError("ImageKit SDK not configured or unavailable.")
    resp = imagekit.delete_file(file_id=file_id)
    return resp

async def _delete_from_imagekit(file_id: str):
    try:
        loop = asyncio.get_running_loop()
        resp = await loop.run_in_executor(None, _delete_imagekit_sync, file_id)
        status = getattr(getattr(resp, "response_metadata", None), "http_status_code", None)
        if status is None:
            status = getattr(resp, "status", None)
        if int(status) in (200, 204):
            return {'success': True, 'message': 'File deleted from ImageKit'}
        else:
            return {'success': False, 'error': f"ImageKit delete failed status {status}"}
    except Exception as e:
        return {'success': False, 'error': str(e)}

# ------------------------------
# GFAPI upload/delete helpers
# ------------------------------
async def _upload_to_gfapi(store_name: str, file_path: Path, filename: str, max_retries: int = 3):
    url = f"{UPLOAD_BASE_URL}/{store_name}/upload"
    attempt = 0
    backoff = 1.0

    while attempt < max_retries:
        attempt += 1
        try:
            with open(file_path, "rb") as f:
                form_data = aiohttp.FormData()
                form_data.add_field('api_key', GKEY)
                form_data.add_field('limit', "true")
                form_data.add_field('files', f, filename=filename, content_type='application/octet-stream')

                async with aiohttp.ClientSession() as session:
                    async with session.post(url, data=form_data, timeout=60) as response:
                        text = await response.text()
                        if response.status == 200:
                            try:
                                result = await response.json()
                            except Exception:
                                result = {}
                            if result.get('success') and result.get('results'):
                                file_result = result['results'][0]
                                return {
                                    'success': True,
                                    'filename': file_result.get('filename'),
                                    'document_id': file_result.get('document_id'),
                                    'document_resource': file_result.get('document_resource')
                                }
                            else:
                                logger.warning("GFAPI upload non-success body: %s (attempt %d)", result, attempt)
                        else:
                            logger.warning("GFAPI upload HTTP %s: %s (attempt %d)", response.status, text, attempt)
        except Exception as e:
            logger.exception("GFAPI upload exception (attempt %d): %s", attempt, e)

        await asyncio.sleep(backoff)
        backoff *= 2

    return {'success': False, 'error': f"Upload to GFAPI failed after {max_retries} attempts"}

async def _delete_file_from_gfapi(store_name: str, document_id: str):
    try:
        url = f"{UPLOAD_BASE_URL}/{store_name}/documents/{document_id}"
        params = {'api_key': GKEY}
        async with aiohttp.ClientSession() as session:
            async with session.delete(url, params=params, timeout=30) as response:
                text = await response.text()
                if response.status in (200, 204):
                    return {'success': True, 'message': 'File deleted successfully'}
                else:
                    return {'success': False, 'error': f"Delete failed: {response.status} {text}"}
    except Exception as e:
        return {'success': False, 'error': str(e)}

# Make _delete_file_from_gfapi available for import by app.py (app.py references it)
# It's defined above and exported by module.

# ------------------------------
# Parse delete_at string
# ------------------------------
def _parse_delete_time(delete_at: Optional[str]):
    try:
        if not delete_at:
            return None
        current_time = datetime.utcnow()
        if len(delete_at) == 10 and delete_at.count('-') == 2:
            delete_date = datetime.strptime(delete_at, '%Y-%m-%d')
            return delete_date.replace(hour=23, minute=59, second=59)
        elif len(delete_at) == 5 and delete_at.count(':') == 1:
            hours, minutes = map(int, delete_at.split(':'))
            delete_time = current_time.replace(hour=hours, minute=minutes, second=0, microsecond=0)
            if delete_time < current_time:
                delete_time += timedelta(days=1)
            return delete_time
        else:
            return None
    except Exception:
        return None

# ---------------------------------------------------------
# DATE FILTER HELPER (today, yesterday, week, month, year)
# ---------------------------------------------------------
def _get_question_date_filter(filter_by: Optional[str]):
    now = datetime.utcnow()

    if filter_by == "today":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        return {"$gte": start, "$lt": end}

    if filter_by == "yesterday":
        end = now.replace(hour=0, minute=0, second=0, microsecond=0)
        start = end - timedelta(days=1)
        return {"$gte": start, "$lt": end}

    if filter_by == "this_week":
        start = now - timedelta(days=now.weekday())  # Monday start
        start = start.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=7)
        return {"$gte": start, "$lt": end}

    if filter_by == "this_month":
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        # Next month
        if start.month == 12:
            end = start.replace(year=start.year + 1, month=1)
        else:
            end = start.replace(month=start.month + 1)
        return {"$gte": start, "$lt": end}

    if filter_by == "this_year":
        start = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        end = start.replace(year=start.year + 1)
        return {"$gte": start, "$lt": end}

    return None


# ------------------------------
# Request models
# ------------------------------
class SendOtpRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)

class VerifyOtpRequest(BaseModel):
    email: EmailStr
    otp: str = Field(..., min_length=6, max_length=6)

class ServiceCreateRequest(BaseModel):
    email: EmailStr
    department: str
    state: str
    city: str
    pincode: str
    description: str

class ServiceUpdateRequest(BaseModel):
    email: EmailStr
    description: str

class TicketSolutionRequest(BaseModel):
    provider_email: EmailStr
    ticket_id: str
    solution: str

class DeleteFileRequest(BaseModel):
    provider_email: EmailStr
    document_name: str

# ------------------------------
# Endpoints
# ------------------------------
@router.post("/send_otp")
async def provider_send_otp(req: SendOtpRequest):
    existing_provider = await _db.providers.find_one({"email": req.email})
    if existing_provider:
        raise HTTPException(status_code=400, detail="Provider already registered")

    otp = _generate_otp()
    now = datetime.utcnow()
    await _db.otps.update_one(
        {"email": req.email, "type": "provider"},
        {"$set": {
            "email": req.email,
            "otp": otp,
            "created_at": now,
            "password": req.password,
            "type": "provider",
            "verified": False
        }},
        upsert=True
    )

    subject = "Your Smart Local Helpdesk OTP"
    html = f"<p>Your verification code is <b>{otp}</b>. It is valid for {_otps_ttl_seconds // 60} minutes.</p>"
    try:
        await _send_brevo_email(req.email, subject, html)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send OTP email: {e}")
    return {"success": True, "message": "OTP sent"}

@router.post("/verify_otp")
async def provider_verify_otp(req: VerifyOtpRequest):
    record = await _db.otps.find_one({"email": req.email, "type": "provider"})
    if not record:
        raise HTTPException(status_code=404, detail="No OTP request found for this email")

    if record.get("verified", False):
        raise HTTPException(status_code=400, detail="OTP already verified")

    if record.get("otp") != req.otp:
        raise HTTPException(status_code=400, detail="OTP mismatch")

    await _db.otps.update_one(
        {"email": req.email, "type": "provider"},
        {"$set": {"verified": True, "verified_at": datetime.utcnow()}}
    )
    return {"success": True, "message": "OTP verified"}

@router.post("/create_service")
async def provider_create_service(req: ServiceCreateRequest, background_tasks: BackgroundTasks):

    # step 1: verify OTP
    otp_doc = await _db.otps.find_one({"email": req.email, "type": "provider"})
    if not otp_doc or not otp_doc.get("verified", False):
        raise HTTPException(status_code=400, detail="OTP not verified for this email")

    # normalize location
    state_norm = _normalize_location(req.state)
    city_norm = _normalize_location(req.city)

    # step 2: check if service already exists
    existing_service = await _db.services.find_one({
        "department": req.department,
        "state": state_norm,
        "city": city_norm
    })
    if existing_service:
        raise HTTPException(status_code=400, detail="A service with same department/state/city already exists")

    # step 3: create store on GFAPI BEFORE creating provider or service
    store_name = f"{req.department.strip().lower()}_{req.city.strip().lower()}".replace(" ", "_")
    create_store_payload = {"api_key": GKEY, "store_name": store_name}

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{UPLOAD_BASE_URL}/create",
            json=create_store_payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        ) as resp:
            if resp.status >= 400:
                text = await resp.text()
                raise HTTPException(status_code=500, detail=f"Create store API failed: {resp.status} {text}")
            store_resp = await resp.json()

    search_store_name = store_resp.get("store_name")
    file_search_store_resource = store_resp.get("file_search_store_resource")

    # step 4: create service in DB FIRST (still no provider created yet)
    service_doc = {
        "provider_email": req.email,
        "department": req.department,
        "state": state_norm,
        "city": city_norm,
        "pincode": req.pincode,
        "description": req.description,
        "search_store": {
            "store_name": search_store_name,
            "file_search_store_resource": file_search_store_resource,
            "gkey": GKEY
        },
        "ticket_counts": {
            "today_ticket_count": 0,
            "pending_ticket_count": 0,
            "completed_ticket_count": 0
        },
        "created_at": datetime.utcnow()
    }

    await _db.services.insert_one(service_doc)

    # step 5: now create provider record 
    password = otp_doc.get("password")
    provider_payload = {
        "email": req.email,
        "password": password,
        "created_at": datetime.utcnow()
    }
    await _db.providers.insert_one(provider_payload)

    # step 6: update all users in that area
    background_tasks.add_task(
        _update_users_for_new_service,
        state_norm, city_norm, search_store_name
    )

    # step 7: send welcome email
    try:
        subject = "Welcome — Your service is registered"
        html = f"""
        <h3>Welcome to Smart Local Helpdesk</h3>
        <p>Your service in {city_norm}, {state_norm} has been registered successfully.</p>
        <p>Store name created: <b>{search_store_name}</b></p>
        """
        await _send_brevo_email(req.email, subject, html)
    except Exception as e:
        await _db.events.insert_one({
            "type": "provider_welcome_email_failed",
            "email": req.email,
            "error": str(e),
            "at": datetime.utcnow()
        })

    # step 8: finally delete OTP
    await _db.otps.delete_one({"email": req.email, "type": "provider"})

    return {
        "success": True,
        "message": "Service created successfully",
        "search_store": service_doc["search_store"]
    }

@router.post("/update_service")
async def provider_update_service(req: ServiceUpdateRequest):
    service = await _db.services.find_one({"provider_email": req.email})
    if not service:
        raise HTTPException(status_code=404, detail="Service not found for this email")

    await _db.services.update_one(
        {"provider_email": req.email},
        {"$set": {"description": req.description}}
    )
    return {"success": True, "message": "Service description updated"}

@router.get("/{provider_email}/tickets")
async def get_provider_tickets(provider_email: str):
    provider = await _db.providers.find_one({"email": provider_email})
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")

    tickets = await _db.tickets.find({"provider_email": provider_email}).to_list(length=None)

    pending_tickets = []
    completed_tickets = []

    for ticket in tickets:
        ticket_data = {
            "ticket_id": ticket.get("ticket_id"),
            "user_email": ticket.get("user_email"),
            "user_name": ticket.get("user_name"),
            "problem": ticket.get("problem"),
            "created_at": ticket.get("created_at"),
            "status": ticket.get("status"),
            "solution": ticket.get("solution", "")
        }
        if ticket.get("status") == "pending":
            pending_tickets.append(ticket_data)
        else:
            completed_tickets.append(ticket_data)

    # update counts before returning
    await _update_ticket_counts(provider_email)

    return {
        "success": True,
        "provider_email": provider_email,
        "ticket_summary": {
            "total_tickets": len(tickets),
            "pending_tickets": len(pending_tickets),
            "completed_tickets": len(completed_tickets)
        },
        "pending_tickets": pending_tickets,
        "completed_tickets": completed_tickets
    }

@router.post("/submit_solution")
async def submit_ticket_solution(req: TicketSolutionRequest, background_tasks: BackgroundTasks):
    ticket = await _db.tickets.find_one({"ticket_id": req.ticket_id})
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    if ticket.get("provider_email") != req.provider_email:
        raise HTTPException(status_code=403, detail="Not authorized to update this ticket")

    await _db.tickets.update_one(
        {"ticket_id": req.ticket_id},
        {"$set": {
            "solution": req.solution,
            "status": "completed",
            "completed_at": datetime.utcnow()
        }}
    )

    await _update_ticket_counts(req.provider_email)

    # schedule background email (async function is fine for BackgroundTasks)
    background_tasks.add_task(
        _send_ticket_completion_email,
        ticket.get("user_email"),
        ticket.get("user_name"),
        ticket.get("ticket_id"),
        ticket.get("problem")
    )

    return {
        "success": True,
        "message": "Solution submitted successfully",
        "ticket_id": req.ticket_id,
        "status": "completed"
    }

async def _send_ticket_completion_email(user_email: str, user_name: str, ticket_id: str, problem: str):
    try:
        subject = "Your Ticket Has Been Resolved - Smart Local Helpdesk"
        html = f"""
        <h3>Hello {user_name},</h3>
        <p>We're pleased to inform you that your service request has been successfully resolved.</p>
        <p><strong>Ticket ID:</strong> {ticket_id}</p>
        <p><strong>Problem:</strong> {problem}</p>
        <p>You can check the complete solution and details in your Smart Local Helpdesk application.</p>
        <p>Thank you for choosing our services!</p>
        <br>
        <p>Best regards,<br>Smart Local Helpdesk Team</p>
        """
        await _send_brevo_email(user_email, subject, html)
    except Exception as e:
        await _db.events.insert_one({
            "type": "ticket_completion_email_failed",
            "user_email": user_email,
            "ticket_id": ticket_id,
            "error": str(e),
            "at": datetime.utcnow()
        })



# ---------------------------------------------------------
# API 1: LIVE STREAM OF LATEST QUESTIONS (for websocket/polling)
# ---------------------------------------------------------
@router.get("/providers/{provider_email}/questions/live")
async def provider_live_questions(
    provider_email: str,
    since: Optional[str] = None,
    limit: int = 30
):
    # Check provider exists
    provider = await _db.providers.find_one({"email": provider_email})
    if not provider:
        raise HTTPException(404, "Provider not found")

    query = {"provider_email": provider_email}

    # If "since" timestamp is provided → return only NEW questions
    if since:
        try:
            since_dt = datetime.fromisoformat(since)
            query["asked_at"] = {"$gt": since_dt}
        except Exception:
            raise HTTPException(
                400,
                "Invalid since timestamp. Must be ISO, e.g. 2025-11-23T14:22:10"
            )

    logs = await _db.questions_asked.find(query) \
        .sort("asked_at", -1) \
        .limit(limit) \
        .to_list(None)

    # Clean ObjectIds
    for log in logs:
        log["_id"] = str(log["_id"])

    # Provide latest timestamp to help client maintain a cursor
    latest_ts = logs[0]["asked_at"] if logs else None

    return {
        "success": True,
        "provider_email": provider_email,
        "count": len(logs),
        "latest_timestamp": latest_ts,
        "questions": logs
    }

# ---------------------------------------------------------
# API 2: FILTERED QUESTIONS (today, yesterday, week, month, year)
# ---------------------------------------------------------
@router.get("/providers/{provider_email}/questions/filter")
async def provider_filtered_questions(
    provider_email: str,
    filter_by: Optional[str] = None
):
    # Check provider exists
    provider = await _db.providers.find_one({"email": provider_email})
    if not provider:
        raise HTTPException(404, "Provider not found")

    query = {"provider_email": provider_email}

    date_filter = _get_question_date_filter(filter_by)
    if date_filter:
        query["asked_at"] = date_filter

    logs = await _db.questions_asked.find(query) \
        .sort("asked_at", -1) \
        .to_list(None)

    for log in logs:
        log["_id"] = str(log["_id"])

    return {
        "success": True,
        "provider_email": provider_email,
        "filter": filter_by or "all",
        "count": len(logs),
        "questions": logs
    }





# upload files is here
@router.post("/upload_files")
async def upload_files(
    provider_email: str = Form(...),
    upload_type: str = Form(...),
    delete_at: Optional[str] = Form(None),
    files: List[UploadFile] = File(...)
):
    provider = await _db.providers.find_one({"email": provider_email})
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")

    if upload_type not in UPLOAD_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid upload type. Must be one of: {UPLOAD_TYPES}"
        )

    service = await _db.services.find_one({"provider_email": provider_email})
    if not service:
        raise HTTPException(status_code=404, detail="Service not found for this provider")

    store_name = service["search_store"]["store_name"]
    delete_time = _parse_delete_time(delete_at)

    uploaded_files = []

    # Sanitize ImageKit folder name (ImageKit rejects @ and .)
    imagekit_folder = (
        f"service_documents/{provider_email.replace('@','_at_').replace('.','_dot_')}"
    )

    for file in files:
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_extension} not allowed. Allowed: {ALLOWED_EXTENSIONS}"
            )

        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        temp_file_path = UPLOAD_DIR / unique_filename

        try:
            # -----------------------------------------------
            # SAVE TEMP FILE
            # -----------------------------------------------
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # -----------------------------------------------
            # 1️⃣ UPLOAD TO IMAGEKIT (OPTIONAL, NON-BLOCKING)
            # -----------------------------------------------
            imagekit_result = await _upload_to_imagekit(
                temp_file_path,
                file.filename,
                folder=imagekit_folder
            )

            if not imagekit_result.get("success"):
                # Log failure but DO NOT STOP GFAPI
                await _db.events.insert_one({
                    "type": "imagekit_upload_failed",
                    "provider_email": provider_email,
                    "filename": file.filename,
                    "folder": imagekit_folder,
                    "error": imagekit_result.get("error"),
                    "at": datetime.utcnow()
                })

                # Default failure-safe structure
                imagekit_result = {
                    "success": False,
                    "file_id": None,
                    "url": None,
                    "thumbnail_url": None
                }

            # -----------------------------------------------
            # 2️⃣ UPLOAD TO GFAPI (MANDATORY)
            # -----------------------------------------------
            gfapi_result = await _upload_to_gfapi(
                store_name,
                temp_file_path,
                file.filename
            )

            if not gfapi_result.get("success"):
                # Rollback ImageKit only if ImageKit succeeded
                if imagekit_result.get("success") and imagekit_result.get("file_id"):
                    await _delete_from_imagekit(imagekit_result.get("file_id"))

                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to upload {file.filename} to GFAPI: {gfapi_result.get('error')}"
                )

            # -----------------------------------------------
            # 3️⃣ STORE RECORD IN DB
            # -----------------------------------------------
            file_doc = {
                "provider_email": provider_email,
                "store_name": store_name,
                "filename": file.filename,
                "uploaded_filename": gfapi_result.get("filename"),
                "document_id": gfapi_result.get("document_id"),
                "document_resource": gfapi_result.get("document_resource"),

                # ImageKit
                "imagekit_file_id": imagekit_result.get("file_id"),
                "imagekit_url": imagekit_result.get("url"),
                "imagekit_thumbnail_url": imagekit_result.get("thumbnail_url"),

                "upload_type": upload_type,
                "file_size": temp_file_path.stat().st_size,
                "file_extension": file_extension,
                "delete_at": delete_time,
                "created_at": datetime.utcnow()
            }

            await _db.uploads.insert_one(file_doc)

            # -----------------------------------------------
            # 4️⃣ RESPONSE AGGREGATION
            # -----------------------------------------------
            uploaded_files.append({
                "original_name": file.filename,
                "uploaded_name": gfapi_result.get("filename"),
                "document_id": gfapi_result.get("document_id"),
                "imagekit_file_id": imagekit_result.get("file_id"),
                "imagekit_url": imagekit_result.get("url"),
                "delete_at": delete_time,
                "size": file_doc["file_size"],
                "upload_type": upload_type
            })

        finally:
            # ALWAYS CLEAN TEMP
            if temp_file_path.exists():
                try:
                    temp_file_path.unlink()
                except:
                    pass

    return {
        "success": True,
        "message": f"Successfully uploaded {len(uploaded_files)} files",
        "uploaded_files": uploaded_files,
        "store_name": store_name,
        "storage": {
            "gfapi_used": True,
            "imagekit_used": bool(imagekit)
        }
    }

@router.delete("/delete_file")
async def delete_file(req: DeleteFileRequest):
    file_doc = await _db.uploads.find_one({
        "provider_email": req.provider_email,
        "filename": req.document_name
    })

    if not file_doc:
        raise HTTPException(status_code=404, detail="File not found")

    gfapi_result = await _delete_file_from_gfapi(file_doc['store_name'], file_doc['document_id'])
    imagekit_result = {'success': True}
    if file_doc.get('imagekit_file_id'):
        imagekit_result = await _delete_from_imagekit(file_doc['imagekit_file_id'])

    try:
        await _db.uploads.delete_one({"_id": file_doc["_id"]})
    except Exception:
        logger.exception("Failed to delete upload record from DB after remote deletion")

    if gfapi_result.get('success') and imagekit_result.get('success'):
        return {
            "success": True,
            "message": "File deleted successfully from both storage systems",
            "filename": req.document_name,
            "deleted_from": ["gfapi", "imagekit"]
        }
    else:
        await _db.events.insert_one({
            "type": "partial_delete_failure",
            "provider_email": req.provider_email,
            "filename": req.document_name,
            "gfapi": gfapi_result,
            "imagekit": imagekit_result,
            "at": datetime.utcnow()
        })
        raise HTTPException(status_code=500, detail=f"Partial deletion: GFAPI: {gfapi_result.get('error', 'success')}, ImageKit: {imagekit_result.get('error', 'success')}")

@router.get("/{provider_email}/files")
async def get_provider_files(provider_email: str, upload_type: Optional[str] = None):
    provider = await _db.providers.find_one({"email": provider_email})
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")

    query = {"provider_email": provider_email}
    if upload_type and upload_type in UPLOAD_TYPES:
        query["upload_type"] = upload_type

    files = await _db.uploads.find(query).sort("created_at", -1).to_list(length=None)

    files_by_type = {}
    for file in files:
        ut = file.get("upload_type", "other")
        files_by_type.setdefault(ut, [])
        files_by_type[ut].append({
            "filename": file.get("filename"),
            "document_id": file.get("document_id"),
            "file_size": file.get("file_size"),
            "file_extension": file.get("file_extension"),
            "imagekit_url": file.get("imagekit_url"),
            "imagekit_thumbnail_url": file.get("imagekit_thumbnail_url"),
            "download_url": file.get("imagekit_url"),
            "created_at": file.get("created_at"),
            "delete_at": file.get("delete_at")
        })

    return {
        "success": True,
        "provider_email": provider_email,
        "files_by_type": files_by_type,
        "total_files": len(files),
        "storage_provider": "ImageKit.io + GFAPI"
    }

