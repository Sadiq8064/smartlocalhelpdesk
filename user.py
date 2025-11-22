# user.py — Production Ready (Brevo via sib_api_v3_sdk, non-blocking)
import json
import os
import uuid
import random
import string
import asyncio
import logging
from datetime import datetime
from typing import Optional, List, Dict
from bson import ObjectId

import sib_api_v3_sdk
from sib_api_v3_sdk.rest import ApiException

import aiohttp  # still used for GFAPI calls
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr, Field

# Try importing google-genai SDK
try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None


# -----------------------------------------------------------
# HARD-CODED API KEYS (from env)
# -----------------------------------------------------------
BREVO_API_KEY = os.getenv("BREVO_API_KEY")
GKEY = os.getenv("GFAPI_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ASK_QUESTION_URL = os.getenv("ASK_QUESTION_URL")

# Validate required envs
if not BREVO_API_KEY:
    raise ValueError("BREVO_API_KEY environment variable is required")
if not GKEY:
    raise ValueError("GFAPI_KEY environment variable is required")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")
if not ASK_QUESTION_URL:
    raise ValueError("ASK_QUESTION_URL environment variable is required")

OTP_TTL_SECONDS = 300  # 5 minutes

# -----------------------------------------------------------
router = APIRouter()
_db = None
logger = logging.getLogger("user_routes")
logging.basicConfig(level=logging.INFO)


# -----------------------------------------------------------
# Initialization
# -----------------------------------------------------------
def init_user_routes(app, db):
    global _db
    _db = db


# -----------------------------------------------------------
# Utility Helpers
# -----------------------------------------------------------
def _generate_otp():
    return "".join(random.choices(string.digits, k=6))


def _generate_session_id():
    return str(uuid.uuid4())


def _generate_ticket_id():
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


def _normalize_location(value: str) -> str:
    """Normalize for case-insensitive matching."""
    return " ".join(word.capitalize() for word in value.strip().split())


# -----------------------------------------------------------
# Email Sender (Brevo) — non-blocking wrapper around sib_api_v3_sdk
# -----------------------------------------------------------
async def _send_brevo_email(to_email: str, subject: str, html_content: str):
    """
    Sends an email using Brevo (sib_api_v3_sdk) without blocking the event loop.
    Runs the blocking SDK call inside run_in_executor.
    Returns parsed JSON-like response on success or raises RuntimeError on failure.
    """
    def _sync_send():
        configuration = sib_api_v3_sdk.Configuration()
        configuration.api_key["api-key"] = BREVO_API_KEY
        api_client = sib_api_v3_sdk.ApiClient(configuration)
        api_instance = sib_api_v3_sdk.TransactionalEmailsApi(api_client)

        email_data = sib_api_v3_sdk.SendSmtpEmail(
            to=[{"email": to_email}],
            sender={"name": "Smart Local Helpdesk", "email": "mrsadiq471@gmail.com"},
            subject=subject,
            html_content=html_content,
            text_content="This is an automated email from Smart Local Helpdesk."
        )

        try:
            resp = api_instance.send_transac_email(email_data)
            # The SDK returns an object — convert minimal useful fields to dict
            return {"success": True, "message_id": getattr(resp, "message_id", None), "raw": resp}
        except ApiException as e:
            # ApiException has body and status — include details
            raise RuntimeError(
                f"Brevo ApiException: status={getattr(e, 'status', None)} "
                f"body={getattr(e, 'body', None)}"
            )

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(None, _sync_send)
        return result
    except Exception as e:
        logger.exception("Brevo send failed: %s", e)
        raise


# -----------------------------------------------------------
# Fetch services for a user's location
# -----------------------------------------------------------
async def _get_services_for_location(state: str, city: str) -> List[str]:
    state = _normalize_location(state)
    city = _normalize_location(city)

    try:
        services = await _db.services.find({"state": state, "city": city}).to_list(None)
        stores = []

        for s in services:
            st = s.get("search_store", {})
            if st.get("store_name"):
                stores.append(st["store_name"])

        return stores

    except Exception as e:
        logger.exception("Error searching services: %s", e)
        return []


# -----------------------------------------------------------
# Gemini (google-genai SDK) Helpers
# -----------------------------------------------------------
def _init_gemini_client_sync(api_key: str):
    if genai is None:
        raise RuntimeError("google-genai SDK not installed.")
    return genai.Client(api_key=api_key)


def _call_gemini_sync(client, stores: List[str], question: str) -> str:
    """
    Synchronous helper to call Gemini model for store selection and question splitting.
    Returns raw text (expected to be JSON).
    """
    model = "gemini-2.5-flash"

    # New routing + splitting system prompt
    system_prompt = f"""
You are a classifier that analyzes a user's question and determines:

1. Which government service departments (from the list below) can answer which parts of the question.
2. A single question may contain:
   - One relevant part
   - Multiple relevant parts for different departments
   - Irrelevant parts that no department can answer

Your task:
---------------------
Given the available stores (departments):
{stores}

And the user's question:
"{question}"

You MUST break the question into meaningful parts and assign them as follows:

1. For every relevant part:
   - Identify which store (department) it belongs to.
   - Rewrite the question part only for clarity, keeping EXACT meaning.
   - Return it in `split_questions` under that store name.

2. If a question part belongs to multiple stores:
   - Add the same rewritten part under all relevant stores.

3. If a question part does NOT belong to any store:
   - Include it in an "unanswered" array with a reason like:
       "No service department can answer this."

IMPORTANT RULES:
---------------------
- DO NOT change the user's original intent.
- DO NOT add or remove information.
- Only restructure the language for clarity.
- You must always return valid JSON exactly in this format:

{{
  "stores": ["store1", "store2"],
  "split_questions": {{
      "store1": "rewritten question part for store1",
      "store2": "rewritten question part for store2"
  }},
  "unanswered": [
      {{
         "text": "part of question that no store can answer",
         "reason": "why no department can answer"
      }}
  ]
}}

NOTES:
---------------------
- If no part of the question belongs to any store, return:
  {{
    "stores": [],
    "split_questions": {{}},
    "unanswered": [
      {{
        "text": "{question}",
        "reason": "No service department can answer this"
      }}
    ]
  }}
- You MUST return valid JSON ONLY. No explanation, no extra text.
"""

    response = client.models.generate_content(
        model=model,
        contents=question,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.0
        )
    )

    txt = getattr(response, "text", None)
    if not txt and getattr(response, "candidates", None):
        c = response.candidates[0]
        txt = getattr(c, "text", None) or getattr(c, "content", None)

    return txt or ""


async def _call_gemini_for_store_selection(stores: List[str], question: str) -> Dict:
    """
    Runs Gemini SDK inside a threadpool, extracts JSON safely.
    Returns:
    {
      "stores": [...],
      "split_questions": {store: question_part},
      "unanswered": [ { "text": "...", "reason": "..." }, ... ]
    }

    IMPORTANT:
    - If Gemini fails or is missing → fall back to all stores (no splitting).
    - Only when Gemini explicitly returns "stores": [] we treat as "no service can answer".
    """
    if not stores:
        return {"stores": [], "split_questions": {}, "unanswered": []}

    if genai is None:
        logger.warning("Gemini SDK missing => returning all stores without splitting.")
        return {"stores": stores, "split_questions": {}, "unanswered": []}

    loop = asyncio.get_running_loop()

    try:
        client = await loop.run_in_executor(None, _init_gemini_client_sync, GEMINI_API_KEY)
        raw = await loop.run_in_executor(None, _call_gemini_sync, client, stores, question)
        if not raw:
            logger.warning("Gemini returned empty response => using all stores.")
            return {"stores": stores, "split_questions": {}, "unanswered": []}

        raw = raw.strip()

        # direct JSON parse
        try:
            parsed = json.loads(raw)
            return {
                "stores": parsed.get("stores", []) or [],
                "split_questions": parsed.get("split_questions", {}) or {},
                "unanswered": parsed.get("unanswered", []) or []
            }
        except Exception:
            # Attempt to extract JSON substring
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1:
                try:
                    parsed = json.loads(raw[start:end + 1])
                    return {
                        "stores": parsed.get("stores", []) or [],
                        "split_questions": parsed.get("split_questions", {}) or {},
                        "unanswered": parsed.get("unanswered", []) or []
                    }
                except Exception:
                    logger.exception("Gemini JSON substring parse failed.")

        # If everything fails, fall back to all stores
        logger.warning("Gemini parsing failed => using all stores.")
        return {"stores": stores, "split_questions": {}, "unanswered": []}

    except Exception as e:
        logger.exception("Gemini failed: %s", e)
        # On failure, still try all stores instead of blocking the flow
        return {"stores": stores, "split_questions": {}, "unanswered": []}


# -----------------------------------------------------------
# GFAPI RAG Caller
# -----------------------------------------------------------
async def _call_rag_api(store: str, question: str):
    payload = {
        "api_key": GKEY,
        "stores": [store],
        "question": question,
        "system_prompt": (
            "You are a bot assisting Indian citizens. "
            "Answer ONLY using File Search documents. "
            "If info missing, say: 'Sorry, no information available. Would you like to create a ticket?'"
        )
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(ASK_QUESTION_URL, json=payload) as resp:
            text = await resp.text()
            if resp.status != 200:
                logger.error("GFAPI error %s %s", resp.status, text)
                return {"error": True, "detail": text}
            return await resp.json()


# -----------------------------------------------------------
# Conversation history storage
# -----------------------------------------------------------
async def _store_conversation(session_id: str, email: str, question: str, resp: Dict):
    try:
        await _db.conversations.insert_one({
            "session_id": session_id,
            "user_email": email,
            "question": question,
            "response": resp,
            "created_at": datetime.utcnow()
        })
    except Exception as e:
        logger.exception("Failed saving conversation: %s", e)


# -----------------------------------------------------------
# Pydantic Models
# -----------------------------------------------------------
class UserSendOtp(BaseModel):
    email: EmailStr
    password: str


class UserVerifyOtp(BaseModel):
    email: EmailStr
    otp: str


class UserCompleteRegistration(BaseModel):
    email: EmailStr
    name: str
    state: str
    city: str
    profile_pic: Optional[str] = None


class UserUpdateLocation(BaseModel):
    email: EmailStr
    state: str
    city: str


class CreateTicketRequest(BaseModel):
    user_email: EmailStr
    user_name: str
    provider_email: EmailStr
    problem: str


class AskQuestionRequest(BaseModel):
    email: EmailStr
    question: str
    session_id: Optional[str] = None


class CreateFeedbackRequest(BaseModel):
    user_email: EmailStr
    provider_email: EmailStr
    rating: int = Field(..., ge=1, le=5)
    feedback_text: Optional[str] = None


class UpdateFeedbackRequest(BaseModel):
    user_email: EmailStr
    feedback_id: str
    rating: Optional[int] = Field(None, ge=1, le=5)
    feedback_text: Optional[str] = None


class DeleteFeedbackRequest(BaseModel):
    user_email: EmailStr
    feedback_id: str


# -----------------------------------------------------------
#  ENDPOINTS
# -----------------------------------------------------------

# ----------------- SEND OTP -----------------
@router.post("/send_otp")
async def user_send_otp(req: UserSendOtp):
    exists = await _db.users.find_one({"email": req.email})
    if exists:
        raise HTTPException(400, "User already registered")

    otp = _generate_otp()

    await _db.otps.update_one(
        {"email": req.email, "type": "user"},
        {"$set": {
            "email": req.email,
            "otp": otp,
            "password": req.password,
            "type": "user",
            "created_at": datetime.utcnow(),
            "verified": False
        }},
        upsert=True
    )

    # send email (non-blocking)
    await _send_brevo_email(
        req.email,
        "Your Smart Local Helpdesk OTP",
        f"<p>Your OTP is <b>{otp}</b>. Valid for 5 mins.</p>"
    )

    return {"success": True, "message": "OTP sent"}


# -----------------------------------------------------------
# CREATE FEEDBACK
# -----------------------------------------------------------
@router.post("/feedback/create")
async def create_feedback(req: CreateFeedbackRequest):
    # Fetch user
    user = await _db.users.find_one({"email": req.user_email})
    if not user:
        raise HTTPException(404, "User not found")

    # Fetch provider
    provider = await _db.providers.find_one({"email": req.provider_email})
    if not provider:
        raise HTTPException(404, "Provider not found")

    # Fetch state & city from USER (not from request)
    state = user["state"]
    city = user["city"]

    # Get provider details
    department_name = provider["department_name"]
    provider_name = provider["name"]

    feedback_doc = {
        "user_email": req.user_email,
        "user_name": user["name"],
        "state": state,
        "city": city,
        "provider_email": req.provider_email,
        "provider_name": provider_name,
        "department_name": department_name,
        "rating": req.rating,
        "feedback_text": req.feedback_text or "",
        "created_at": datetime.utcnow(),
        "updated_at": None
    }

    result = await _db.feedback.insert_one(feedback_doc)

    return {
        "success": True,
        "message": "Feedback created successfully",
        "feedback_id": str(result.inserted_id)
    }


# -----------------------------------------------------------
# GET ALL FEEDBACK FOR A PROVIDER
# -----------------------------------------------------------
@router.get("/feedback")
async def provider_feedback(provider_email: str):
    # Check if provider exists
    provider = await _db.providers.find_one({"email": provider_email})
    if not provider:
        raise HTTPException(404, "Provider not found")

    provider_name = provider["name"]
    department_name = provider["department_name"]
    state = provider["state"]
    city = provider["city"]

    # Fetch feedback from feedback collection
    feedbacks = await _db.feedback.find({
        "provider_email": provider_email
    }).sort("created_at", -1).to_list(None)

    total = len(feedbacks)
    avg_rating = round(
        (sum(f["rating"] for f in feedbacks) / total) if total > 0 else 0, 2
    )

    return {
        "success": True,
        "provider_name": provider_name,
        "department_name": department_name,
        "state": state,
        "city": city,
        "total_feedbacks": total,
        "average_rating": avg_rating,
        "feedbacks": [
            {
                "feedback_id": str(f["_id"]),
                "user_email": f["user_email"],
                "user_name": f["user_name"],
                "rating": f["rating"],
                "feedback_text": f["feedback_text"],
                "created_at": f["created_at"],
                "updated_at": f.get("updated_at")
            }
            for f in feedbacks
        ]
    }


# -----------------------------------------------------------
# UPDATE FEEDBACK
# -----------------------------------------------------------
@router.post("/feedback/update")
async def update_feedback(req: UpdateFeedbackRequest):
    # Convert id
    try:
        fid = ObjectId(req.feedback_id)
    except Exception:
        raise HTTPException(400, "Invalid feedback ID")

    fb = await _db.feedback.find_one({"_id": fid})
    if not fb:
        raise HTTPException(404, "Feedback not found")

    # Ensure user owns the feedback
    if fb["user_email"] != req.user_email:
        raise HTTPException(403, "You cannot edit someone else's feedback")

    updates = {}
    if req.rating is not None:
        updates["rating"] = req.rating
    if req.feedback_text is not None:
        updates["feedback_text"] = req.feedback_text

    updates["updated_at"] = datetime.utcnow()

    await _db.feedback.update_one({"_id": fid}, {"$set": updates})

    return {
        "success": True,
        "message": "Feedback updated successfully"
    }


# -----------------------------------------------------------
# DELETE FEEDBACK
# -----------------------------------------------------------
@router.post("/feedback/delete")
async def delete_feedback(req: DeleteFeedbackRequest):
    try:
        fid = ObjectId(req.feedback_id)
    except Exception:
        raise HTTPException(400, "Invalid feedback ID")

    fb = await _db.feedback.find_one({"_id": fid})
    if not fb:
        raise HTTPException(404, "Feedback not found")

    # Ensure user owns the feedback
    if fb["user_email"] != req.user_email:
        raise HTTPException(403, "You cannot delete someone else's feedback")

    await _db.feedback.delete_one({"_id": fid})

    return {
        "success": True,
        "message": "Feedback deleted successfully"
    }


# ----------------- VERIFY OTP -----------------
@router.post("/verify_otp")
async def user_verify_otp(req: UserVerifyOtp):
    otp_doc = await _db.otps.find_one({"email": req.email, "type": "user"})
    if not otp_doc:
        raise HTTPException(404, "No OTP request found")

    if otp_doc.get("verified"):
        raise HTTPException(400, "OTP already verified")

    if otp_doc["otp"] != req.otp:
        raise HTTPException(400, "OTP mismatch")

    await _db.otps.update_one(
        {"email": req.email, "type": "user"},
        {"$set": {"verified": True}}
    )

    return {"success": True, "message": "OTP verified"}


# ----------------- COMPLETE REGISTRATION -----------------
@router.post("/complete_registration")
async def user_complete_registration(req: UserCompleteRegistration):
    otp_doc = await _db.otps.find_one({"email": req.email, "type": "user"})
    if not otp_doc or not otp_doc.get("verified"):
        raise HTTPException(400, "OTP not verified")

    if await _db.users.find_one({"email": req.email}):
        raise HTTPException(400, "User already registered")

    state = _normalize_location(req.state)
    city = _normalize_location(req.city)

    stores = await _get_services_for_location(state, city)

    user_doc = {
        "email": req.email,
        "password": otp_doc.get("password"),
        "name": req.name,
        "state": state,
        "city": city,
        "profile_pic": req.profile_pic,
        "available_service_stores": stores,
        "created_at": datetime.utcnow()
    }

    await _db.users.insert_one(user_doc)
    await _db.otps.delete_one({"email": req.email, "type": "user"})

    return {
        "success": True,
        "message": "User registration completed successfully",
        "available_services": len(stores),
        "user_details": {
            "name": req.name,
            "email": req.email,
            "state": state,
            "city": city
        }
    }


# ----------------- UPDATE LOCATION -----------------
@router.post("/update_location")
async def user_update_location(req: UserUpdateLocation):
    user = await _db.users.find_one({"email": req.email})
    if not user:
        raise HTTPException(404, "User not found")

    state = _normalize_location(req.state)
    city = _normalize_location(req.city)

    if user["state"] == state and user["city"] == city:
        return {
            "success": True,
            "message": "Location unchanged",
            "available_services": len(user.get("available_service_stores", []))
        }

    stores = await _get_services_for_location(state, city)

    await _db.users.update_one(
        {"email": req.email},
        {"$set": {
            "state": state,
            "city": city,
            "available_service_stores": stores,
            "location_updated_at": datetime.utcnow()
        }}
    )

    return {
        "success": True,
        "message": "Location updated",
        "available_services": len(stores),
        "available_service_stores": stores
    }


# ----------------- LIST USER SERVICES -----------------
@router.get("/{email}/services")
async def get_user_services(email: str):
    user = await _db.users.find_one({"email": email})
    if not user:
        raise HTTPException(404, "User not found")

    return {
        "success": True,
        "user": {
            "name": user["name"],
            "email": email,
            "state": user["state"],
            "city": user["city"],
            "profile_pic": user.get("profile_pic"),
            "available_service_stores": user.get("available_service_stores", []),
            "total_services": len(user.get("available_service_stores", []))
        }
    }


# ----------------- CREATE TICKET -----------------
@router.post("/create_ticket")
async def create_ticket(req: CreateTicketRequest):
    if not await _db.users.find_one({"email": req.user_email}):
        raise HTTPException(404, "User not found")

    if not await _db.providers.find_one({"email": req.provider_email}):
        raise HTTPException(404, "Provider not found")

    ticket_id = _generate_ticket_id()

    ticket_doc = {
        "ticket_id": ticket_id,
        "user_email": req.user_email,
        "user_name": req.user_name,
        "provider_email": req.provider_email,
        "problem": req.problem,
        "status": "pending",
        "created_at": datetime.utcnow(),
        "solution": "",
        "completed_at": None
    }

    await _db.tickets.insert_one(ticket_doc)

    # update provider ticket counts
    from service_provider import _update_ticket_counts
    await _update_ticket_counts(req.provider_email)

    # --- SEND EMAIL TO USER ---
    try:
        await _send_brevo_email(
            req.user_email,
            f"Ticket Created Successfully #{ticket_id}",
            f"""
            <p>Hello {req.user_name},</p>
            <p>Your ticket has been created successfully.</p>
            <p><b>Ticket ID:</b> {ticket_id}</p>
            <p><b>Provider:</b> {req.provider_email}</p>
            <p><b>Problem:</b> {req.problem}</p>
            <p>We will notify you when the provider responds.</p>
            <br>
            <p>Smart Local Helpdesk</p>
            """
        )
    except Exception as e:
        logger.error("Failed to send ticket creation email: %s", e)

    return {
        "success": True,
        "message": "Ticket created",
        "ticket_id": ticket_id
    }


# ----------------- LIST USER TICKETS -----------------
@router.get("/{email}/tickets")
async def get_user_tickets(email: str):
    if not await _db.users.find_one({"email": email}):
        raise HTTPException(404, "User not found")

    tickets = await _db.tickets.find({"user_email": email}).to_list(None)

    pending = []
    completed = []

    for t in tickets:
        tid = {
            "ticket_id": t["ticket_id"],
            "provider_email": t["provider_email"],
            "problem": t["problem"],
            "status": t["status"],
            "solution": t["solution"],
            "created_at": t["created_at"],
            "completed_at": t["completed_at"]
        }
        if t["status"] == "pending":
            pending.append(tid)
        else:
            completed.append(tid)

    return {
        "success": True,
        "pending_tickets": pending,
        "completed_tickets": completed
    }


# ----------------- ASK QUESTION (RAG) -----------------
@router.post("/ask_question")
async def ask_question(req: AskQuestionRequest):
    user = await _db.users.find_one({"email": req.email})
    if not user:
        raise HTTPException(404, "User not found")

    session_id = req.session_id or _generate_session_id()
    stores = user.get("available_service_stores", [])

    if not stores:
        resp = {
            "success": True,
            "session_id": session_id,
            "response": "No services available in your area.",
            "stores_used": [],
            "sources": [],
            "unanswered_parts": []
        }
        await _store_conversation(session_id, req.email, req.question, resp)
        return resp

    # --- Predict store relevance using Gemini ---
    gemini_result = await _call_gemini_for_store_selection(stores, req.question)
    selected_stores = gemini_result.get("stores", []) or []
    split_q = gemini_result.get("split_questions", {}) or {}
    unanswered_parts = gemini_result.get("unanswered", []) or []

    # OPTION A: If Gemini explicitly finds no matching store → return immediately
    if not selected_stores:
        resp_text = "Sorry, no available service can answer this question."
        if unanswered_parts:
            # Make response a bit more informative, but frontend still has structured data
            extra = []
            for part in unanswered_parts:
                txt = part.get("text") or ""
                reason = part.get("reason") or ""
                if txt:
                    extra.append(f"- \"{txt}\" ({reason})" if reason else f"- \"{txt}\"")
            if extra:
                resp_text += "\n\nDetails:\n" + "\n".join(extra)

        resp = {
            "success": True,
            "session_id": session_id,
            "response": resp_text,
            "stores_used": [],
            "sources": [],
            "unanswered_parts": unanswered_parts
        }
        await _store_conversation(session_id, req.email, req.question, resp)
        return resp

    final_answers = []
    all_sources = []

    # Call RAG only for selected stores
    for store in selected_stores:
        q = split_q.get(store, req.question)
        rag = await _call_rag_api(store, q)

        if rag.get("error"):
            ans = "Sorry, I could not retrieve information."
            sources = []
        else:
            ans = rag.get("response_text", "") or ""
            grounding = rag.get("grounding_metadata", {}) or {}
            chunks = grounding.get("groundingChunks", [])
            sources = []

            for c in chunks:
                ctx = c.get("retrievedContext", {})
                if ctx.get("text"):
                    sources.append(ctx["text"])

        final_answers.append({
            "store": store,
            "question": q,
            "answer": ans,
            "sources": sources
        })
        all_sources.extend(sources)

    # Combine multi-store answers into a single text response
    if len(final_answers) == 1:
        final_text = final_answers[0]["answer"]
    else:
        parts = []
        for fa in final_answers:
            parts.append(f"**{fa['store']}**:\n{fa['answer']}")
        final_text = "\n\n".join(parts)

    # Append info about unanswered parts (if any)
    if unanswered_parts:
        extra_lines = []
        for part in unanswered_parts:
            txt = part.get("text") or ""
            reason = part.get("reason") or ""
            if not txt:
                continue
            if reason:
                extra_lines.append(f"- \"{txt}\" ({reason})")
            else:
                extra_lines.append(f"- \"{txt}\"")
        if extra_lines:
            final_text += (
                "\n\nThe following parts of your question could not be answered "
                "by any available service department:\n" + "\n".join(extra_lines)
            )

    resp = {
        "success": True,
        "session_id": session_id,
        "response": final_text,
        "stores_used": selected_stores,
        "sources": all_sources[:1],
        # "detailed": final_answers,
        "unanswered_parts": unanswered_parts
    }

    await _store_conversation(session_id, req.email, req.question, resp)
    return resp


# ----------------- FETCH USER CONVERSATIONS -----------------
@router.get("/{email:path}/conversations")
async def get_user_conversations(email: str):
    # 1️⃣ Fetch user
    user = await _db.users.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # 2️⃣ Fetch conversation docs
    convos = await _db.conversations.find(
        {"user_email": email}
    ).sort("created_at", -1).to_list(length=None)

    # 3️⃣ Convert ObjectId → string
    cleaned = []
    for c in convos:
        c["_id"] = str(c["_id"])
        # If conversation has messages
        if "messages" in c:
            for m in c["messages"]:
                if "_id" in m:
                    m["_id"] = str(m["_id"])
        cleaned.append(c)

    # 4️⃣ Return perfectly JSON serializable output
    return {
        "success": True,
        "count": len(cleaned),
        "conversations": cleaned
    }
