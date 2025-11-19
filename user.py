# user.py — Production Ready (Hard-coded API keys as requested)

import json
import os  # Add this if not already imported
import uuid
import random
import string
import asyncio
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

import aiohttp
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
# HARD-CODED API KEYS (as you requested)
# -----------------------------------------------------------

# Replace with:
BREVO_API_KEY = os.getenv("BREVO_API_KEY")
GKEY = os.getenv("GFAPI_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ASK_QUESTION_URL = os.getenv("ASK_QUESTION_URL")

# Add validation after these lines:
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
# Email Sender (Brevo)
# -----------------------------------------------------------

async def _send_brevo_email(to_email: str, subject: str, html_content: str):
    headers = {
        "api-key": BREVO_API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "sender": {"name": "Smart Local Helpdesk", "email": "no-reply@brevo.com"},
        "to": [{"email": to_email}],
        "subject": subject,
        "htmlContent": html_content
    }

    async with aiohttp.ClientSession() as session:
        async with session.post("https://api.brevo.com/v3/smtp/email",
                                json=payload, headers=headers) as resp:
            text = await resp.text()
            if resp.status >= 400:
                logger.error("Brevo failed: %s %s", resp.status, text)
                raise RuntimeError("Failed sending email")
            return await resp.json()


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
    model = "gemini-2.5-flash"

    system_prompt = f"""
Given:
Available Stores = {stores}
User Question = "{question}"

Return ONLY valid JSON:
{{
  "stores": ["store1", "store2"],
  "split_questions": {{
      "store1": "q1",
      "store2": "q2"
  }}
}}
If no store relevant: {{"stores":[]}}
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


async def _call_gemini_for_store_selection(stores: List[str], question: str):
    """Runs Gemini SDK inside a threadpool, extracts JSON safely."""
    if not stores:
        return {"stores": []}

    if genai is None:
        logger.warning("Gemini SDK missing => returning all stores.")
        return {"stores": stores}

    loop = asyncio.get_running_loop()

    try:
        client = await loop.run_in_executor(None, _init_gemini_client_sync, GEMINI_API_KEY)

        raw = await loop.run_in_executor(None, _call_gemini_sync, client, stores, question)
        if not raw:
            return {"stores": stores}

        raw = raw.strip()

        # direct JSON parse
        try:
            parsed = json.loads(raw)
            return {
                "stores": parsed.get("stores", []),
                "split_questions": parsed.get("split_questions", {})
            }
        except:
            # Attempt to extract JSON substring
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1:
                try:
                    parsed = json.loads(raw[start:end + 1])
                    return {
                        "stores": parsed.get("stores", []),
                        "split_questions": parsed.get("split_questions", {})
                    }
                except:
                    pass

        return {"stores": stores}

    except Exception as e:
        logger.exception("Gemini failed: %s", e)
        return {"stores": stores}


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

    await _send_brevo_email(
        req.email,
        "Your Smart Local Helpdesk OTP",
        f"<p>Your OTP is <b>{otp}</b>. Valid for 5 mins.</p>"
    )

    return {"success": True, "message": "OTP sent"}


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

    await _db.otps.update_one({"email": req.email, "type": "user"},
                              {"$set": {"verified": True}})

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
        }
        await _store_conversation(session_id, req.email, req.question, resp)
        return resp

    # --- Predict store relevance using Gemini ---
    gemini_result = await _call_gemini_for_store_selection(stores, req.question)
    selected_stores = gemini_result.get("stores") or stores
    split_q = gemini_result.get("split_questions", {})

    final_answers = []
    all_sources = []

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
            "answer": ans,
            "sources": sources
        })
        all_sources.extend(sources)

    # Combine multi-store answers
    if len(final_answers) == 1:
        final_text = final_answers[0]["answer"]
    else:
        final_text = ""
        for fa in final_answers:
            final_text += f"**{fa['store']}**: {fa['answer']}\n\n"

    resp = {
        "success": True,
        "session_id": session_id,
        "response": final_text,
        "stores_used": selected_stores,
        "sources": all_sources[:5],
        "detailed": final_answers
    }

    await _store_conversation(session_id, req.email, req.question, resp)
    return resp


# ----------------- FETCH USER CONVERSATIONS -----------------
@router.get("/{email}/conversations")
async def get_user_conversations(email: str, session_id: Optional[str] = None):
    if not await _db.users.find_one({"email": email}):
        raise HTTPException(404, "User not found")

    query = {"user_email": email}
    if session_id:
        query["session_id"] = session_id

    convs = await _db.conversations.find(query).sort("created_at").to_list(None)

    sessions = {}
    for c in convs:
        sid = c["session_id"]
        if sid not in sessions:
            sessions[sid] = []
        sessions[sid].append(c)

    return {
        "success": True,
        "sessions": sessions,
        "total_sessions": len(sessions)
    }
