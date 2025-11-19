# delete.py
import aiohttp
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr
from datetime import datetime

router = APIRouter()
_db = None

# Get from environment variables
GKEY = os.getenv("GFAPI_KEY")
GFAPI_BASE = os.getenv("GFAPI_BASE_URL")

if not GKEY:
    raise ValueError("GFAPI_KEY environment variable is required")
if not GFAPI_BASE:
    raise ValueError("GFAPI_BASE_URL environment variable is required")


def init_delete_routes(app, db):
    global _db
    _db = db
    app.include_router(router, prefix="/delete", tags=["delete"])


# --------------------------------------------------------
# Utility: Delete entire store from GFAPI
# --------------------------------------------------------
async def delete_store_from_gfapi(store_name: str):
    url = f"{GFAPI_BASE}/{store_name}?api_key={GKEY}"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.delete(url) as resp:
                text = await resp.text()

                if resp.status == 200:
                    return True, text

                return False, text

    except Exception as e:
        return False, str(e)


# --------------------------------------------------------
# 1️⃣ DELETE ALL SERVICES + THEIR TICKETS + THEIR STORES
# --------------------------------------------------------
@router.delete("/all_services")
async def delete_all_services():
    services = await _db.services.find().to_list(length=None)

    deleted_services = []
    failed = []

    for svc in services:
        store_name = svc["search_store"]["store_name"]
        provider_email = svc["provider_email"]

        # 1 delete store from GFAPI
        ok, msg = await delete_store_from_gfapi(store_name)

        # 2 delete all uploads linked to this service
        await _db.uploads.delete_many({"provider_email": provider_email})

        # 3 delete all tickets for this provider
        await _db.tickets.delete_many({"provider_email": provider_email})

        # 4 delete provider account
        await _db.providers.delete_one({"email": provider_email})

        # 5 delete service
        await _db.services.delete_one({"_id": svc["_id"]})

        if ok:
            deleted_services.append(store_name)
        else:
            failed.append({"store": store_name, "error": msg})

    return {
        "success": True,
        "message": "All services deleted",
        "deleted_stores": deleted_services,
        "failed": failed
    }


# --------------------------------------------------------
# 2️⃣ DELETE ALL USERS + ALL THEIR TICKETS
# --------------------------------------------------------
@router.delete("/all_users")
async def delete_all_users():
    users = await _db.users.find().to_list(length=None)

    for user in users:
        email = user["email"]

        # delete tickets created by this user
        await _db.tickets.delete_many({"user_email": email})

        # delete user
        await _db.users.delete_one({"email": email})

        # delete conversations
        await _db.conversations.delete_many({"user_email": email})

    return {"success": True, "message": "All users and their tickets deleted"}


# --------------------------------------------------------
# Request models for specific deletion
# --------------------------------------------------------
class DeleteAccountRequest(BaseModel):
    email: EmailStr
    password: str


# --------------------------------------------------------
# 3️⃣ DELETE SPECIFIC SERVICE  (email + password)
# --------------------------------------------------------
@router.delete("/service")
async def delete_specific_service(req: DeleteAccountRequest):
    provider = await _db.providers.find_one({"email": req.email})
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")

    if provider["password"] != req.password:
        raise HTTPException(status_code=403, detail="Invalid password")

    # get service
    service = await _db.services.find_one({"provider_email": req.email})
    if not service:
        raise HTTPException(status_code=404, detail="Service not found")

    store_name = service["search_store"]["store_name"]

    # 1 delete GFAPI store
    ok, msg = await delete_store_from_gfapi(store_name)

    # 2 delete uploads
    await _db.uploads.delete_many({"provider_email": req.email})

    # 3 delete tickets where provider_email == this service
    await _db.tickets.delete_many({"provider_email": req.email})

    # 4 delete provider
    await _db.providers.delete_one({"email": req.email})

    # 5 delete service
    await _db.services.delete_one({"provider_email": req.email})

    return {
        "success": True,
        "message": "Service deleted successfully",
        "store_deleted": ok,
        "store_name": store_name,
        "store_response": msg
    }


# --------------------------------------------------------
# 4️⃣ DELETE SPECIFIC USER (email + password)
# --------------------------------------------------------
@router.delete("/user")
async def delete_specific_user(req: DeleteAccountRequest):
    user = await _db.users.find_one({"email": req.email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user["password"] != req.password:
        raise HTTPException(status_code=403, detail="Invalid password")

    # delete user's tickets
    await _db.tickets.delete_many({"user_email": req.email})

    # delete conversation history
    await _db.conversations.delete_many({"user_email": req.email})

    # remove user
    await _db.users.delete_one({"email": req.email})

    return {
        "success": True,
        "message": "User deleted successfully (all tickets removed as well)"
    }
