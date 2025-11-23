# service_provider_web.py
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Dict

import aiohttp
from bson import ObjectId
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr
from urllib.parse import urlparse, quote
import uuid
import logging

# Reuse config + GFAPI helpers from main service_provider module
from .service_provider import (
    _upload_to_gfapi,
    _delete_file_from_gfapi,
    GKEY,
)

# ------------------------------
# Setup
# ------------------------------
router = APIRouter()
_db = None

WEB_SCRAPER_BASE_URL = "https://webscrapperr-production.up.railway.app/crawl"

SITES_COLLECTION_NAME = "service_provider_sites"

# Temp dir for JSON files
WEB_UPLOAD_DIR = Path("temp_web_uploads")
WEB_UPLOAD_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("service_provider_web")


def init_service_web_routes(app, db):
    """
    Called from app.py to initialize DB reference & indexes for web scraping.
    """
    global _db
    _db = db
    asyncio.create_task(_ensure_web_indexes())


async def _ensure_web_indexes():
    try:
        await _db[SITES_COLLECTION_NAME].create_index(
            "provider_email", unique=True
        )
        await _db[SITES_COLLECTION_NAME].create_index("store_name")
        logger.info("service_provider_web indexes ensured")
    except Exception as e:
        logger.exception("Error ensuring service_provider_web indexes: %s", e)


# ------------------------------
# Helpers
# ------------------------------
def _extract_website_name(url: str) -> str:
    """
    Extract a clean website name (no http/https, no .com/.in/etc).
    Example:
      https://www.example.com/path -> example
      http://sub.domain.gov.in -> sub
      example.org -> example
    """
    if not url:
        return "website"

    url = url.strip()
    parsed = urlparse(url)

    # Handle case where user sends "example.com" without scheme
    host = parsed.netloc or parsed.path
    host = host.lower()

    # Remove port if any
    if ":" in host:
        host = host.split(":", 1)[0]

    # Remove www.
    if host.startswith("www."):
        host = host[4:]

    parts = [p for p in host.split(".") if p]
    if not parts:
        return "website"

    # Take only the first label -> no .com/.in/etc and no dots
    name = parts[0]
    if not name:
        name = "website"
    return name


async def _call_web_scraper(url: str) -> Any:
    """
    Call external webscrapper service and return JSON body.
    GET https://webscrapperr-production.up.railway.app/crawl?url={url}
    """
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")

    encoded_url = quote(url, safe="")
    full_url = f"{WEB_SCRAPER_BASE_URL}?url={encoded_url}"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(full_url, timeout=120) as resp:
                text = await resp.text()
                if resp.status != 200:
                    logger.warning("Web scraper HTTP %s: %s", resp.status, text)
                    raise HTTPException(
                        status_code=502,
                        detail=f"Web scraper failed with status {resp.status}",
                    )
                try:
                    data = await resp.json()
                except Exception:
                    # Try parsing explicitly if resp.json() fails
                    try:
                        data = json.loads(text)
                    except Exception:
                        raise HTTPException(
                            status_code=500,
                            detail="Failed to parse web scraper response as JSON",
                        )
                return data
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Web scraper call failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Error calling web scraper: {e}",
        )


def _write_temp_json_file(website_name: str, json_content: Any) -> Path:
    """
    Write JSON content to a temp file and return the Path.
    The GFAPI filename will be {website_name}.json, but the temp file
    will have a unique suffix to avoid collisions.
    """
    WEB_UPLOAD_DIR.mkdir(exist_ok=True)
    temp_name = f"{website_name}_{uuid.uuid4().hex}.json"
    temp_path = WEB_UPLOAD_DIR / temp_name

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(json_content, f, ensure_ascii=False, indent=2)

    return temp_path


async def _ensure_provider_and_service(provider_email: str) -> Dict[str, Any]:
    """
    Validate provider exists and return (provider, service).
    Also ensures the service has a search_store with store_name.
    """
    provider = await _db.providers.find_one({"email": provider_email})
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")

    service = await _db.services.find_one({"provider_email": provider_email})
    if not service:
        raise HTTPException(status_code=404, detail="Service not found for this provider")

    search_store = service.get("search_store") or {}
    store_name = search_store.get("store_name")
    if not store_name:
        raise HTTPException(
            status_code=500,
            detail="Service search_store.store_name not configured for this provider",
        )

    return {
        "provider": provider,
        "service": service,
        "store_name": store_name,
    }


async def _get_site_doc(provider_email: str) -> Optional[Dict[str, Any]]:
    return await _db[SITES_COLLECTION_NAME].find_one({"provider_email": provider_email})


# ------------------------------
# Request models
# ------------------------------
class ScrapeAndUploadRequest(BaseModel):
    provider_email: EmailStr
    url: str
    text: str


class SetReadFlagRequest(BaseModel):
    provider_email: EmailStr
    read_website: bool


class UpdateWebsiteRequest(BaseModel):
    provider_email: EmailStr
    new_url: str
    new_text: str


# ---------------------------------------------------------
# API 1: SCRAPE + UPLOAD JSON TO GFAPI + STORE IN MONGO
# ---------------------------------------------------------
@router.post("/scrape_and_upload")
async def scrape_and_upload(req: ScrapeAndUploadRequest):
    """
    1. Validate provider & service
    2. Call webscrapper API with given URL
    3. Derive website_name -> website_name.json
    4. Upload JSON to GFAPI using provider's store_name and GKEY
    5. Store record in MongoDB collection `service_provider_sites`:
         - url
         - text
         - read_website = true
         - json_content
         - filename (from GFAPI)
         - store_name
         - gfapi.document_id, gfapi.document_resource
    """
    # Ensure provider + service + store_name
    ctx = await _ensure_provider_and_service(req.provider_email)
    store_name = ctx["store_name"]

    # Check if site doc already exists
    existing = await _get_site_doc(req.provider_email)
    if existing:
        raise HTTPException(
            status_code=400,
            detail="Website already configured for this provider. Use /update_website to change it.",
        )

    # 1) Crawl website
    scraped_json = await _call_web_scraper(req.url)

    # 2) Prepare website name + filenames
    website_name = _extract_website_name(req.url)
    gfapi_filename = f"{website_name}.json"

    # 3) Write JSON to temp file and upload to GFAPI
    temp_path = _write_temp_json_file(website_name, scraped_json)

    try:
        gfapi_result = await _upload_to_gfapi(
            store_name=store_name,
            file_path=temp_path,
            filename=gfapi_filename,
        )
    finally:
        # Always clean temp file
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass

    if not gfapi_result.get("success"):
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload JSON to GFAPI: {gfapi_result.get('error')}",
        )

    # 4) Store in MongoDB
    now = datetime.utcnow()
    site_doc = {
        "provider_email": req.provider_email,
        "store_name": store_name,
        "url": req.url,
        "text": req.text,
        "read_website": True,
        "json_content": scraped_json,
        "filename": gfapi_result.get("filename") or gfapi_filename,
        "gfapi": {
            "active": True,
            "filename": gfapi_result.get("filename") or gfapi_filename,
            "document_id": gfapi_result.get("document_id"),
            "document_resource": gfapi_result.get("document_resource"),
        },
        "created_at": now,
        "updated_at": now,
    }

    await _db[SITES_COLLECTION_NAME].insert_one(site_doc)

    return {
        "success": True,
        "message": "Website scraped and JSON uploaded successfully",
        "provider_email": req.provider_email,
        "url": req.url,
        "web_json_filename": site_doc["gfapi"]["filename"],
        "gfapi_document_id": site_doc["gfapi"]["document_id"],
    }


# ---------------------------------------------------------
# API 2: SET / TOGGLE READ_WEBSITE FLAG
#   - false -> true : re-upload JSON to GFAPI (no re-scrape)
#   - true  -> false: delete JSON from GFAPI
# ---------------------------------------------------------
@router.post("/set_read_flag")
async def set_read_flag(req: SetReadFlagRequest):
    """
    Toggle read_website:
      - If changing False -> True:
          * Re-upload existing json_content to GFAPI
          * Update gfapi.* fields and read_website = True
      - If changing True -> False:
          * Delete file from GFAPI using document_id + store_name
          * Set read_website = False
    """
    # Ensure provider & service to get store_name
    ctx = await _ensure_provider_and_service(req.provider_email)
    store_name = ctx["store_name"]

    site = await _get_site_doc(req.provider_email)
    if not site:
        raise HTTPException(
            status_code=404,
            detail="No website configuration found for this provider",
        )

    current_flag = bool(site.get("read_website", False))
    desired_flag = bool(req.read_website)

    if current_flag == desired_flag:
        # No change required
        return {
            "success": True,
            "message": "read_website is already set to the requested value",
            "read_website": current_flag,
        }

    gfapi_info = site.get("gfapi") or {}
    filename = gfapi_info.get("filename") or site.get("filename")

    # -----------------------------
    # Case 1: False -> True
    # -----------------------------
    if not current_flag and desired_flag:
        json_content = site.get("json_content")
        if json_content is None:
            raise HTTPException(
                status_code=500,
                detail="json_content missing in site document; cannot re-upload",
            )

        website_name = _extract_website_name(site.get("url") or "")
        gfapi_filename = filename or f"{website_name}.json"

        temp_path = _write_temp_json_file(website_name, json_content)
        try:
            gfapi_result = await _upload_to_gfapi(
                store_name=store_name,
                file_path=temp_path,
                filename=gfapi_filename,
            )
        finally:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except Exception:
                pass

        if not gfapi_result.get("success"):
            raise HTTPException(
                status_code=500,
                detail=f"Failed to re-upload JSON to GFAPI: {gfapi_result.get('error')}",
            )

        # Update document
        update_fields = {
            "read_website": True,
            "gfapi": {
                "active": True,
                "filename": gfapi_result.get("filename") or gfapi_filename,
                "document_id": gfapi_result.get("document_id"),
                "document_resource": gfapi_result.get("document_resource"),
            },
            "updated_at": datetime.utcnow(),
        }

        await _db[SITES_COLLECTION_NAME].update_one(
            {"_id": site["_id"]},
            {"$set": update_fields},
        )

        return {
            "success": True,
            "message": "read_website enabled and JSON re-uploaded to GFAPI",
            "read_website": True,
            "gfapi_document_id": gfapi_result.get("document_id"),
        }

    # -----------------------------
    # Case 2: True -> False
    # -----------------------------
    if current_flag and not desired_flag:
        document_id = gfapi_info.get("document_id")
        if document_id:
            gf_del_result = await _delete_file_from_gfapi(
                store_name=store_name,
                document_id=document_id,
            )
            if not gf_del_result.get("success"):
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to delete web JSON from GFAPI: {gf_del_result.get('error')}",
                )

        # Update Mongo: turn off read_website, mark gfapi inactive
        update_fields = {
            "read_website": False,
            "gfapi.active": False,
            "updated_at": datetime.utcnow(),
        }

        await _db[SITES_COLLECTION_NAME].update_one(
            {"_id": site["_id"]},
            {"$set": update_fields},
        )

        return {
            "success": True,
            "message": "read_website disabled and JSON deleted from GFAPI",
            "read_website": False,
        }


# ---------------------------------------------------------
# API 4: UPDATE WEBSITE (URL + TEXT + JSON)
#   - Delete old JSON from GFAPI
#   - Delete old Mongo doc
#   - Crawl new URL
#   - Upload new JSON to GFAPI
#   - Store new document
# ---------------------------------------------------------
@router.post("/update_website")
async def update_website(req: UpdateWebsiteRequest):
    """
    Replace website config:
      1. Ensure provider & service
      2. If old site exists:
          - Delete JSON from GFAPI
          - Delete site document from Mongo
      3. Crawl new_url
      4. Upload new JSON to GFAPI
      5. Insert new site document
    """
    # Ensure provider & service & store_name
    ctx = await _ensure_provider_and_service(req.provider_email)
    store_name = ctx["store_name"]

    # Handle old site if present
    old_site = await _get_site_doc(req.provider_email)
    if old_site:
        old_gfapi = old_site.get("gfapi") or {}
        old_doc_id = old_gfapi.get("document_id")
        if old_doc_id and old_gfapi.get("active", True):
            gf_del_result = await _delete_file_from_gfapi(
                store_name=store_name,
                document_id=old_doc_id,
            )
            if not gf_del_result.get("success"):
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to delete old JSON from GFAPI: {gf_del_result.get('error')}",
                )

        await _db[SITES_COLLECTION_NAME].delete_one({"_id": old_site["_id"]})

    # Crawl new URL
    scraped_json = await _call_web_scraper(req.new_url)

    # Prepare new filename
    website_name = _extract_website_name(req.new_url)
    gfapi_filename = f"{website_name}.json"

    # Write JSON temp & upload to GFAPI
    temp_path = _write_temp_json_file(website_name, scraped_json)
    try:
        gfapi_result = await _upload_to_gfapi(
            store_name=store_name,
            file_path=temp_path,
            filename=gfapi_filename,
        )
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass

    if not gfapi_result.get("success"):
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload new JSON to GFAPI: {gfapi_result.get('error')}",
        )

    now = datetime.utcnow()
    new_site_doc = {
        "provider_email": req.provider_email,
        "store_name": store_name,
        "url": req.new_url,
        "text": req.new_text,
        "read_website": True,
        "json_content": scraped_json,
        "filename": gfapi_result.get("filename") or gfapi_filename,
        "gfapi": {
            "active": True,
            "filename": gfapi_result.get("filename") or gfapi_filename,
            "document_id": gfapi_result.get("document_id"),
            "document_resource": gfapi_result.get("document_resource"),
        },
        "created_at": now,
        "updated_at": now,
    }

    await _db[SITES_COLLECTION_NAME].insert_one(new_site_doc)

    return {
        "success": True,
        "message": "Website updated, scraped, and JSON uploaded successfully",
        "provider_email": req.provider_email,
        "url": req.new_url,
        "web_json_filename": new_site_doc["gfapi"]["filename"],
        "gfapi_document_id": new_site_doc["gfapi"]["document_id"],
    }
