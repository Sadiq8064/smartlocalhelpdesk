# app.py
import os
import asyncio
from datetime import datetime
from fastapi import FastAPI

# 🔥 SmartSolve module
from smartsolve import init_smartsolve_routes

# MongoDB
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

# Scheduler
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

import logging

# ---------------------------------------------------------
# Import modules
# ---------------------------------------------------------
from delete import init_delete_routes

from service_provider import (
    router as service_router,
    init_service_routes,
    _delete_file_from_gfapi
)

from user import (
    router as user_router,
    init_user_routes
)

# 🔥 Social Media ML module
from social import (
    init_social_routes   # NOTE: router included inside this initializer
)

# ---------------------------------------------------------
# Load env variables
# ---------------------------------------------------------
load_dotenv()

MONGO_URI = os.getenv("MONGO_URL") or os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "smart_local_db")

if not MONGO_URI:
    raise ValueError(
        "MongoDB connection string missing. Set MONGO_URL (Railway) or MONGO_URI (Atlas)."
    )

# ---------------------------------------------------------
# App & Logging Setup
# ---------------------------------------------------------
app = FastAPI(title="Smart Local Helpdesk API")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# ---------------------------------------------------------
# MongoDB Connection
# ---------------------------------------------------------
try:
    if "mongodb.net" in MONGO_URI:
        mongo_client = AsyncIOMotorClient(
            MONGO_URI,
            tls=True,
            tlsAllowInvalidCertificates=True,
            retryWrites=True,
            w="majority",
            serverSelectionTimeoutMS=30000,
            connectTimeoutMS=30000
        )
        logger.info("Connected to MongoDB Atlas (TLS bypass).")
    else:
        mongo_client = AsyncIOMotorClient(MONGO_URI)
        logger.info("Connected to Railway MongoDB.")

    db = mongo_client[DB_NAME]

except Exception as e:
    logger.error(f"MongoDB connection failed: {e}")
    raise

# ---------------------------------------------------------
# Initialize Routes (Attach DB)
# ---------------------------------------------------------
init_service_routes(app, db)
init_user_routes(app, db)
init_delete_routes(app, db)
init_smartsolve_routes(app, db)

# 🔥 Initialize social module (loads model, binds DB, includes router)
init_social_routes(app, db)

# ---------------------------------------------------------
# Register Routers
# ---------------------------------------------------------
app.include_router(service_router, prefix="/providers", tags=["providers"])
app.include_router(user_router, prefix="/users", tags=["users"])
# ❗ DO NOT include social_router again — it is inside init_social_routes()

# ---------------------------------------------------------
# Scheduler
# ---------------------------------------------------------
scheduler = AsyncIOScheduler()

async def reset_daily_ticket_counts():
    """Reset today's ticket count at 23:59 UTC."""
    try:
        result = await db.services.update_many(
            {},
            {"$set": {"ticket_counts.today_ticket_count": 0}}
        )
        logger.info(
            f"[RESET] Reset ticket counts for {result.modified_count} services"
        )
    except Exception as e:
        logger.exception(f"Error resetting ticket counts: {e}")

async def cleanup_expired_files():
    """Delete expired uploaded files automatically."""
    try:
        now = datetime.utcnow()
        expired_files = await db.uploads.find(
            {"delete_at": {"$lte": now}}
        ).to_list(length=None)

        for file_doc in expired_files:
            try:
                await _delete_file_from_gfapi(
                    file_doc["store_name"],
                    file_doc["document_id"]
                )
            except Exception as e:
                logger.warning(
                    f"GFAPI deletion failed for {file_doc.get('filename')}: {e}"
                )

            await db.uploads.delete_one({"_id": file_doc["_id"]})
            logger.info(f"[CLEANUP] Deleted expired file: {file_doc.get('filename')}")

    except Exception as e:
        logger.exception(f"Error during expired file cleanup: {e}")

# ---------------------------------------------------------
# Startup Event
# ---------------------------------------------------------
@app.on_event("startup")
async def startup():
    logger.info("🚀 Starting Smart Local Helpdesk API...")

    try:
        await db.command("ping")
        logger.info("MongoDB connection OK")
    except Exception as e:
        logger.error(f"MongoDB ping failed: {e}")

    try:
        scheduler.remove_all_jobs()
    except Exception:
        pass

    scheduler.add_job(
        reset_daily_ticket_counts,
        CronTrigger(hour=23, minute=59),
        id="reset_ticket_counts"
    )

    scheduler.add_job(
        cleanup_expired_files,
        CronTrigger(hour=0, minute=0),
        id="cleanup_expired"
    )

    scheduler.start()
    logger.info("Scheduler started successfully.")

# ---------------------------------------------------------
# Shutdown Event
# ---------------------------------------------------------
@app.on_event("shutdown")
async def shutdown():
    logger.info("Shutting down API...")
    try:
        scheduler.shutdown()
    except Exception:
        pass
    mongo_client.close()
    logger.info("Shutdown complete.")

# ---------------------------------------------------------
# Root Endpoint
# ---------------------------------------------------------
@app.get("/")
async def root():
    return {"message": "Smart Local Helpdesk API Running Successfully 🚀"}
