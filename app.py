# app.py
import os
import asyncio
from datetime import datetime
from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import logging

# Routes
from delete import init_delete_routes
from service_provider import (
    router as service_router,
    init_service_routes,
    _delete_file_from_gfapi
)
from user import router as user_router, init_user_routes


# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------
load_dotenv()

# Try MONGO_URL (Railway) first, then MONGO_URI (fallback)
MONGO_URI = os.getenv("MONGO_URL") or os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "smart_local_db")

if not MONGO_URI:
    raise ValueError(
        "MongoDB connection string not found. "
        "Please set either MONGO_URL (Railway MongoDB) or MONGO_URI (MongoDB Atlas) environment variable."
    )


# ---------------------------------------------------------
# App + Logging Setup
# ---------------------------------------------------------
app = FastAPI(title="Smart Local Helpdesk API")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")


# ---------------------------------------------------------
# MongoDB Setup with SSL handling
# ---------------------------------------------------------
try:
    # Check if it's MongoDB Atlas (contains mongodb.net) and needs SSL handling
    if "mongodb.net" in MONGO_URI:
        mongo_client = AsyncIOMotorClient(
            MONGO_URI,
            tls=True,
            tlsAllowInvalidCertificates=True,  # Bypass SSL verification for Railway
            retryWrites=True,
            w="majority",
            serverSelectionTimeoutMS=30000,
            connectTimeoutMS=30000
        )
        logger.info("MongoDB Atlas connection with SSL configuration")
    else:
        # Railway MongoDB or local connection
        mongo_client = AsyncIOMotorClient(MONGO_URI)
        logger.info("Railway MongoDB connection established")
    
    db = mongo_client[DB_NAME]
    
except Exception as e:
    logger.error(f"MongoDB connection failed: {e}")
    raise


# ---------------------------------------------------------
# Initialize Routes
# ---------------------------------------------------------
init_service_routes(app, db)
init_user_routes(app, db)
init_delete_routes(app, db)

app.include_router(service_router, prefix="/providers", tags=["providers"])
app.include_router(user_router, prefix="/users", tags=["users"])


# ---------------------------------------------------------
# Scheduler
# ---------------------------------------------------------
scheduler = AsyncIOScheduler()


# ---------------------------------------------------------
# Scheduled Jobs
# ---------------------------------------------------------
async def reset_daily_ticket_counts():
    """Reset today's ticket count at 23:59 UTC."""
    try:
        result = await db.services.update_many(
            {},
            {"$set": {"ticket_counts.today_ticket_count": 0}}
        )
        logger.info(f"[RESET] Today's ticket counts reset for {result.modified_count} services")
    except Exception as e:
        logger.exception(f"Error resetting ticket counts: {e}")


async def cleanup_expired_files():
    """Delete expired files from GFAPI + DB."""
    try:
        now = datetime.utcnow()
        expired_files = await db.uploads.find({
            "delete_at": {"$lte": now}
        }).to_list(length=None)

        for file_doc in expired_files:
            try:
                await _delete_file_from_gfapi(
                    file_doc["store_name"],
                    file_doc["document_id"]
                )
            except Exception as e:
                logger.warning(f"GFAPI deletion failed for {file_doc.get('filename')}: {e}")

            await db.uploads.delete_one({"_id": file_doc["_id"]})
            logger.info(f"[CLEANUP] Removed expired file: {file_doc.get('filename')}")

    except Exception as e:
        logger.exception(f"Error during expired file cleanup: {e}")


# ---------------------------------------------------------
# Startup & Shutdown Events
# ---------------------------------------------------------
@app.on_event("startup")
async def startup():
    logger.info("Starting Smart Local Helpdesk API...")

    try:
        # Test MongoDB connection
        await db.command('ping')
        logger.info("MongoDB connection test successful")
    except Exception as e:
        logger.error(f"MongoDB connection test failed: {e}")
        # Don't crash the app, but log the error

    try:
        scheduler.remove_all_jobs()
    except:
        pass

    scheduler.add_job(
        reset_daily_ticket_counts,
        CronTrigger(hour=23, minute=59),
        id="reset_daily_ticket_counts"
    )

    scheduler.add_job(
        cleanup_expired_files,
        CronTrigger(hour=0, minute=0),
        id="cleanup_expired_files"
    )

    scheduler.start()
    logger.info("Scheduler started successfully.")


@app.on_event("shutdown")
async def shutdown():
    logger.info("Shutting down API...")
    try:
        scheduler.shutdown()
    except:
        pass
    mongo_client.close()
    logger.info("Shutdown complete.")


# ---------------------------------------------------------
# Root Endpoint
# ---------------------------------------------------------
@app.get("/")
async def root():
    return {"message": "Smart Local Helpdesk API Running"}
