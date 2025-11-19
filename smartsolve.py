# smartsolve.py
import os
import uuid
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Optional
from fastapi import APIRouter, HTTPException
import logging
import numpy as np

import sib_api_v3_sdk

from sentence_transformers import SentenceTransformer, util
import hdbscan
import torch
import spacy
import re
from collections import defaultdict

from google import genai
from google.genai import types

# =============================================================================
# ENVIRONMENT
# =============================================================================
BREVO_API_KEY = os.getenv("BREVO_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not BREVO_API_KEY:
    raise ValueError("BREVO_API_KEY env missing")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY env missing")

# =============================================================================
# ROUTER + DB
# =============================================================================
router = APIRouter()
_db = None
logger = logging.getLogger("smartsolve")
logging.basicConfig(level=logging.INFO)

def init_smartsolve_routes(app, db):
    global _db
    _db = db
    app.include_router(router, prefix="/smartsolve", tags=["smartsolve"])

# =============================================================================
# EMAIL SENDER (same as other files)
# =============================================================================
async def _send_brevo_email(to_email: str, subject: str, html_content: str):
    def _sync_send():
        configuration = sib_api_v3_sdk.Configuration()
        configuration.api_key['api-key'] = BREVO_API_KEY
        api_client = sib_api_v3_sdk.ApiClient(configuration)
        api_instance = sib_api_v3_sdk.TransactionalEmailsApi(api_client)

        email_data = sib_api_v3_sdk.SendSmtpEmail(
            to=[{"email": to_email}],
            sender={"name": "Smart Local Helpdesk", "email": "mrsadiq471@gmail.com"},
            subject=subject,
            html_content=html_content,
            text_content="Notification from Smart Local Helpdesk"
        )

        return api_instance.send_transac_email(email_data)

    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, _sync_send)
    except Exception as e:
        logger.exception("Email failed: %s", e)


# =============================================================================
# GEMINI SUMMARY
# =============================================================================
async def _summarize_with_gemini(queries: List[str]) -> str:
    """
    Summarize multiple similar ticket problems into ONE clean problem statement.
    """
    if not queries:
        return ""

    prompt = f"""
You are an assistant. These are multiple user complaints written differently but about the same issue:
{json.dumps(queries, indent=2)}

Write ONE clean summarized problem that captures the core issue.
Do NOT mention 'users' or 'multiple people'. Just describe the problem as one person reported it.
"""

    loop = asyncio.get_running_loop()
    def _sync_call():
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.1)
        )
        return response.text

    try:
        result = await loop.run_in_executor(None, _sync_call)
        return result.strip()
    except Exception as e:
        logger.exception("Gemini summarization failed: %s", e)
        return queries[0]  # fallback


# =============================================================================
# CLUSTERING ENGINE (your exact algorithm)
# =============================================================================
class SmartHelpdeskClusterer:
    def __init__(self, model_name='sentence-transformers/all-mpnet-base-v2', min_cluster_size=2):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name).to(self.device)
        self.nlp = spacy.load('en_core_web_sm')

        self.min_cluster_size = min_cluster_size
        self.queries = []
        self.embeddings = None
        self.cluster_labels = None
        self.clusters = {}

    # ------ LOCATION EXTRACTION ------
    def extract_location(self, text):
        text_lower = text.lower()

        r1 = re.search(r'near\s+([a-z]+)', text_lower)
        if r1: return r1.group(1).title()

        r2 = re.search(r'from\s+([a-z]+)', text_lower)
        if r2: return r2.group(1).title()

        known_areas = ['hubli', 'dharwad', 'kusugal', 'vidyanagar', 'rajajinagar',
                       'jayanagar', 'koramangala', 'gokul', 'keshwapur', 'navnagar']
        for area in known_areas:
            if area in text_lower:
                return area.title()

        try:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ['GPE', 'LOC']:
                    return ent.text.title()
        except:
            pass

        return "Location Unknown"

    # ------ PROBLEM TYPE ------
    def extract_problem_type(self, text):
        text_lower = text.lower()
        mapping = {
            'Electricity': ['electricity','current','power','light','outage','blackout'],
            'Water Supply': ['water','tap','supply','pani'],
            'Roads': ['road','pothole'],
            'Sanitation': ['garbage','waste','trash','cleaning','drainage'],
            'Traffic': ['traffic','jam','signal']
        }
        for k, arr in mapping.items():
            if any(a in text_lower for a in arr):
                return k
        return "General Issue"

    # ------ ADD QUERIES ------
    def add_queries(self, queries):
        for q in queries:
            q['detected_location'] = self.extract_location(q['query'])
            q['detected_problem'] = self.extract_problem_type(q['query'])
        self.queries = queries

    # ------ EMBEDDINGS ------
    def generate_embeddings(self):
        texts = [q['query'] for q in self.queries]
        self.embeddings = self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=32
        )

    # ------ CLUSTERING ------
    def perform_clustering(self, sensitivity='loose'):
        params = {
            'strict': {'min_cluster_size': 3, 'epsilon': 0.2},
            'medium': {'min_cluster_size': 2, 'epsilon': 0.35},
            'loose': {'min_cluster_size': 2, 'epsilon': 0.5}
        }[sensitivity]

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=params['min_cluster_size'],
            min_samples=1,
            metric='euclidean',
            cluster_selection_epsilon=params['epsilon'],
            core_dist_n_jobs=-1
        )

        self.cluster_labels = clusterer.fit_predict(self.embeddings)
        self.clusters = defaultdict(list)

        for idx, label in enumerate(self.cluster_labels):
            if label != -1:
                self.clusters[label].append(idx)

        self.clusters = dict(sorted(self.clusters.items(), key=lambda x: len(x[1]), reverse=True))

    # ------ SUMMARIES ------
    def get_cluster_data(self):
        """
        Returns:
        [
          { "queries": [list_of_texts], "indices": [0,1,4], "cluster_id": 0 },
          { ... }
        ]
        """
        final = []

        # clusters
        for cid, idxs in self.clusters.items():
            qtexts = [self.queries[i]['query'] for i in idxs]
            final.append({
                "cluster_id": cid,
                "queries": qtexts,
                "indices": idxs
            })

        # unclustered
        unclustered = np.where(self.cluster_labels == -1)[0]
        for idx in unclustered:
            final.append({
                "cluster_id": None,
                "queries": [self.queries[idx]['query']],
                "indices": [idx]
            })

        return final


# =============================================================================
# API #1: CLUSTER PENDING TICKETS
# =============================================================================
@router.get("/{provider_email}/cluster_pending")
async def cluster_pending(provider_email: str):
    provider = await _db.providers.find_one({"email": provider_email})
    if not provider:
        raise HTTPException(404, "Provider not found")

    pending_tickets = await _db.tickets.find({
        "provider_email": provider_email,
        "status": "pending"
    }).to_list(None)

    if not pending_tickets:
        return {
            "success": True,
            "before_count": 0,
            "after_count": 0,
            "questions": []
        }

    before_count = len(pending_tickets)

    # Convert to clustering format
    q_list = []
    idx_map = {}   # map clustering index -> ticket_id
    for i, t in enumerate(pending_tickets):
        q_list.append({
            "id": i,
            "query": t["problem"],
            "timestamp": str(t["created_at"])
        })
        idx_map[i] = t["ticket_id"]

    # Run clustering
    clusterer = SmartHelpdeskClusterer()
    clusterer.add_queries(q_list)
    clusterer.generate_embeddings()
    clusterer.perform_clustering()

    clusters = clusterer.get_cluster_data()

    # Clean previous cache
    await _db.cluster_cache.delete_many({"provider_email": provider_email})

    final_questions = []
    question_counter = 1

    # Process clusters & unclustered
    for blk in clusters:
        text_list = blk["queries"]
        indices = blk["indices"]

        if len(text_list) > 1:
            # summarize
            summary = await _summarize_with_gemini(text_list)
            qid = f"C_{uuid.uuid4().hex[:6]}"
        else:
            summary = text_list[0]
            qid = f"T_{idx_map[indices[0]]}"

        # store cache (mapping qid -> actual tickets)
        mapped_ticket_ids = [idx_map[x] for x in indices]

        await _db.cluster_cache.insert_one({
            "provider_email": provider_email,
            "question_id": qid,
            "summary": summary,
            "tickets": mapped_ticket_ids,
            "created_at": datetime.utcnow()
        })

        final_questions.append({
            "question_id": qid,
            "question": summary
        })

        question_counter += 1

    after_count = len(final_questions)

    return {
        "success": True,
        "before_count": before_count,
        "after_count": after_count,
        "questions": final_questions
    }


# =============================================================================
# API #2: SUBMIT SOLUTIONS
# =============================================================================
@router.post("/{provider_email}/submit_solutions")
async def submit_solutions(provider_email: str, payload: List[Dict]):
    provider = await _db.providers.find_one({"email": provider_email})
    if not provider:
        raise HTTPException(404, "Provider not found")

    for item in payload:
        qid = item.get("question_id")
        sol = item.get("solution")

        cache = await _db.cluster_cache.find_one({
            "provider_email": provider_email,
            "question_id": qid
        })

        if not cache:
            raise HTTPException(400, f"Invalid question_id: {qid}")

        ticket_ids = cache["tickets"]

        # Update all real tickets
        for tid in ticket_ids:
            await _db.tickets.update_one(
                {"ticket_id": tid},
                {"$set": {
                    "solution": sol,
                    "status": "completed",
                    "completed_at": datetime.utcnow()
                }}
            )

            # notify user
            ticket = await _db.tickets.find_one({"ticket_id": tid})
            await _send_brevo_email(
                ticket["user_email"],
                "Your Ticket Has Been Resolved",
                f"""
                <h3>Hello,</h3>
                <p>Your ticket (<b>{tid}</b>) has been resolved.</p>
                <p><strong>Solution:</strong> {sol}</p>
                <p>Thank you.</p>
                """
            )

    # Cleanup cache after applying solutions
    await _db.cluster_cache.delete_many({"provider_email": provider_email})

    # Update ticket counters
    from service_provider import _update_ticket_counts
    await _update_ticket_counts(provider_email)

    return {"success": True}

