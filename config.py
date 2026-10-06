"""Central configuration for the Gmail auto-triage bot.

Edit CATEGORIES (rules) here without changing classification or Gmail logic.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent
CREDENTIALS_PATH = BASE_DIR / "credentials.json"
TOKEN_PATH = BASE_DIR / "token.json"

# OAuth — gmail.modify is required to add labels and archive (remove INBOX).
# Never use gmail.readonly. Never delete messages.
GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]

# Classification model (overridable via GROQ_MODEL in .env)
# llama-3.3-70b-versatile was shut down 2026-08-16; Groq recommends gpt-oss-120b.
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()

# --- Editable triage rules -------------------------------------------------
# label name -> short description of when to apply (fed to the LLM prompt)
CATEGORIES: dict[str, str] = {
    "assessment": (
        "HackerRank / Codility / take-home / coding-test invite, or a "
        '"next steps" email that asks you to complete a timed assessment. '
        "Not a generic job posting."
    ),
    "rejected at resume": (
        "rejection at resume/screening stage "
        '("not moving forward", "other candidates", "position filled", etc.)'
    ),
    "lead": (
        "ONLY direct personal interest from a recruiter/company in YOU: "
        "wants to schedule a call/interview, or outreach about YOUR specific "
        "application. NOT job postings, NOT Indeed/LinkedIn suggestions, "
        "NOT 'submit your application' invites, NOT marketing/coaching"
    ),
    "job alert": (
        "automated job digests and role suggestions — including Indeed, "
        "LinkedIn, Glassdoor 'you'd be a great fit' / 'new jobs for you' / "
        "single-role recommendation emails that are NOT application "
        "confirmations. Never personal recruiter outreach."
    ),
    "indeed apply": (
        "ONLY Indeed Apply application confirmations / receipts "
        '(subjects like "Indeed Application: <role> - <company>", '
        '"your application was submitted"). NOT Indeed job suggestions or '
        "digests — those are job alert."
    ),
    "loop": (
        "already in an interview process: multi-round / onsite / loop "
        "scheduling, panel coordination, next-round logistics with a "
        "recruiter or hiring team"
    ),
    "notification": (
        "generic platform noise, marketing/promo, career coaching, webinars, "
        "'connect your profile', security alerts — no recruiter interest"
    ),
}

# Few-shot examples injected into the classifier system prompt (editable).
# Each entry: (subject_or_snippet, category, brief_why)
FEW_SHOT_EXAMPLES: list[tuple[str, str, str]] = [
    (
        "Free 30-minute career coaching session",
        "notification",
        "marketing/coaching promo, not recruiter interest",
    ),
    (
        "Product Manager role @ X — submit your application",
        "job alert",
        "job posting / apply invite, not personal outreach",
    ),
    (
        "DevOps Engineer - AI Trainer at DataAnnotation (from Indeed)",
        "job alert",
        "Indeed job suggestion/digest, NOT an application confirmation",
    ),
    (
        "Indeed Application: Frontend Application Developer - Acme Corp",
        "indeed apply",
        "Indeed Apply confirmation that an application was submitted",
    ),
    (
        "Hi Ahmed, we'd love to schedule a call to discuss your application",
        "lead",
        "direct recruiter interest about the user's application",
    ),
    (
        "Please complete the HackerRank assessment for the SWE role",
        "assessment",
        "coding/test invite",
    ),
    (
        "Confirming your onsite loop — Round 2 system design Thu 2pm",
        "loop",
        "multi-round interview scheduling already in process",
    ),
]

# Special classifier outputs (not Gmail labels)
NONE_CATEGORY = "none"
MANAGED_LABELS: frozenset[str] = frozenset(CATEGORIES.keys())

# Runtime settings from env
DRY_RUN: bool = os.getenv("DRY_RUN", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "").strip()
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
GMAIL_QUERY: str = os.getenv("GMAIL_QUERY", "is:unread in:inbox").strip()
BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "10"))
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").strip().upper()
# Pause between Groq classify calls to stay under free-tier rate limits.
REQUEST_DELAY_SECONDS: float = float(os.getenv("REQUEST_DELAY_SECONDS", "3"))

# Body trim for LLM context
MAX_BODY_CHARS: int = 4000

# Retry / backoff
MAX_RETRIES: int = 5
RETRY_BASE_SECONDS: float = 1.0
