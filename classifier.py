"""Groq LLM classifier for Gmail triage."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass

from groq import APIConnectionError, APIStatusError, Groq, RateLimitError

from config import (
    CATEGORIES,
    FEW_SHOT_EXAMPLES,
    GROQ_API_KEY,
    GROQ_MODEL,
    MAX_RETRIES,
    NONE_CATEGORY,
    RETRY_BASE_SECONDS,
)

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)
_ALLOWED = frozenset(CATEGORIES) | {NONE_CATEGORY}


@dataclass(frozen=True)
class ClassificationResult:
    category: str
    confidence: float
    reason: str
    raw_text: str = ""


def _build_system_prompt() -> str:
    category_lines = "\n".join(
        f'- "{name}": {desc}' for name, desc in CATEGORIES.items()
    )
    few_shot_lines = "\n".join(
        f'- Subject/snippet: "{snippet}" → "{label}" ({why})'
        for snippet, label, why in FEW_SHOT_EXAMPLES
    )
    allowed = ", ".join(f'"{name}"' for name in CATEGORIES) + f', "{NONE_CATEGORY}"'
    return f"""You are an email triage classifier for a job-seeker's Gmail inbox.

Classify each email into exactly ONE category from this list, or "{NONE_CATEGORY}"
if nothing fits well or you are unsure.

Categories (label -> when to apply):
{category_lines}

Critical disambiguation rules:
- Prefer "{NONE_CATEGORY}" over a forced guess.
- "indeed apply" vs "job alert" (Indeed):
  * "Indeed Application: …" / application submitted/received receipts
    → "indeed apply" ONLY.
  * Indeed job suggestions, "new jobs for you", single-role pitches,
    digests from sender Indeed → "job alert", NEVER "indeed apply".
- "lead" = ONLY personal recruiter/company interest in YOU (schedule a
  call/interview about YOUR application). NOT postings, NOT Indeed/LinkedIn
  suggestions, NOT marketing.
- "assessment" = coding test / take-home / HackerRank-style invite.
- "loop" = already interviewing; multi-round / onsite scheduling logistics.
- Marketing/promo (coaching, webinars, "connect your profile") → "notification".
- "rejected at resume" is early-stage rejection, not later interview rejection.
- confidence is a float from 0.0 to 1.0 reflecting how sure you are.

Few-shot examples:
{few_shot_lines}

Respond with ONLY a single JSON object (no markdown, no commentary):
{{"category": <one of {allowed}>, "confidence": <0.0-1.0>, "reason": "<short>"}}
"""


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = _FENCE_RE.sub("", cleaned).strip()
    return cleaned


def _extract_json_object(text: str) -> dict:
    cleaned = _strip_code_fences(text)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # Fallback: first {...} blob in the response
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in model response: {text[:200]!r}")
    data = json.loads(cleaned[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("Parsed JSON is not an object")
    return data


def parse_classification(text: str) -> ClassificationResult:
    """Defensively parse model output into a ClassificationResult."""
    try:
        data = _extract_json_object(text)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning("Malformed classifier JSON (%s); treating as none", exc)
        return ClassificationResult(
            category=NONE_CATEGORY,
            confidence=0.0,
            reason=f"unparseable model output: {exc}",
            raw_text=text,
        )

    category = str(data.get("category", NONE_CATEGORY)).strip().lower()
    # Preserve exact label casing from CATEGORIES keys where possible.
    category_map = {name.lower(): name for name in CATEGORIES}
    category_map[NONE_CATEGORY.lower()] = NONE_CATEGORY
    category = category_map.get(category, NONE_CATEGORY)
    if category not in _ALLOWED:
        category = NONE_CATEGORY

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    reason = str(data.get("reason", "")).strip() or "(no reason)"
    return ClassificationResult(
        category=category,
        confidence=confidence,
        reason=reason,
        raw_text=text,
    )


class EmailClassifier:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = GROQ_MODEL,
    ) -> None:
        key = (api_key if api_key is not None else GROQ_API_KEY).strip()
        if not key:
            raise ValueError(
                "GROQ_API_KEY is missing. Set it in .env (see .env.example)."
            )
        self.model = model
        self._client = Groq(api_key=key)
        self._system_prompt = _build_system_prompt()

    def classify(self, *, sender: str, subject: str, body: str) -> ClassificationResult:
        user_content = (
            f"From: {sender}\n"
            f"Subject: {subject}\n"
            f"Body:\n{body or '(empty)'}\n"
        )
        last_error: Exception | None = None
        # gpt-oss models may spend tokens on reasoning; keep headroom for JSON.
        max_tokens = 1024

        for attempt in range(MAX_RETRIES):
            try:
                completion = self._client.chat.completions.create(
                    model=self.model,
                    temperature=0.1,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": self._system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                )
                content = (completion.choices[0].message.content or "").strip()
                return parse_classification(content)
            except RateLimitError as exc:
                last_error = exc
                delay = RETRY_BASE_SECONDS * (2**attempt)
                logger.warning(
                    "Groq rate limited; retry %s/%s in %.1fs",
                    attempt + 1,
                    MAX_RETRIES,
                    delay,
                )
                time.sleep(delay)
            except APIConnectionError as exc:
                last_error = exc
                delay = RETRY_BASE_SECONDS * (2**attempt)
                logger.warning(
                    "Groq connection error; retry %s/%s in %.1fs",
                    attempt + 1,
                    MAX_RETRIES,
                    delay,
                )
                time.sleep(delay)
            except APIStatusError as exc:
                last_error = exc
                status = getattr(exc, "status_code", None)
                # JSON mode can fail if max_tokens cuts off mid-object — bump and retry.
                err_text = str(exc).lower()
                if status == 400 and (
                    "json_validate_failed" in err_text
                    or "max completion tokens" in err_text
                    or "failed to generate json" in err_text
                ):
                    max_tokens = min(max_tokens * 2, 4096)
                    delay = RETRY_BASE_SECONDS * (2**attempt)
                    logger.warning(
                        "Groq JSON generation failed; retry %s/%s "
                        "(max_tokens=%s) in %.1fs",
                        attempt + 1,
                        MAX_RETRIES,
                        max_tokens,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                if status is not None and status < 500 and status != 429:
                    logger.error("Groq API error %s: %s", status, exc)
                    raise
                delay = RETRY_BASE_SECONDS * (2**attempt)
                logger.warning(
                    "Groq API status %s; retry %s/%s in %.1fs",
                    status,
                    attempt + 1,
                    MAX_RETRIES,
                    delay,
                )
                time.sleep(delay)

        assert last_error is not None
        raise last_error
