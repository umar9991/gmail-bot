"""Gmail API client: OAuth desktop flow, message fetch, labels, archive."""

from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass
from email.utils import parseaddr
from typing import Any, Callable, TypeVar

from google.auth.exceptions import RefreshError, TransportError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import Resource, build
from googleapiclient.errors import HttpError

from config import (
    CREDENTIALS_PATH,
    GMAIL_SCOPES,
    MANAGED_LABELS,
    MAX_RETRIES,
    RETRY_BASE_SECONDS,
    TOKEN_PATH,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class EmailMessage:
    id: str
    thread_id: str
    sender: str
    subject: str
    body: str
    label_ids: frozenset[str]


def _with_retry(fn: Callable[[], T], *, what: str) -> T:
    """Call fn with exponential backoff on transient Gmail API errors."""
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn()
        except HttpError as exc:
            last_error = exc
            status = getattr(exc.resp, "status", None)
            # Retry rate limits and 5xx; fail fast on other client errors.
            if status not in {429, 500, 502, 503, 504}:
                raise
            delay = RETRY_BASE_SECONDS * (2**attempt)
            logger.warning(
                "%s failed (HTTP %s), retry %s/%s in %.1fs",
                what,
                status,
                attempt + 1,
                MAX_RETRIES,
                delay,
            )
            time.sleep(delay)
        except (TimeoutError, ConnectionError, OSError) as exc:
            last_error = exc
            delay = RETRY_BASE_SECONDS * (2**attempt)
            logger.warning(
                "%s failed (%s), retry %s/%s in %.1fs",
                what,
                exc,
                attempt + 1,
                MAX_RETRIES,
                delay,
            )
            time.sleep(delay)
    assert last_error is not None
    raise last_error


def get_gmail_service() -> Resource:
    """Authenticate via OAuth desktop flow; cache tokens in token.json."""
    if not CREDENTIALS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CREDENTIALS_PATH.name}. Download OAuth client "
            "credentials from Google Cloud Console and place them here. "
            "See README.md."
        )

    creds: Credentials | None = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), GMAIL_SCOPES)

    if creds and creds.expired and creds.refresh_token:
        logger.info("Refreshing expired Gmail OAuth token")
        try:
            creds.refresh(Request())
        except TransportError as exc:
            raise RuntimeError(
                "Could not reach Google OAuth (network/DNS). "
                "Check your internet connection, VPN, or DNS, then retry. "
                f"Detail: {exc}"
            ) from exc
        except RefreshError as exc:
            # Invalid/revoked refresh token — force a fresh browser login.
            logger.warning(
                "Token refresh rejected (%s); deleting %s and re-authorizing",
                exc,
                TOKEN_PATH.name,
            )
            TOKEN_PATH.unlink(missing_ok=True)
            creds = None
        else:
            TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")
    if not creds or not creds.valid:
        # Fixed loopback port — random ports often trigger Google's generic
        # "400 malformed" / redirect mismatch pages for misconfigured clients.
        oauth_port = 8080
        logger.info(
            "Starting OAuth desktop flow on http://localhost:%s/ (browser will open)",
            oauth_port,
        )
        flow = InstalledAppFlow.from_client_secrets_file(
            str(CREDENTIALS_PATH), GMAIL_SCOPES
        )
        try:
            # prompt=select_account forces the Google account chooser so a
            # previous browser session does not silently reuse the old Gmail.
            creds = flow.run_local_server(
                host="localhost",
                port=oauth_port,
                open_browser=True,
                prompt="consent select_account",
                access_type="offline",
                authorization_prompt_message=(
                    "Please visit this URL to authorize the Gmail bot:\n{url}\n"
                ),
                success_message=(
                    "Authorization complete. You can close this tab and return "
                    "to the terminal."
                ),
            )
        except OSError as exc:
            raise RuntimeError(
                f"Could not bind http://localhost:{oauth_port}/ for OAuth "
                f"({exc}). Free that port or change oauth_port in "
                "gmail_client.get_gmail_service, then retry."
            ) from exc
        TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")
        logger.info("OAuth complete; token saved to %s", TOKEN_PATH.name)

    return build("gmail", "v1", credentials=creds, cache_discovery=False)


class GmailClient:
    def __init__(self, service: Resource | None = None) -> None:
        self.service = service or get_gmail_service()
        self._label_id_by_name: dict[str, str] = {}
        self._label_name_by_id: dict[str, str] = {}
        self._labels_loaded = False

    def _users(self) -> Any:
        return self.service.users()

    def _refresh_label_cache(self) -> None:
        def _list() -> dict[str, Any]:
            return self._users().labels().list(userId="me").execute()

        result = _with_retry(_list, what="labels.list")
        self._label_id_by_name.clear()
        self._label_name_by_id.clear()
        for label in result.get("labels", []):
            name = label["name"]
            lid = label["id"]
            self._label_id_by_name[name] = lid
            self._label_name_by_id[lid] = name
        self._labels_loaded = True
        logger.debug("Cached %s Gmail labels", len(self._label_id_by_name))

    def ensure_labels_cached(self) -> None:
        if not self._labels_loaded:
            self._refresh_label_cache()

    def get_or_create_label_id(self, label_name: str) -> str:
        """Return label id, creating the user label if it does not exist."""
        self.ensure_labels_cached()
        if label_name in self._label_id_by_name:
            return self._label_id_by_name[label_name]

        body = {
            "name": label_name,
            "labelListVisibility": "labelShow",
            "messageListVisibility": "show",
        }

        def _create() -> dict[str, Any]:
            return self._users().labels().create(userId="me", body=body).execute()

        created = _with_retry(_create, what=f"labels.create({label_name})")
        label_id = created["id"]
        self._label_id_by_name[label_name] = label_id
        self._label_name_by_id[label_id] = label_name
        logger.info("Created Gmail label %r (id=%s)", label_name, label_id)
        return label_id

    def list_message_ids(self, query: str, *, limit: int | None = None) -> list[str]:
        """List message IDs matching query (does not fetch full messages)."""
        ids: list[str] = []
        page_token: str | None = None

        while True:
            remaining = None if limit is None else max(limit - len(ids), 0)
            if remaining == 0:
                break
            page_size = 100 if remaining is None else min(100, remaining)

            def _list(token: str | None = page_token, size: int = page_size) -> dict:
                kwargs: dict[str, Any] = {
                    "userId": "me",
                    "q": query,
                    "maxResults": size,
                }
                if token:
                    kwargs["pageToken"] = token
                return self._users().messages().list(**kwargs).execute()

            result = _with_retry(_list, what="messages.list")
            for item in result.get("messages", []) or []:
                ids.append(item["id"])
                if limit is not None and len(ids) >= limit:
                    return ids
            page_token = result.get("nextPageToken")
            if not page_token:
                break

        return ids

    def get_message(self, message_id: str, *, max_body_chars: int = 4000) -> EmailMessage:
        def _get() -> dict[str, Any]:
            return (
                self._users()
                .messages()
                .get(userId="me", id=message_id, format="full")
                .execute()
            )

        raw = _with_retry(_get, what=f"messages.get({message_id})")
        headers = {
            h["name"].lower(): h["value"]
            for h in raw.get("payload", {}).get("headers", [])
        }
        _, sender = parseaddr(headers.get("from", ""))
        subject = headers.get("subject", "(no subject)")
        body = _extract_plain_text(raw.get("payload", {}))
        if len(body) > max_body_chars:
            body = body[:max_body_chars] + "\n…[truncated]"

        label_ids = frozenset(raw.get("labelIds", []) or [])
        return EmailMessage(
            id=raw["id"],
            thread_id=raw.get("threadId", ""),
            sender=sender or headers.get("from", ""),
            subject=subject,
            body=body,
            label_ids=label_ids,
        )

    def managed_label_names_on_message(self, message: EmailMessage) -> set[str]:
        """Return which of our managed category labels are already on the message."""
        self.ensure_labels_cached()
        names: set[str] = set()
        for lid in message.label_ids:
            name = self._label_name_by_id.get(lid)
            if name in MANAGED_LABELS:
                names.add(name)
        return names

    def apply_label_and_archive(self, message_id: str, label_name: str) -> None:
        """Add category label and remove INBOX (archive). Never deletes."""
        label_id = self.get_or_create_label_id(label_name)
        body = {
            "addLabelIds": [label_id],
            "removeLabelIds": ["INBOX"],
        }

        def _modify() -> dict[str, Any]:
            return (
                self._users()
                .messages()
                .modify(userId="me", id=message_id, body=body)
                .execute()
            )

        _with_retry(_modify, what=f"messages.modify({message_id})")
        logger.info(
            "Labeled + archived message %s as %r", message_id, label_name
        )


def _decode_b64url(data: str) -> str:
    padded = data + "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(padded.encode("utf-8")).decode(
        "utf-8", errors="replace"
    )


def _extract_plain_text(payload: dict[str, Any]) -> str:
    """Walk MIME tree; prefer text/plain, fall back to stripped text/html."""
    mime = payload.get("mimeType", "")
    body = payload.get("body", {}) or {}
    data = body.get("data")

    if mime == "text/plain" and data:
        return _decode_b64url(data).strip()

    parts = payload.get("parts") or []
    plain_chunks: list[str] = []
    html_chunks: list[str] = []

    def walk(part: dict[str, Any]) -> None:
        part_mime = part.get("mimeType", "")
        part_data = (part.get("body") or {}).get("data")
        if part_mime == "text/plain" and part_data:
            plain_chunks.append(_decode_b64url(part_data))
        elif part_mime == "text/html" and part_data:
            html_chunks.append(_decode_b64url(part_data))
        for child in part.get("parts") or []:
            walk(child)

    if parts:
        for p in parts:
            walk(p)
    elif mime == "text/html" and data:
        html_chunks.append(_decode_b64url(data))

    if plain_chunks:
        return "\n".join(c.strip() for c in plain_chunks if c.strip()).strip()

    if html_chunks:
        # Lightweight HTML strip — enough for LLM triage context.
        import re

        text = "\n".join(html_chunks)
        text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
        text = re.sub(r"(?s)<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    return ""
