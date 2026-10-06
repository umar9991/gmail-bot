"""Gmail auto-triage bot — entrypoint.

Fetch unread inbox mail → classify with Groq → label + archive (unless DRY_RUN).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from classifier import EmailClassifier
from config import (
    BATCH_SIZE,          
    CONFIDENCE_THRESHOLD,
    DRY_RUN,
    GMAIL_QUERY,
    LOG_LEVEL,
    MAX_BODY_CHARS,
    NONE_CATEGORY,
    REQUEST_DELAY_SECONDS,
)
from gmail_client import GmailClient


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gmail auto-triage bot (classify + label/archive)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N emails (useful for first-run tests)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help=f"Gmail search query (default: {GMAIL_QUERY!r})",
    )
    return parser.parse_args(argv)


def triage(
    client: GmailClient,
    classifier: EmailClassifier,
    query: str,
    limit: int | None,
) -> int:
    """Classify matching messages; apply labels only when DRY_RUN is false."""
    logger = logging.getLogger("main")
    logger.info(
        "DRY_RUN=%s | threshold=%.2f | request_delay=%.1fs | query=%r | limit=%s",
        DRY_RUN,
        CONFIDENCE_THRESHOLD,
        REQUEST_DELAY_SECONDS,
        query,
        limit,
    )

    ids = client.list_message_ids(query, limit=limit)
    logger.info("Found %s message(s) matching query", len(ids))
    if not ids:
        logger.info("Inbox has no matching unread messages.")
        return 0

    processed = 0
    skipped = 0
    applied = 0
    left_alone = 0
    seen: set[str] = set()
    classified_count = 0

    for start in range(0, len(ids), BATCH_SIZE):
        batch = ids[start : start + BATCH_SIZE]
        for mid in batch:
            if mid in seen:
                continue
            seen.add(mid)

            msg = client.get_message(mid, max_body_chars=MAX_BODY_CHARS)
            existing = client.managed_label_names_on_message(msg)
            if existing:
                logger.info(
                    "SKIP already labeled | id=%s | subject=%r | labels=%s",
                    msg.id,
                    msg.subject,
                    sorted(existing),
                )
                skipped += 1
                continue

            # Pace Groq calls; retry/backoff in classifier still handles 429s.
            if classified_count > 0 and REQUEST_DELAY_SECONDS > 0:
                time.sleep(REQUEST_DELAY_SECONDS)

            try:
                result = classifier.classify(
                    sender=msg.sender,
                    subject=msg.subject,
                    body=msg.body,
                )
            except Exception:
                logger.exception(
                    "CLASSIFY FAILED | id=%s | subject=%r — leaving untouched",
                    msg.id,
                    msg.subject,
                )
                left_alone += 1
                continue

            classified_count += 1
            logger.info(
                "DECISION | id=%s | subject=%r | category=%r | "
                "confidence=%.2f | reason=%s",
                msg.id,
                msg.subject,
                result.category,
                result.confidence,
                result.reason,
            )

            should_apply = (
                result.category != NONE_CATEGORY
                and result.confidence >= CONFIDENCE_THRESHOLD
            )
            if not should_apply:
                left_alone += 1
                processed += 1
                continue

            if DRY_RUN:
                logger.info(
                    "DRY_RUN would label+archive | id=%s | label=%r",
                    msg.id,
                    result.category,
                )
            else:
                client.apply_label_and_archive(msg.id, result.category)
                applied += 1

            processed += 1

    logger.info(
        "Done | processed=%s skipped=%s applied=%s left_untouched=%s dry_run=%s",
        processed,
        skipped,
        applied,
        left_alone,
        DRY_RUN,
    )
    return processed


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(LOG_LEVEL)
    logger = logging.getLogger("main")

    try:
        client = GmailClient()
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1
    except Exception:
        logger.exception("Gmail authentication failed")
        return 1

    try:
        classifier = EmailClassifier()
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    query = args.query or GMAIL_QUERY
    try:
        triage(client, classifier, query, args.limit)
    except Exception:
        logger.exception("Triage run failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
