from __future__ import annotations

from typing import Any

from app.ingestion.public_sources import load_public_extracts, load_source_registry
from app.utils.io import write_json
from app.utils.paths import TIMELINE_PATH


def build_timeline() -> list[dict[str, Any]]:
    registry = load_source_registry().set_index("id")
    corpus = load_public_extracts()
    events: list[dict[str, Any]] = []
    for document in corpus.get("documents", []):
        source_meta = registry.loc[document["source_id"]].to_dict()
        for event in document.get("event_candidates", []):
            events.append(
                {
                    "date": event["date"],
                    "description": event["description"],
                    "tags": event.get("tags", []),
                    "source_id": document["source_id"],
                    "source_title": document["title"],
                    "source_url": source_meta["url"],
                    "reliability": source_meta["reliability"],
                    "entities": document.get("entities", []),
                }
            )
    events = sorted(events, key=lambda item: item["date"])
    write_json(TIMELINE_PATH, events)
    return events

