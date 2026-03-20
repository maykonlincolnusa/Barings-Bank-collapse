from __future__ import annotations

from typing import Any

import pandas as pd

from app.utils.io import read_json
from app.utils.paths import MARKET_DATA_PATH, PUBLIC_EXTRACTS_PATH, SOURCE_REGISTRY_PATH


def load_source_registry() -> pd.DataFrame:
    frame = pd.read_csv(SOURCE_REGISTRY_PATH)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def load_public_extracts() -> dict[str, Any]:
    return read_json(PUBLIC_EXTRACTS_PATH, default={"documents": []})


def load_market_data() -> pd.DataFrame:
    frame = pd.read_csv(MARKET_DATA_PATH)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values("date").reset_index(drop=True)

