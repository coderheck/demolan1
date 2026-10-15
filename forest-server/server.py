
import os
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

DB_FILE = "forest.db"
API_KEY = os.environ.get("FOREST_API_KEY")

if not API_KEY:
    raise RuntimeError("Set the FOREST_API_KEY environment variable first.")


def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                received_at TEXT NOT NULL,
                device_id TEXT NOT NULL,
                recorded_at TEXT,
                latitude REAL,
                longitude REAL,
                density TEXT,
                confidence REAL,
                telemetry TEXT
            )
        """)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Forest Telemetry API", lifespan=lifespan)


class Record(BaseModel):
    device_id: str = Field(min_length=1, max_length=64)
    recorded_at: datetime | None = None
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    density: Literal["dense", "medium", "sparse", "unknown"] | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    telemetry: str | None = Field(default=None, max_length=1000)


def check_key(x_api_key: str | None):
    if x_api_key is None or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health")
def health():
    return {"status": "ok", "service": "forest-telemetry"}


@app.post("/api/records", status_code=201)
def add_record(
    record: Record,
    x_api_key: str | None = Header(default=None),
):
    check_key(x_api_key)

    received_at = datetime.now(timezone.utc).isoformat()
    recorded_at = (
        record.recorded_at.isoformat()
        if record.recorded_at is not None
        else None
    )

    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.execute("""
            INSERT INTO records (
                received_at, device_id, recorded_at,
                latitude, longitude, density, confidence, telemetry
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            received_at,
            record.device_id,
            recorded_at,
            record.latitude,
            record.longitude,
            record.density,
            record.confidence,
            record.telemetry,
        ))
        record_id = cursor.lastrowid

    return {
        "status": "stored",
        "id": record_id,
        "received_at": received_at,
    }


@app.get("/api/records")
def list_records(
    limit: int = 100,
    x_api_key: str | None = Header(default=None),
):
    check_key(x_api_key)

    if not 1 <= limit <= 1000:
        raise HTTPException(status_code=400, detail="limit must be 1–1000")

    with sqlite3.connect(DB_FILE) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT * FROM records
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()

    return [dict(row) for row in rows]

