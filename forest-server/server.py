import os
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal

import folium
from folium.plugins import HeatMap
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

DB_FILE = "forest.db"
API_KEY = os.environ.get("FOREST_API_KEY")

if not API_KEY:
    raise RuntimeError("Set the FOREST_API_KEY environment variable first.")


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        columns = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(records)"
            ).fetchall()
        }

        if not columns:
            conn.execute("""
                CREATE TABLE records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    received_at TEXT NOT NULL,
                    recorded_at TEXT,
                    latitude REAL,
                    longitude REAL,
                    density TEXT,
                    confidence REAL
                )
            """)
            return

        # Migrate the old database if it still contains
        # device_id or telemetry, preserving useful records.
        required = {
            "id", "received_at", "recorded_at",
            "latitude", "longitude", "density", "confidence"
        }

        if columns != required:
            conn.execute("""
                CREATE TABLE records_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    received_at TEXT NOT NULL,
                    recorded_at TEXT,
                    latitude REAL,
                    longitude REAL,
                    density TEXT,
                    confidence REAL
                )
            """)

            def expr(column, fallback="NULL"):
                return column if column in columns else fallback

            received_expr = expr("received_at", "NULL")
            recorded_expr = expr("recorded_at")
            latitude_expr = expr("latitude")
            longitude_expr = expr("longitude")
            density_expr = expr("density")
            confidence_expr = expr("confidence")
            id_expr = expr("id", "NULL")

            conn.execute(f"""
                INSERT INTO records_new (
                    id, received_at, recorded_at,
                    latitude, longitude, density, confidence
                )
                SELECT
                    {id_expr},
                    COALESCE({received_expr}, ?),
                    {recorded_expr},
                    {latitude_expr},
                    {longitude_expr},
                    {density_expr},
                    {confidence_expr}
                FROM records
            """, (now_utc(),))

            conn.execute("DROP TABLE records")
            conn.execute(
                "ALTER TABLE records_new RENAME TO records"
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Forest Telemetry API", lifespan=lifespan)


class Record(BaseModel):
    recorded_at: datetime | None = None
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


def check_key(x_api_key: str | None):
    if x_api_key is None or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health")
def health():
    return {"status": "ok", "service": "forest-telemetry"}


# ESP32: save a GPS fix without a forest classification.
@app.post("/api/records", status_code=201)
def add_record(
    record: Record,
    x_api_key: str | None = Header(default=None),
):
    check_key(x_api_key)

    received_at = now_utc()
    recorded_at = (
        record.recorded_at.isoformat()
        if record.recorded_at is not None
        else None
    )

    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.execute("""
            INSERT INTO records (
                received_at, recorded_at,
                latitude, longitude, density, confidence
            ) VALUES (?, ?, ?, ?, NULL, NULL)
        """, (
            received_at,
            recorded_at,
            record.latitude,
            record.longitude,
        ))

        record_id = cursor.lastrowid

    return {
        "status": "stored",
        "id": record_id,
        "received_at": received_at,
    }


# Laptop: associate a YOLO result with the latest unclassified GPS fix.
@app.post("/api/annotate-latest", status_code=201)
def annotate_latest(
    density: Literal["dense", "medium", "sparse", "unknown"],
    confidence: float = Query(ge=0, le=1),
    x_api_key: str | None = Header(default=None),
):
    check_key(x_api_key)

    timestamp = now_utc()

    with sqlite3.connect(DB_FILE) as conn:
        gps = conn.execute("""
            SELECT id, recorded_at
            FROM records
            WHERE density IS NULL
              AND latitude IS NOT NULL
              AND longitude IS NOT NULL
            ORDER BY id DESC
            LIMIT 1
        """).fetchone()

        if gps is None:
            raise HTTPException(
                status_code=404,
                detail="No unclassified GPS fix is available",
            )

        gps_id, gps_timestamp = gps

        cursor = conn.execute("""
            UPDATE records
            SET density = ?,
                confidence = ?,
                recorded_at = COALESCE(recorded_at, ?)
            WHERE id = ?
              AND density IS NULL
        """, (
            density,
            confidence,
            timestamp,
            gps_id,
        ))

        if cursor.rowcount == 0:
            raise HTTPException(
                status_code=409,
                detail="GPS record was already classified",
            )

    return {
        "status": "stored",
        "id": gps_id,
        "recorded_at": gps_timestamp or timestamp,
        "density": density,
        "confidence": confidence,
    }


@app.get("/api/records")
def list_records(
    limit: int = Query(default=100, ge=1, le=1000),
    x_api_key: str | None = Header(default=None),
):
    check_key(x_api_key)

    with sqlite3.connect(DB_FILE) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT id, received_at, recorded_at,
                   latitude, longitude, density, confidence
            FROM records
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()

    return [dict(row) for row in rows]


@app.get("/heatmap", response_class=HTMLResponse)
def forest_heatmap(
    forest_type: str = "all",
    start: str | None = None,
    end: str | None = None,
):
    allowed_types = {"dense", "medium", "sparse", "unknown", "all"}

    if forest_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Invalid forest_type",
        )

    query = """
        SELECT id, received_at, recorded_at,
               latitude, longitude, density, confidence
        FROM records
        WHERE latitude IS NOT NULL
          AND longitude IS NOT NULL
          AND density IS NOT NULL
    """
    params = []

    if forest_type != "all":
        query += " AND density = ?"
        params.append(forest_type)

    if start:
        query += " AND COALESCE(recorded_at, received_at) >= ?"
        params.append(start)

    if end:
        query += " AND COALESCE(recorded_at, received_at) <= ?"
        params.append(end)

    query += " ORDER BY id ASC"

    with sqlite3.connect(DB_FILE) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, params).fetchall()

    if not rows:
        return HTMLResponse(
            "<h2>No classified GPS observations match these filters.</h2>"
        )

    center = [
        sum(row["latitude"] for row in rows) / len(rows),
        sum(row["longitude"] for row in rows) / len(rows),
    ]

    m = folium.Map(location=center, zoom_start=15)

    colors = {
        "dense": "red",
        "medium": "orange",
        "sparse": "green",
        "unknown": "gray",
    }

    for kind, color in colors.items():
        matching = [
            row for row in rows
            if row["density"] == kind
        ]

        if not matching:
            continue

        layer = folium.FeatureGroup(
            name=f"{kind.title()} forest ({len(matching)} records)"
        )

        heat_data = [
            [
                row["latitude"],
                row["longitude"],
                max(0.1, row["confidence"] or 0.5),
            ]
            for row in matching
        ]

        HeatMap(
            heat_data,
            radius=25,
            blur=18,
            min_opacity=0.3,
        ).add_to(layer)

        for row in matching:
            timestamp = row["recorded_at"] or row["received_at"]

            popup = folium.Popup(
                f"""
                <b>Forest type:</b> {row['density']}<br>
                <b>Confidence:</b> {row['confidence']}<br>
                <b>Timestamp:</b> {timestamp}<br>
                <b>Latitude:</b> {row['latitude']}<br>
                <b>Longitude:</b> {row['longitude']}
                """,
                max_width=350,
            )

            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=4,
                color=color,
                fill=True,
                fill_opacity=0.85,
                popup=popup,
                tooltip=f"{kind.title()} - {timestamp}",
            ).add_to(layer)

        layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    return HTMLResponse(content=m.get_root().render())