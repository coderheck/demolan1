
import os
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

import folium
from folium.plugins import HeatMap
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse

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

@app.post("/api/annotate-latest", status_code=201)
def annotate_latest(
    device_id: str,
    density: str,
    confidence: float,
    telemetry: str | None = None,
    x_api_key: str | None = Header(default=None),
):
    check_key(x_api_key)

    if density not in {"dense", "medium", "sparse", "unknown"}:
        raise HTTPException(status_code=400, detail="Invalid forest type")

    if not 0 <= confidence <= 1:
        raise HTTPException(status_code=400, detail="Invalid confidence")

    now = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.execute("""
            INSERT INTO records (
                received_at, device_id, recorded_at,
                latitude, longitude, density, confidence, telemetry
            )
            SELECT ?, device_id, ?, latitude, longitude, ?, ?, ?
            FROM records
            WHERE device_id = ?
              AND latitude IS NOT NULL
              AND longitude IS NOT NULL
            ORDER BY id DESC
            LIMIT 1
        """, (
            now, now, density, confidence, telemetry, device_id
        ))

        if cursor.rowcount == 0:
            raise HTTPException(
                status_code=404,
                detail="No GPS record found for this device",
            )

    return {"status": "stored"}

@app.get("/heatmap")
def forest_heatmap(
    forest_type: str = "all",
    start: str | None = None,
    end: str | None = None,
):
    allowed_types = {"dense", "medium", "sparse", "unknown", "all"}
    if forest_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Invalid forest_type")

    query = """
        SELECT id, received_at, recorded_at, device_id,
               latitude, longitude, density, confidence, telemetry
        FROM records
        WHERE latitude IS NOT NULL
          AND longitude IS NOT NULL
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
        return {
            "message": "No GPS records match these filters.",
            "count": 0,
        }

    # Center the map on the collected GPS points.
    points = [
        (row["latitude"], row["longitude"])
        for row in rows
    ]

    center = [
        sum(p[0] for p in points) / len(points),
        sum(p[1] for p in points) / len(points),
    ]

    m = folium.Map(location=center, zoom_start=15)

    colors = {
        "dense": "red",
        "medium": "orange",
        "sparse": "green",
        "unknown": "gray",
    }

    # Create a separate heatmap layer for each forest type.
    for kind, color in colors.items():
        matching = [
            row for row in rows
            if row["density"] == kind
        ]

        if not matching:
            continue

        layer = folium.FeatureGroup(
            name=f"{kind.title()} forest ({len(matching)} records)",
            show=True,
        )

        # Heatmap intensity is weighted by classification confidence.
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
            gradient={
                0.2: "blue",
                0.5: "lime",
                0.8: "yellow",
                1.0: color,
            },
        ).add_to(layer)

        # Show individual records and their timestamps on click.
        for row in matching:
            timestamp = row["recorded_at"] or row["received_at"]
            popup = folium.Popup(
                f"""
                <b>Forest type:</b> {row['density']}<br>
                <b>Confidence:</b> {row['confidence']}<br>
                <b>Timestamp:</b> {timestamp}<br>
                <b>Device:</b> {row['device_id']}<br>
                <b>Latitude:</b> {row['latitude']}<br>
                <b>Longitude:</b> {row['longitude']}<br>
                <b>Telemetry:</b> {row['telemetry'] or 'N/A'}
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

    # Return a standalone HTML map.
    output_file = "forest_heatmap.html"
    m.save(output_file)

    # return FileResponse(
    #     output_file,
    #     media_type="text/html",
    #     filename="forest_heatmap.html",
    # )
    return HTMLResponse(
        content=m.get_root().render(), 
        # media_type="text/html"
    )