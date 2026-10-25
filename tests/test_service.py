# SPDX-License-Identifier: AGPL-3.0-only
"""Unit tests for the isolated AGPL-3.0 omr-homr FastAPI microservice."""
from __future__ import annotations

import cv2
import numpy as np
from fastapi.testclient import TestClient

from omr_homr_service.app import app

client = TestClient(app)


def _make_synthetic_score_png(
    width: int = 1200,
    height: int = 1600,
    staff_spacing: int = 12,
    num_systems: int = 2,
    staves_per_system: int = 3,
    measures_per_system: int = 4,
) -> bytes:
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    x0, x1 = 100, width - 100

    for sys_idx in range(num_systems):
        sys_top = 140 + sys_idx * 600
        sys_bottom = sys_top
        for st_idx in range(staves_per_system):
            st_top = sys_top + st_idx * (staff_spacing * 9)
            for line_i in range(5):
                y = st_top + line_i * staff_spacing
                cv2.line(img, (x0, y), (x1, y), (0, 0, 0), 2)
                sys_bottom = max(sys_bottom, y)

        # Draw vertical barlines across the system
        measure_w = (x1 - x0) // measures_per_system
        for m_i in range(measures_per_system + 1):
            bx = x0 + m_i * measure_w
            cv2.line(img, (bx, sys_top), (bx, sys_bottom), (0, 0, 0), 3)

    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_source_endpoint_returns_agpl_metadata() -> None:
    resp = client.get("/source")
    assert resp.status_code == 200
    data = resp.json()
    assert data["license"] == "AGPL-3.0-only"
    assert "repo_url" in data and data["repo_url"].startswith("http")
    assert "commit_sha" in data and len(data["commit_sha"]) > 0


def test_health_endpoint() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_post_omr_extracts_musicxml_and_normalized_measure_boxes() -> None:
    png_bytes = _make_synthetic_score_png(
        width=1200,
        height=1600,
        staff_spacing=12,
        num_systems=2,
        staves_per_system=3,
        measures_per_system=4,
    )
    resp = client.post(
        "/omr",
        files={"image": ("synthetic_page.png", png_bytes, "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "<score-partwise" in body["musicxml"]
    assert 10.0 <= body["staff_spacing_px"] <= 14.0
    boxes = body["measure_boxes"]
    assert len(boxes) == 8
    for idx, b in enumerate(boxes):
        assert b["measure_index"] == idx
        assert 0.0 <= b["x"] <= 1.0
        assert 0.0 <= b["y"] <= 1.0
        assert 0.0 < b["w"] <= 1.0
        assert 0.0 < b["h"] <= 1.0
        assert len(b["staff_boxes"]) == 3
