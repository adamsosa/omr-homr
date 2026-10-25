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


def test_post_omr_fails_closed_503_when_homr_missing(monkeypatch: object) -> None:
    monkeypatch.setattr("shutil.which", lambda _cmd: None)  # type: ignore[attr-defined]
    png_bytes = _make_synthetic_score_png()
    resp = client.post(
        "/omr",
        files={"image": ("synthetic_page.png", png_bytes, "image/png")},
    )
    assert resp.status_code == 503
    assert resp.headers.get("retry-after") == "5"
    body = resp.json()
    assert body["error_code"] == "OMR_ENGINE_UNAVAILABLE"
    assert body["retryable"] is True


def test_post_omr_extracts_musicxml_and_normalized_measure_boxes(monkeypatch: object) -> None:
    import pathlib as _pathlib
    import subprocess as _subprocess

    monkeypatch.setattr("shutil.which", lambda _cmd: "/usr/local/bin/homr")  # type: ignore[attr-defined]

    def _fake_subprocess_run(
        cmd: list[str], cwd: str, **kwargs: object
    ) -> _subprocess.CompletedProcess[str]:
        out_xml = _pathlib.Path(cwd) / "page.musicxml"
        out_xml.write_text(
            '<?xml version="1.0" encoding="UTF-8"?><score-partwise version="4.0">'
            '<part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>'
            '<part id="P1"><measure number="1"><attributes><divisions>1</divisions>'
            "<time><beats>4</beats><beat-type>4</beat-type></time></attributes>"
            "<note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><type>whole</type></note>"
            "</measure></part></score-partwise>",
            encoding="utf-8",
        )
        return _subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", _fake_subprocess_run)  # type: ignore[attr-defined]

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
    assert body["engine"] == "homr-cli"
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


def test_post_omr_timeout_returns_504_with_retry_after(monkeypatch: object) -> None:
    import subprocess as _subprocess

    monkeypatch.setattr("shutil.which", lambda _cmd: "/usr/local/bin/homr")  # type: ignore[attr-defined]

    def _timeout_run(cmd: list[str], **_kwargs: object) -> None:
        raise _subprocess.TimeoutExpired(cmd=cmd, timeout=5.0)

    monkeypatch.setattr("subprocess.run", _timeout_run)  # type: ignore[attr-defined]

    png_bytes = _make_synthetic_score_png()
    resp = client.post(
        "/omr",
        files={"image": ("synthetic_page.png", png_bytes, "image/png")},
    )
    assert resp.status_code == 504
    assert resp.headers.get("retry-after") == "5"
    assert resp.json()["error_code"] == "OMR_UPSTREAM_TIMEOUT"


def test_post_omr_rejects_oversized_upload() -> None:
    huge = b"x" * (4 * 1024 * 1024 + 10)
    resp = client.post(
        "/omr",
        files={"image": ("huge.png", huge, "image/png")},
    )
    assert resp.status_code == 413
    assert resp.json()["error_code"] == "IMAGE_TOO_LARGE"
