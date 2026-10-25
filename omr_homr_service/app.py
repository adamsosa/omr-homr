# SPDX-License-Identifier: AGPL-3.0-only
"""FastAPI HTTP wrapper around homr CLI subprocess with OpenCV measure-box extraction."""

from __future__ import annotations

import io
import json
import os
import pathlib
import shutil
import subprocess
import tempfile
import time
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MAX_UPLOAD_BYTES = 4 * 1024 * 1024
MAX_DIMENSION_PX = 8000
MAX_TOTAL_PIXELS = 25_000_000

app = FastAPI(
    title="omr-homr Service",
    version="1.0.0",
    description="Isolated AGPL-3.0 Optical Music Recognition service wrapping homr CLI subprocess.",
)


class OMREngineUnavailableError(RuntimeError):
    """Raised when the homr CLI binary is missing or exits non-zero."""


class OMREngineTimeoutError(TimeoutError):
    """Raised when the homr CLI subprocess exceeds the request deadline."""


class BoundingBoxItem(BaseModel):
    measure_index: int = Field(ge=0)
    system_index: int = Field(ge=0)
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    w: float = Field(ge=0.0, le=1.0)
    h: float = Field(ge=0.0, le=1.0)
    staff_boxes: list[dict[str, float]] = Field(default_factory=list)


class OMRResponse(BaseModel):
    musicxml: str
    measure_boxes: list[BoundingBoxItem]
    staff_spacing_px: float
    engine: str = "homr-cli"


class SourceResponse(BaseModel):
    repo_url: str
    commit_sha: str
    license: str = "AGPL-3.0-only"


def get_commit_sha() -> str:
    env_sha = os.environ.get("OMR_HOMR_COMMIT_SHA")
    if env_sha and env_sha != "unknown":
        return env_sha
    try:
        res = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2.0,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return "dev-uncommitted"


def _clamp01(val: float) -> float:
    return float(max(0.0, min(1.0, round(val, 6))))


def _default_single_box() -> list[BoundingBoxItem]:
    return [
        BoundingBoxItem(
            measure_index=0,
            system_index=0,
            x=0.05,
            y=0.08,
            w=0.90,
            h=0.84,
            staff_boxes=[{"x": 0.05, "y": 0.08, "w": 0.90, "h": 0.84}],
        )
    ]


def _vertical_line_response(gray: np.ndarray) -> np.ndarray:
    h, _ = gray.shape[:2]
    line_resp = np.zeros_like(gray, dtype=np.float32)
    gray_f = gray.astype(np.float32)
    for d in (1, 2, 3):
        if h > 2 * d:
            resp = 0.5 * (gray_f[: -2 * d, :] + gray_f[2 * d :, :]) - gray_f[d:-d, :]
            line_resp[d:-d, :] = np.maximum(line_resp[d:-d, :], np.maximum(0.0, resp))
    return line_resp


def _estimate_staff_spacing_strip_ac(gray: np.ndarray) -> float:
    """Estimate staff-line spacing (px) via vertical second-derivative strip autocorrelation."""
    h, w = gray.shape[:2]
    line_resp = _vertical_line_response(gray)

    n_strips = 8
    x0, x1 = int(w * 0.12), int(w * 0.88)
    strip_w = max(8, (x1 - x0) // n_strips)
    max_lag = min(36, max(16, h // 12))
    ac = np.zeros(max_lag + 1, dtype=np.float64)

    for s in range(n_strips):
        sx0, sx1 = x0 + s * strip_w, min(w, x0 + (s + 1) * strip_w)
        if sx1 > sx0:
            prof = np.mean(line_resp[:, sx0:sx1], axis=1)
            prof = prof - np.mean(prof)
            for lag in range(4, max_lag + 1):
                ac[lag] += float(np.sum(prof[:-lag] * prof[lag:]))

    if np.max(ac[4:]) <= 0.0:
        return 12.0

    best_lag = 4 + int(np.argmax(ac[4:]))
    if 4 < best_lag < max_lag:
        y_m1, y_0, y_p1 = ac[best_lag - 1], ac[best_lag], ac[best_lag + 1]
        denom = y_m1 - 2.0 * y_0 + y_p1
        delta = 0.5 * (y_m1 - y_p1) / denom if abs(denom) > 1e-9 else 0.0
        return float(best_lag + max(-0.5, min(0.5, delta)))
    return float(best_lag)


def _collapse_line_centers(line_rows: np.ndarray) -> list[float]:
    if len(line_rows) == 0:
        return []
    centers: list[float] = []
    start = prev = int(line_rows[0])
    for r in line_rows[1:]:
        r_int = int(r)
        if r_int <= prev + 2:
            prev = r_int
        else:
            centers.append(0.5 * (start + prev))
            start = prev = r_int
    centers.append(0.5 * (start + prev))
    return centers


def _group_centers_into_staves(
    line_centers: list[float], staff_spacing_px: float, h: int
) -> list[tuple[int, int]]:
    if len(line_centers) < 3:
        return []
    staves: list[tuple[int, int]] = []
    curr_group = [line_centers[0]]
    max_gap = staff_spacing_px * 2.4
    for lc in line_centers[1:]:
        if (lc - curr_group[-1]) <= max_gap:
            curr_group.append(lc)
        else:
            if len(curr_group) >= 3:
                y0 = max(0, int(round(curr_group[0] - staff_spacing_px * 0.5)))
                y1 = min(h - 1, int(round(curr_group[-1] + staff_spacing_px * 0.5)))
                if y1 > y0:
                    staves.append((y0, y1))
            curr_group = [lc]
    if len(curr_group) >= 3:
        y0 = max(0, int(round(curr_group[0] - staff_spacing_px * 0.5)))
        y1 = min(h - 1, int(round(curr_group[-1] + staff_spacing_px * 0.5)))
        if y1 > y0:
            staves.append((y0, y1))
    return staves


def _comb_filter_staves(gray: np.ndarray, staff_spacing_px: float) -> list[tuple[int, int]]:
    h, w = gray.shape[:2]
    sp_int = max(4, int(round(staff_spacing_px)))
    line_resp = _vertical_line_response(gray)
    prof = np.mean(line_resp[:, int(w * 0.2) : int(w * 0.8)], axis=1)
    comb = np.zeros(h, dtype=np.float32)
    for k in (-2, -1, 0, 1, 2):
        comb += np.roll(prof, k * sp_int)
    min_dist = max(18, int(staff_spacing_px * 6.0))
    half_staff = max(6, int(round(staff_spacing_px * 2.2)))
    staves: list[tuple[int, int]] = []
    thresh = float(np.mean(comb) * 1.15)
    for y in range(half_staff + 4, h - half_staff - 4):
        w0, w1 = max(0, y - min_dist // 2), min(h, y + min_dist // 2 + 1)
        if (
            comb[y] > thresh
            and comb[y] >= np.max(comb[w0:w1])
            and (not staves or (y - half_staff - staves[-1][0]) >= min_dist)
        ):
            staves.append((max(0, y - half_staff), min(h - 1, y + half_staff)))
    return staves


def _group_staves_into_systems(
    staves: list[tuple[int, int]], staff_spacing_px: float
) -> list[list[tuple[int, int]]]:
    if len(staves) == 1:
        return [[staves[0]]]
    inter_gaps = [staves[i + 1][0] - staves[i][1] for i in range(len(staves) - 1)]
    median_gap = float(np.median(inter_gaps)) if inter_gaps else staff_spacing_px * 6.0
    systems: list[list[tuple[int, int]]] = []
    curr_sys = [staves[0]]
    for i in range(1, len(staves)):
        gap = staves[i][0] - staves[i - 1][1]
        if len(curr_sys) < 3 and gap <= max(median_gap * 1.35, staff_spacing_px * 9.0):
            curr_sys.append(staves[i])
        else:
            systems.append(curr_sys)
            curr_sys = [staves[i]]
    if curr_sys:
        systems.append(curr_sys)
    return systems


def _detect_bar_x_positions(
    binary: np.ndarray,
    sys_y0: int,
    sys_y1: int,
    x_left: int,
    x_right: int,
    staff_spacing_px: float,
) -> list[int]:
    sys_h = max(1, sys_y1 - sys_y0)
    vert_len = max(12, int(staff_spacing_px * 3.2))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_len))
    vert_lines = cv2.morphologyEx(binary[sys_y0:sys_y1, :], cv2.MORPH_OPEN, vert_kernel)

    col_proj = np.sum(vert_lines > 0, axis=0) / float(sys_h)
    bar_cols = np.where(col_proj > 0.32)[0]
    min_measure_w = max(int(staff_spacing_px * 5.0), int((x_right - x_left) * 0.14))

    bar_x: list[int] = []
    for c in bar_cols:
        c_int = int(c)
        if x_left - 10 <= c_int <= x_right + 10 and (
            not bar_x or (c_int - bar_x[-1]) >= min_measure_w
        ):
            bar_x.append(c_int)

    if len(bar_x) < 2:
        step_x = max(1, (x_right - x_left) // 4)
        return [x_left + k * step_x for k in range(5)]

    if bar_x[0] > x_left + min_measure_w:
        bar_x.insert(0, x_left)
    if bar_x[-1] < x_right - min_measure_w:
        bar_x.append(x_right)
    return bar_x


def _build_boxes_for_systems(
    binary: np.ndarray,
    systems: list[list[tuple[int, int]]],
    x_left: int,
    x_right: int,
    staff_spacing_px: float,
    w: int,
    h: int,
) -> list[BoundingBoxItem]:
    measure_boxes: list[BoundingBoxItem] = []
    global_m_idx = 0
    for sys_idx, sys_staves in enumerate(systems):
        sys_y0, sys_y1 = sys_staves[0][0], sys_staves[-1][1]
        bar_x = _detect_bar_x_positions(binary, sys_y0, sys_y1, x_left, x_right, staff_spacing_px)
        for b_i in range(len(bar_x) - 1):
            mx0, mx1 = bar_x[b_i], bar_x[b_i + 1]
            if (mx1 - mx0) >= int(staff_spacing_px * 2.5):
                s_boxes = [
                    {
                        "x": _clamp01(mx0 / float(w)),
                        "y": _clamp01(st_y0 / float(h)),
                        "w": _clamp01((mx1 - mx0) / float(w)),
                        "h": _clamp01((st_y1 - st_y0) / float(h)),
                    }
                    for (st_y0, st_y1) in sys_staves
                ]
                measure_boxes.append(
                    BoundingBoxItem(
                        measure_index=global_m_idx,
                        system_index=sys_idx,
                        x=_clamp01(mx0 / float(w)),
                        y=_clamp01(sys_y0 / float(h)),
                        w=_clamp01((mx1 - mx0) / float(w)),
                        h=_clamp01((sys_y1 - sys_y0) / float(h)),
                        staff_boxes=s_boxes,
                    )
                )
                global_m_idx += 1
    return measure_boxes


def detect_staff_and_measure_boxes_opencv(
    img_bgr: np.ndarray,
) -> tuple[float, list[BoundingBoxItem]]:
    """Detect staff-line spacing, staff systems, and measure bounding boxes via OpenCV morphology."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr.copy()
    h, w = gray.shape[:2]
    if h < 16 or w < 16:
        return 10.0, _default_single_box()

    block_size = max(15, (min(h, w) // 40) | 1)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, 12
    )
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, w // 25), 1))
    horiz_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel)

    row_proj = (np.sum(horiz_lines > 0, axis=1) / float(w)).astype(np.float64)
    line_centers = _collapse_line_centers(np.where(row_proj > 0.12)[0])

    staff_spacing_px = _estimate_staff_spacing_strip_ac(gray)
    if len(line_centers) >= 8:
        diffs = np.diff(line_centers)
        plausible = diffs[(diffs >= 4.0) & (diffs <= max(60.0, h * 0.05))]
        if len(plausible) >= 6 and abs(float(np.median(plausible)) - staff_spacing_px) <= 3.0:
            staff_spacing_px = float(np.median(plausible))

    staves = _group_centers_into_staves(line_centers, staff_spacing_px, h)
    if len(staves) < 2:
        staves = _comb_filter_staves(gray, staff_spacing_px)
    if not staves:
        return round(staff_spacing_px, 2), _default_single_box()

    systems = _group_staves_into_systems(staves, staff_spacing_px)
    active_cols = np.where(np.sum(horiz_lines > 0, axis=0) > max(2, len(staves)))[0]
    x_left = int(active_cols[0]) if len(active_cols) > 0 else int(w * 0.06)
    x_right = int(active_cols[-1]) if len(active_cols) > 0 else int(w * 0.94)
    if x_right - x_left < int(w * 0.3):
        x_left, x_right = int(w * 0.06), int(w * 0.94)

    measure_boxes = _build_boxes_for_systems(
        binary, systems, x_left, x_right, staff_spacing_px, w, h
    )
    return round(staff_spacing_px, 2), (measure_boxes or _default_single_box())


def _validate_image_dimensions(image_bytes: bytes) -> None:
    try:
        with Image.open(io.BytesIO(image_bytes)) as pil_img:
            width, height = pil_img.size
    except Exception as exc:
        raise ValueError("Unable to read image header") from exc
    if width <= 0 or height <= 0:
        raise ValueError("Invalid non-positive image dimensions")
    if (
        width > MAX_DIMENSION_PX
        or height > MAX_DIMENSION_PX
        or (width * height) > MAX_TOTAL_PIXELS
    ):
        raise ValueError(
            f"Image dimensions {width}x{height} exceed safety limit ({MAX_DIMENSION_PX}px / {MAX_TOTAL_PIXELS} pixels)"
        )


def run_omr_on_image_bytes(image_bytes: bytes, timeout_s: float = 115.0) -> OMRResponse:
    _validate_image_dimensions(image_bytes)
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unable to decode image bytes as a valid raster image")

    staff_spacing_px, opencv_boxes = detect_staff_and_measure_boxes_opencv(img)
    homr_bin = shutil.which("homr")
    if not homr_bin:
        raise OMREngineUnavailableError(
            "homr CLI binary is not installed on PATH; refusing to return fabricated MusicXML"
        )

    with tempfile.TemporaryDirectory(prefix="omr_homr_") as tmpdir:
        tmp_path = pathlib.Path(tmpdir)
        img_path = tmp_path / "page.png"
        cv2.imwrite(str(img_path), img)
        try:
            proc = subprocess.run(
                [homr_bin, str(img_path)],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                check=False,
                timeout=max(1.0, min(118.0, timeout_s)),
            )
        except subprocess.TimeoutExpired as exc:
            raise OMREngineTimeoutError("homr CLI subprocess timed out") from exc

        xml_candidates = list(tmp_path.glob("*.musicxml")) + list(tmp_path.glob("*.xml"))
        if proc.returncode != 0 or not xml_candidates:
            raise OMREngineUnavailableError(
                f"homr CLI failed with exit code {proc.returncode} (xml_files={len(xml_candidates)})"
            )

        musicxml_str = xml_candidates[0].read_text(encoding="utf-8")
        boxes_file = img_path.with_suffix(".boxes.json")
        if boxes_file.exists():
            raw_boxes: dict[str, Any] = json.loads(boxes_file.read_text(encoding="utf-8"))
            parsed_boxes = [
                BoundingBoxItem.model_validate(b) for b in raw_boxes.get("measure_boxes", [])
            ]
            spacing = float(raw_boxes.get("staff_spacing_px", staff_spacing_px))
            return OMRResponse(
                musicxml=musicxml_str,
                measure_boxes=parsed_boxes or opencv_boxes,
                staff_spacing_px=spacing,
                engine="homr-cli",
            )
        return OMRResponse(
            musicxml=musicxml_str,
            measure_boxes=opencv_boxes,
            staff_spacing_px=staff_spacing_px,
            engine="homr-cli",
        )


@app.get("/health")
def get_health() -> dict[str, str]:
    return {"status": "ok", "license": "AGPL-3.0-only"}


@app.get("/source", response_model=SourceResponse)
def get_source() -> SourceResponse:
    return SourceResponse(
        repo_url=os.environ.get("OMR_HOMR_REPO_URL", "https://github.com/copianist/omr-homr"),
        commit_sha=get_commit_sha(),
        license="AGPL-3.0-only",
    )


def _compute_deadline_timeout(x_request_deadline: str | None) -> float:
    if not x_request_deadline:
        return 115.0
    try:
        deadline_epoch = float(x_request_deadline)
        remaining = deadline_epoch - time.time() - 2.0
        return max(1.0, min(115.0, remaining))
    except ValueError:
        return 115.0


@app.post("/omr", response_model=OMRResponse)
async def post_omr(
    image: UploadFile = File(...),
    x_request_deadline: str | None = Header(default=None),
) -> OMRResponse | JSONResponse:
    raw_bytes = await image.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty image payload")
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        return JSONResponse(
            status_code=413,
            content={
                "error_code": "IMAGE_TOO_LARGE",
                "message": "Image exceeds 4 MB upload limit",
                "retryable": False,
            },
        )
    timeout_s = _compute_deadline_timeout(x_request_deadline)
    try:
        return await run_in_threadpool(run_omr_on_image_bytes, raw_bytes, timeout_s)
    except ValueError:
        return JSONResponse(
            status_code=422,
            content={
                "error_code": "INVALID_IMAGE",
                "message": "Invalid image format or dimensions exceeded",
                "retryable": False,
            },
        )
    except OMREngineTimeoutError:
        return JSONResponse(
            status_code=504,
            headers={"Retry-After": "5"},
            content={
                "error_code": "OMR_UPSTREAM_TIMEOUT",
                "message": "homr OMR subprocess timed out",
                "retryable": True,
            },
        )
    except OMREngineUnavailableError:
        return JSONResponse(
            status_code=503,
            headers={"Retry-After": "5"},
            content={
                "error_code": "OMR_ENGINE_UNAVAILABLE",
                "message": "homr OMR engine is unavailable or failed",
                "retryable": True,
            },
        )
