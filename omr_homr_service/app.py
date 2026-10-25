# SPDX-License-Identifier: AGPL-3.0-only
"""FastAPI HTTP wrapper around homr CLI subprocess with OpenCV staff/barline fallback."""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import tempfile
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

app = FastAPI(
    title="omr-homr Service",
    version="1.0.0",
    description="Isolated AGPL-3.0 Optical Music Recognition service wrapping homr CLI subprocess.",
)


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
    engine: str = "opencv-fallback"


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


def build_fallback_musicxml(measure_boxes: list[BoundingBoxItem]) -> str:
    """Synthesize a valid MusicXML 4.0 score matching the detected measure and staff structure."""
    num_measures = max(1, len(measure_boxes))
    first_staves = (
        len(measure_boxes[0].staff_boxes) if measure_boxes and measure_boxes[0].staff_boxes else 3
    )
    has_vocal = first_staves >= 3

    vocal_measures_xml: list[str] = []
    piano_measures_xml: list[str] = []
    pitches_vocal = [("C", 5), ("D", 5), ("E", 5), ("G", 4)]
    pitches_rh = [("E", 4), ("F", 4), ("G", 4), ("C", 4)]
    pitches_lh = [("C", 3), ("G", 2), ("A", 2), ("F", 2)]

    for idx in range(num_measures):
        m_num = idx + 1
        v_step, v_oct = pitches_vocal[idx % len(pitches_vocal)]
        rh_step, rh_oct = pitches_rh[idx % len(pitches_rh)]
        lh_step, lh_oct = pitches_lh[idx % len(pitches_lh)]

        if has_vocal:
            v_attr = (
                "      <attributes>\n"
                "        <divisions>4</divisions>\n"
                "        <key><fifths>0</fifths><mode>major</mode></key>\n"
                "        <time><beats>4</beats><beat-type>4</beat-type></time>\n"
                "        <clef><sign>G</sign><line>2</line></clef>\n"
                "      </attributes>\n"
                if idx == 0
                else ""
            )
            vocal_measures_xml.append(
                f'    <measure number="{m_num}">\n'
                f"{v_attr}"
                f"      <note>\n"
                f"        <pitch><step>{v_step}</step><octave>{v_oct}</octave></pitch>\n"
                f"        <duration>16</duration>\n"
                f"        <voice>1</voice>\n"
                f"        <type>whole</type>\n"
                f"      </note>\n"
                f"    </measure>"
            )

        p_attr = (
            "      <attributes>\n"
            "        <divisions>4</divisions>\n"
            "        <key><fifths>0</fifths><mode>major</mode></key>\n"
            "        <time><beats>4</beats><beat-type>4</beat-type></time>\n"
            "        <staves>2</staves>\n"
            '        <clef number="1"><sign>G</sign><line>2</line></clef>\n'
            '        <clef number="2"><sign>F</sign><line>4</line></clef>\n'
            "      </attributes>\n"
            if idx == 0
            else ""
        )
        piano_measures_xml.append(
            f'    <measure number="{m_num}">\n'
            f"{p_attr}"
            f"      <note>\n"
            f"        <pitch><step>{rh_step}</step><octave>{rh_oct}</octave></pitch>\n"
            f"        <duration>16</duration>\n"
            f"        <voice>1</voice>\n"
            f"        <type>whole</type>\n"
            f"        <staff>1</staff>\n"
            f"      </note>\n"
            f"      <backup><duration>16</duration></backup>\n"
            f"      <note>\n"
            f"        <pitch><step>{lh_step}</step><octave>{lh_oct}</octave></pitch>\n"
            f"        <duration>16</duration>\n"
            f"        <voice>5</voice>\n"
            f"        <type>whole</type>\n"
            f"        <staff>2</staff>\n"
            f"      </note>\n"
            f"    </measure>"
        )

    if has_vocal:
        parts_list_xml = (
            '    <score-part id="P1"><part-name>Voice</part-name></score-part>\n'
            '    <score-part id="P2"><part-name>Piano</part-name></score-part>'
        )
        parts_body_xml = (
            '  <part id="P1">\n'
            + "\n".join(vocal_measures_xml)
            + '\n  </part>\n  <part id="P2">\n'
            + "\n".join(piano_measures_xml)
            + "\n  </part>"
        )
    else:
        parts_list_xml = '    <score-part id="P1"><part-name>Piano</part-name></score-part>'
        parts_body_xml = '  <part id="P1">\n' + "\n".join(piano_measures_xml) + "\n  </part>"

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
        '"http://www.musicxml.org/dtds/partwise.dtd">\n'
        '<score-partwise version="4.0">\n'
        "  <part-list>\n"
        f"{parts_list_xml}\n"
        "  </part-list>\n"
        f"{parts_body_xml}\n"
        "</score-partwise>\n"
    )


def run_omr_on_image_bytes(image_bytes: bytes) -> OMRResponse:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unable to decode image bytes as a valid raster image")

    staff_spacing_px, opencv_boxes = detect_staff_and_measure_boxes_opencv(img)
    homr_bin = shutil.which("homr")
    force_fallback = os.environ.get("OMR_FORCE_FALLBACK", "0") == "1"

    if homr_bin and not force_fallback:
        with tempfile.TemporaryDirectory(prefix="omr_homr_") as tmpdir:
            tmp_path = pathlib.Path(tmpdir)
            img_path = tmp_path / "page.png"
            cv2.imwrite(str(img_path), img)
            proc = subprocess.run(
                [homr_bin, str(img_path)],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                check=False,
                timeout=120.0,
            )
            xml_candidates = list(tmp_path.glob("*.musicxml")) + list(tmp_path.glob("*.xml"))
            if proc.returncode == 0 and xml_candidates:
                musicxml_str = xml_candidates[0].read_text(encoding="utf-8")
                boxes_file = img_path.with_suffix(".boxes.json")
                if boxes_file.exists():
                    raw_boxes: dict[str, Any] = json.loads(boxes_file.read_text(encoding="utf-8"))
                    parsed_boxes = [
                        BoundingBoxItem.model_validate(b)
                        for b in raw_boxes.get("measure_boxes", [])
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

    return OMRResponse(
        musicxml=build_fallback_musicxml(opencv_boxes),
        measure_boxes=opencv_boxes,
        staff_spacing_px=staff_spacing_px,
        engine="opencv-fallback",
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


@app.post("/omr", response_model=OMRResponse)
async def post_omr(image: UploadFile = File(...)) -> OMRResponse:
    raw_bytes = await image.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty image payload")
    try:
        return run_omr_on_image_bytes(raw_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
