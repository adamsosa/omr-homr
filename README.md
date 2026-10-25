# omr-homr (Isolated AGPL-3.0 OMR Service)

**License:** `AGPL-3.0-only`

`omr-homr` is a standalone, stateless HTTP microservice that wraps [`homr`](https://github.com/liebharc/homr) via CLI subprocess and applies a minimal patch (`patches/export_measure_boxes.patch`) to export normalized `[x, y, w, h] in [0, 1]` measure and staff bounding boxes alongside MusicXML 4.0 output.

## Architectural Boundary & AGPL-3.0 Section 13 Compliance

1. **Process & Repository Isolation:** This repository contains all AGPL-3.0 code and patches. The proprietary CoPianist backend (`backend/api`) never imports or links `homr`; it communicates with this service exclusively over HTTP (`POST /omr`).
2. **Source Disclosure (`GET /source`):** In compliance with AGPL-3.0 Section 13, `GET /source` returns the public repository URL, exact Git commit SHA, and SPDX license identifier (`AGPL-3.0-only`).
3. **Local Development Fallback:** When the `homr` CLI binary/weights are not installed in a local development or CI environment, `omr_homr_service` automatically falls back to a deterministic OpenCV morphological staff-line + barline detector and MusicXML 4.0 generator so contract and integration tests run without GPU/model weights.

## Endpoints

- `POST /omr` (`multipart/form-data` with `image` file):
  - Returns `{ "musicxml": "<score-partwise...>", "measure_boxes": [{"x": ..., "y": ..., "w": ..., "h": ..., ...}], "staff_spacing_px": 14.0, "engine": "..." }`
- `GET /source`:
  - Returns `{ "repo_url": "https://github.com/copianist/omr-homr", "commit_sha": "<sha>", "license": "AGPL-3.0-only" }`
- `GET /health`:
  - Returns `{ "status": "ok", "license": "AGPL-3.0-only" }`
