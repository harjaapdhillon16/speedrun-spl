"""Re-OCR every stored card frame with EasyOCR (Devanagari + English) + a
layout-aware parser, join to NSE market data, and fill the dedicated
`stock_analysis_calls` table — in parallel on a powerful CPU.

Standalone for RunPod. It reuses the heavy market/enrich code from
spl_midcap_speedrun.py (yahoo_search / yahoo_history / enrich_card) and only adds
a much more accurate frame reader. No video download needed — it reads the card
frames already uploaded to Supabase Storage (video_frames/<job>/<name>.png).

Why this exists: the old Tesseract whole-frame parser returned the wrong company
on most cards, polluted targets with duration/quote numbers, and garbled the
Hindi analyst names. The EasyOCR layout parser reads each field by its position
on the card and is dramatically more accurate.

Run (after bootstrap):
    export url=...; export secret_key=...
    python ocr_stock_analysis_fill.py --workers 16

Background:
    nohup python -u ocr_stock_analysis_fill.py --workers 16 \
        </dev/null >ocr_fill.log 2>&1 & disown
    tail -f ocr_fill.log
"""

from __future__ import annotations

import argparse
import json
import os
import re
import ssl
import subprocess
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

# macOS / minimal Linux Python may lack CA certs for the one-time EasyOCR
# model-weight download.
ssl._create_default_https_context = ssl._create_unverified_context


def _ensure_deps() -> None:
    need = []
    for mod, pkg in (("cv2", "opencv-python-headless"), ("numpy", "numpy"),
                     ("httpx", "httpx"), ("easyocr", "easyocr")):
        try:
            __import__(mod)
        except ImportError:
            need.append(pkg)
    if need:
        print(json.dumps({"installing": need}), flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", *need])


_ensure_deps()

import httpx  # noqa: E402

# Reuse the runner's market-join + Supabase REST helpers (importing it also
# installs its own deps and defines enrich_card/yahoo_*).
from spl_midcap_speedrun import enrich_card, log_event, pct  # noqa: E402

# Market enrichment (Yahoo Finance) is OFF by default: RunPod/datacenter IPs are
# routinely blocked by Yahoo, which makes every worker hang on network timeouts.
# OCR-only fill is fast and Supabase-only; enable --enrich where Yahoo is reachable.
ENRICH = False

SUPABASE_URL = (os.getenv("SUPABASE_URL") or os.getenv("url") or "").rstrip("/")
SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_SECRET_KEY") or os.getenv("secret_key")
BUCKET = os.getenv("SUPABASE_MEDIA_BUCKET", "stock-call-media")
DATA_TABLE = os.getenv("SUPABASE_DATA_TABLE", "app_records")
TARGET_TABLE = "stock_analysis_calls"
_STATUS_RE = re.compile(r"/status/(\d+)")


def _h(extra: dict | None = None) -> dict:
    h = {"apikey": SERVICE_KEY, "Authorization": f"Bearer {SERVICE_KEY}", "Content-Type": "application/json"}
    if extra:
        h.update(extra)
    return h


def _storage_list(prefix: str, limit: int = 5000) -> list[str]:
    url = f"{SUPABASE_URL}/storage/v1/object/list/{BUCKET}"
    r = httpx.post(url, headers=_h(), json={"prefix": prefix, "limit": limit}, timeout=60)
    return [e["name"] for e in r.json()] if r.status_code == 200 else []


def _storage_get(path: str) -> bytes:
    # Service key can read objects directly (no signing round-trip).
    url = f"{SUPABASE_URL}/storage/v1/object/{BUCKET}/{path.lstrip('/')}"
    r = httpx.get(url, headers=_h(), timeout=60)
    r.raise_for_status()
    return r.content


# --------------------------------------------------------------------------- #
# EasyOCR layout extractor (validated; see python-backend/scripts/ocr_extract.py)
# --------------------------------------------------------------------------- #
_DEVA = str.maketrans("०१२३४५६७८९", "0123456789")
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_PANEL_X = 0.56
# ONE shared EasyOCR Reader, built once in the main thread and reused by every
# worker thread. Building a torch model concurrently in many threads deadlocks;
# inference (readtext) on a shared eval model is fine across threads.
_READER = None
_READER_LOCK = threading.Lock()


def _get_reader():
    global _READER
    if _READER is None:
        with _READER_LOCK:
            if _READER is None:
                import easyocr

                _READER = easyocr.Reader(["hi", "en"], gpu=False, verbose=False)
    return _READER


def _readtext(img):
    # Shared reader, concurrent inference (verified working). Model is built once
    # in the main thread; building per-thread deadlocks, so never do that.
    return _get_reader().readtext(img)


class _Box:
    __slots__ = ("text", "conf", "cx", "cy", "x0", "y0", "x1", "y1")

    def __init__(self, text, conf, pts, w, h):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        self.text = text.translate(_DEVA).strip()
        self.conf = conf
        self.x0, self.x1 = min(xs) / w, max(xs) / w
        self.y0, self.y1 = min(ys) / h, max(ys) / h
        self.cx = (self.x0 + self.x1) / 2
        self.cy = (self.y0 + self.y1) / 2


def _digits(s: str):
    s = re.sub(r"(?<=\d)[oOlISB]", lambda m: {"o": "0", "O": "0", "l": "1", "I": "1", "S": "5", "B": "8"}[m.group()], s)
    m = _NUM.search(s)
    if not m:
        return None
    try:
        return float(m.group().replace(",", ""))
    except ValueError:
        return None


def _split_nums(s: str) -> list:
    out = []
    for part in re.split(r"[/,]", s):
        v = _digits(part)
        if v is not None:
            out.append(v)
    return out


def extract(img) -> dict:
    h, w = img.shape[:2]
    boxes = [_Box(t, c, pts, w, h) for pts, t, c in _readtext(img)]
    panel = [b for b in boxes if b.x0 < _PANEL_X]

    def deva_ratio(s: str) -> float:
        alpha = [c for c in s if c.isalpha()]
        return (sum("ऀ" <= c <= "ॿ" for c in alpha) / len(alpha)) if alpha else 0.0

    banner = sorted((b for b in panel if 0.12 <= b.cy <= 0.27 and deva_ratio(b.text) >= 0.6), key=lambda b: b.cx)
    analyst = None
    if banner:
        joined = " ".join(b.text for b in banner)
        t = re.split(r"\s*की\s*(?:पसंद|राय)|\s+(?:पसंद|पसद|राय)\b", joined)[0].strip()
        t = re.sub(r"\s*की\s*$", "", t).strip()
        analyst = t or None

    def latin_name(b: _Box):
        t = re.sub(r"[^A-Za-z0-9 .&'\-]", "", b.text).strip(" .-")
        return t if len(re.sub(r"[^A-Za-z]", "", t)) >= 4 else None

    cands = [(b.conf, latin_name(b)) for b in panel if 0.24 <= b.cy <= 0.34 and latin_name(b)]
    stock = max(cands)[1] if cands else None

    alltext = " ".join(b.text for b in panel)
    if re.search(r"खरीद|रीद|\bbuy\b", alltext, re.I):
        action = "Buy"
    elif re.search(r"बेच|ेचें|\bsell\b", alltext, re.I):
        action = "Sell"
    else:
        action = None

    target = duration = stop = None
    for b in panel:
        low = b.text.lower()
        if 0.35 <= b.cy <= 0.62:
            if duration is None and re.search(r"dura|rati|ation|tion", low):
                duration = b
            elif target is None and "arg" in low:
                target = b
            elif stop is None and re.search(r"l[o0u]?[s5]{2}|st[o0]p|si[o0]p|oss", low):
                stop = b

    stop_loss, targets, duration_months = None, [], None
    assigned: dict = {}
    if target is not None or duration is not None:
        if target is not None and duration is not None:
            target_cx, colw = target.cx, (duration.cx - target.cx)
        elif target is not None:
            target_cx, colw = target.cx, 0.165
        else:
            target_cx, colw = duration.cx - 0.165, 0.165
        anchor_y = (target.cy if target is not None else duration.cy)
        cols = {"target": target_cx}
        if duration is not None:
            cols["duration"] = duration.cx
        sl_cx = target_cx - colw
        if sl_cx > 0.03:
            cols["stop"] = sl_cx
        values = [b for b in panel if anchor_y + 0.02 < b.cy < anchor_y + 0.13 and re.search(r"\d", b.text)]
        for col, cx in cols.items():
            near = [v for v in values if abs(v.cx - cx) <= colw * 0.55]
            if near:
                assigned[col] = min(near, key=lambda v: abs(v.cx - cx))
        if "stop" in assigned:
            stop_loss = _digits(assigned["stop"].text) or None
        if "target" in assigned:
            targets = [t for t in _split_nums(assigned["target"].text) if t > 0]
        if "duration" in assigned:
            m = re.search(r"(\d{1,2})\s*[-–—]\s*(\d{1,2})", assigned["duration"].text)
            if m:
                duration_months = [int(m.group(1)), int(m.group(2))]

    current = None
    used = {id(b) for b in assigned.values()}
    cand = []
    for b in panel:
        if id(b) in used or not (0.33 <= b.cy <= 0.66):
            continue
        if "%" in b.text or "." not in b.text:
            continue
        for v in _split_nums(b.text):
            if v >= 5:
                cand.append(v)
    if cand:
        current = max(cand)

    date = None
    for b in boxes:
        if b.cy > 0.78 and b.cx > 0.55:
            m = re.search(r"(\d{1,2})\s*[/|!.\-]\s*(\d{1,2})\s*[/|!.\-]\s*(\d{4})", b.text)
            if m:
                d, mo, y = m.groups()
                date = f"{y}-{int(mo):02d}-{int(d):02d}"
                break

    return {
        "stock": stock, "analyst": analyst, "action": action, "entry": current,
        "stop_loss": stop_loss, "targets": targets, "duration_months": duration_months, "date": date,
    }


# --------------------------------------------------------------------------- #
# Worker: list a job's frames, download, OCR
# --------------------------------------------------------------------------- #
_META: dict = {}


def _process_job(job_id: str) -> list[dict]:
    """OCR every frame of a job AND market-enrich it (both parallel across the
    pool). Returns finished stock_analysis_calls rows (or {'error': ...})."""
    import cv2
    import numpy as np

    out = []
    job = _META.get(job_id, {})
    cards = job.get("cards", {})
    for name in _storage_list(f"video_frames/{job_id}/", 300):
        if not name.endswith(".png"):
            continue
        try:
            content = _storage_get(f"video_frames/{job_id}/{name}")
            img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            ocr = extract(img)
        except Exception as exc:  # noqa: BLE001
            out.append({"error": str(exc)})
            continue
        cmeta = cards.get(name, {})
        rec = {
            "job_id": job_id, "frame": name,
            "image_url": cmeta.get("image_url") or f"/video/frames/{job_id}/{name}",
            "call_id": cmeta.get("call_id") or f"vcall_{job_id}_{name.rsplit('.', 1)[0]}",
            "video_timestamp": cmeta.get("timestamp") or 0.0,
            "source_url": job.get("url"),
            "snowflake": job.get("snowflake"),
            "ocr": ocr,
        }
        try:
            out.append({"row": _to_row(rec)})
        except Exception as exc:  # noqa: BLE001
            out.append({"error": f"enrich:{exc}"})
    return out


# --------------------------------------------------------------------------- #
# Main: build meta, OCR in parallel, enrich + upsert
# --------------------------------------------------------------------------- #
def _snowflake_iso(url):
    if not url:
        return None
    m = _STATUS_RE.search(url)
    if not m:
        return None
    from datetime import datetime, timezone

    ts_ms = (int(m.group(1)) >> 22) + 1288834974657
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).date().isoformat()


def _load_video_jobs() -> list[dict]:
    rows = []
    for start in range(0, 1_000_000, 1000):
        h = _h({"Range-Unit": "items", "Range": f"{start}-{start + 999}"})
        r = httpx.get(
            f"{SUPABASE_URL}/rest/v1/{DATA_TABLE}",
            headers=h,
            params={"select": "payload", "collection": "eq.video_jobs"},
            timeout=120,
        )
        r.raise_for_status()
        batch = [x.get("payload") or {} for x in r.json()]
        rows.extend(batch)
        if len(batch) < 1000:
            break
    return rows


def _build_meta() -> dict:
    meta = {}
    for job in _load_video_jobs():
        if job.get("status") != "done":
            continue
        cards = {}
        for c in job.get("cards") or []:
            nm = (c.get("image_url") or "").rsplit("/", 1)[-1]
            if nm:
                cards[nm] = {"call_id": c.get("call_id"), "timestamp": c.get("timestamp"), "image_url": c.get("image_url")}
        meta[job["id"]] = {"url": job.get("url"), "snowflake": _snowflake_iso(job.get("url")), "cards": cards}
    return meta


def _to_row(rec: dict):
    from datetime import date as _date, timedelta as _td

    ocr = rec["ocr"]
    a = (ocr.get("action") or "").lower()
    is_buy = a != "sell"
    reco = "Sell" if a == "sell" else "Buy"
    dur = ocr.get("duration_months")
    horizon = dur[1] if dur else None
    entry_date = rec.get("snowflake") or ocr.get("date")
    targets = ocr.get("targets") or []
    target_price = float(targets[0]) if targets else None
    entry = ocr.get("entry")
    expected = pct(entry, target_price, is_buy) if (entry and target_price) else None

    # Base row from OCR only — no external network. Symbol is the raw OCR name
    # (NSE-symbol resolution + mark-to-market happen in --enrich mode).
    target_date = None
    try:
        if entry_date and horizon:
            target_date = (_date.fromisoformat(entry_date) + _td(days=horizon * 30)).isoformat()
    except ValueError:
        pass

    stock = ocr.get("stock")
    row = {
        "call_id": rec["call_id"],
        "video_job_id": rec["job_id"],
        "source_url": rec.get("source_url"),
        "image_url": rec["image_url"],
        "video_timestamp": rec.get("video_timestamp"),
        "raw_ocr_text": json.dumps(ocr, ensure_ascii=False),
        "analyst": ocr.get("analyst"),
        "stock": stock,
        "stock_full_name": None,
        "analyst_company": None,
        "entry_date": entry_date,
        "target_date": target_date,
        "close_date": None,
        "entry_price": round(entry, 2) if entry else None,
        "stop_loss": ocr.get("stop_loss"),
        "target_price": target_price,
        "expected_return_pct": expected,
        "reco": reco,
        "open_close": None,
        "current_price": None,
        "actual_return_pct": None,
        "annualized_pct": None,
        "success": None,
        "source": "video",
        "platform": "Zee Business",
        "program": "SPL Midcap",
        "theme": ocr.get("theme"),
    }

    if ENRICH:
        # Resolve NSE symbol + mark-to-market via Yahoo (only where reachable).
        card_in = {
            "call_id": rec["call_id"], "stock": stock,
            "action": "SELL" if not is_buy else "BUY",
            "entry": entry, "stop_loss": ocr.get("stop_loss"), "targets": targets,
            "entry_date": entry_date, "horizon_months": horizon, "current_price": entry,
        }
        try:
            e = enrich_card(card_in, rec["image_url"], rec.get("source_url"), rec.get("video_timestamp") or 0.0)
            status = e.get("status")
            open_close = "Open" if status == "open" else ("Close" if (status or "").startswith("closed on") else None)
            close_date = None
            if open_close == "Close":
                m = re.search(r"(\d{2})/(\d{2})/(\d{4})", status)
                if m:
                    close_date = f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
            row.update({
                "stock": e.get("stock") or stock,
                "stock_full_name": e.get("stock_full_name"),
                "entry_date": e.get("entry_date") or entry_date,
                "target_date": e.get("target_date") or target_date,
                "close_date": close_date,
                "entry_price": e.get("entry_price") if e.get("entry_price") is not None else row["entry_price"],
                "target_price": e.get("target_price") if e.get("target_price") is not None else target_price,
                "expected_return_pct": e.get("expected_return_pct") if e.get("expected_return_pct") is not None else expected,
                "open_close": open_close,
                "current_price": e.get("current_price"),
                "actual_return_pct": e.get("actual_return_pct"),
                "annualized_pct": e.get("annualized_pct"),
                "success": e.get("success"),
            })
        except Exception:  # noqa: BLE001 - Yahoo unreachable/blocked; keep OCR-only row
            pass

    nat = f"{row.get('source_url') or ''}|{row.get('stock') or ''}|{row.get('entry_date') or '1900-01-01'}"
    row["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, nat))
    return row


def _upsert(rows: list[dict]) -> None:
    if not rows:
        return
    r = httpx.post(
        f"{SUPABASE_URL}/rest/v1/{TARGET_TABLE}",
        headers=_h({"Prefer": "resolution=merge-duplicates,return=minimal"}),
        json=rows,
        timeout=180,
    )
    if r.status_code >= 300:
        log_event({"upsert_error": r.status_code, "body": r.text[:200]})


def main() -> None:
    if not SUPABASE_URL or not SERVICE_KEY:
        raise SystemExit("Set SUPABASE_URL/url and SUPABASE_SERVICE_ROLE_KEY/secret_key first.")
    global _META, ENRICH
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=min(12, (os.cpu_count() or 8)))
    p.add_argument("--limit", type=int, default=None, help="process only the first N job dirs (testing)")
    p.add_argument("--enrich", action="store_true", help="also resolve NSE symbol + mark-to-market via Yahoo (slow/blocked on many datacenter IPs)")
    args = p.parse_args()
    ENRICH = args.enrich

    # Each thread runs EasyOCR with 1 torch thread so the pool — not torch's
    # intra-op threads — provides the parallelism. cv2 threads off too.
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:  # noqa: BLE001
        pass
    try:
        import cv2

        cv2.setNumThreads(0)
    except Exception:  # noqa: BLE001
        pass

    _META = _build_meta()
    jobs = _storage_list("video_frames/")
    if args.limit:
        jobs = jobs[: args.limit]
    log_event({"jobs_with_frames": len(jobs), "workers": args.workers, "known_jobs": len(_META), "enrich": ENRICH})

    # Warm the EasyOCR model cache ONCE in the main thread before the pool, so
    # worker threads read the weights from cache instead of racing to download.
    log_event({"warming_easyocr_model": True})
    _get_reader()
    log_event({"easyocr_model_ready": True})

    total, done, errors, dupes = 0, 0, 0, 0
    seen: set = set()
    pending: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_process_job, j): j for j in jobs}
        log_event({"pool_submitted": len(futs)})
        for fut in as_completed(futs):
            done += 1
            if done <= 3 or done % 10 == 0:
                log_event({"job_done": done, "rows": total, "pending": len(pending)})
            try:
                recs = fut.result()
            except Exception as exc:  # noqa: BLE001
                log_event({"job_failed": futs[fut], "error": str(exc)})
                continue
            for item in recs:
                if "error" in item:
                    errors += 1
                    continue
                row = item["row"]
                key = (row.get("source_url") or "", row.get("stock") or "", row.get("entry_date") or "1900-01-01")
                if key in seen:
                    dupes += 1
                    continue
                seen.add(key)
                pending.append(row)
            if len(pending) >= 25:
                _upsert(pending)
                total += len(pending)
                pending = []
            if done % 10 == 0:
                log_event({"progress": f"{done}/{len(jobs)}", "rows": total, "dupes": dupes, "errors": errors})
    if pending:
        _upsert(pending)
        total += len(pending)
    log_event({"DONE": True, "rows_upserted": total, "dup_frames_collapsed": dupes, "frame_errors": errors})


if __name__ == "__main__":
    main()
