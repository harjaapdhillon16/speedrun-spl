"""GPU vision-language re-fill of `stock_analysis_calls` — RunPod / Ubuntu.

Reads the card frames already in Supabase Storage
(video_frames/<job>/<name>.png) and, instead of EasyOCR + brittle layout regex,
feeds each card to a vision-language model (Qwen2.5-VL) that returns the call
fields directly as JSON. The model *understands* the card — Hindi analyst names,
"खरीदें/बेचें", multi-target rows, the price column — so the output is far more
accurate than the old whole-frame OCR that returned the wrong company on most
cards.

Why this is fast and just works on RunPod:
  * GPU inference (bf16), ONE model load, then it rips — no per-thread torch
    model rebuild (that is what deadlocked the EasyOCR run).
  * BATCH = 3 images per forward pass (override with --batch). Downloads for the
    next batch overlap the GPU on a tiny I/O thread pool — no torch threading.
  * Images are downscaled before the model, so token count (and latency) stays
    bounded regardless of source resolution.

Run on a RunPod GPU pod (Ubuntu, CUDA base image):
    export url='https://YOUR_PROJECT.supabase.co'
    export secret_key='YOUR_SUPABASE_SERVICE_ROLE_KEY'
    bash run_vlm_fill.sh                 # background, logs to ./vlm_fill.log
    tail -f vlm_fill.log

Or directly:
    python ocr_vlm_fill.py --batch 3
    python ocr_vlm_fill.py --batch 3 --model Qwen/Qwen2.5-VL-3B-Instruct  # smaller GPU
    python ocr_vlm_fill.py --limit 5     # smoke test on 5 job dirs
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date as _date, timedelta as _td

# --------------------------------------------------------------------------- #
# Dependencies. On a RunPod CUDA base image torch is already present; we only
# add the model stack. flash-attn is optional (we fall back to sdpa).
# --------------------------------------------------------------------------- #
def _ensure_deps() -> None:
    import subprocess
    import sys

    need = []
    for mod, pkg in (("httpx", "httpx"), ("PIL", "pillow"),
                     ("transformers", "transformers>=4.49.0"),
                     ("accelerate", "accelerate")):
        try:
            __import__(mod)
        except ImportError:
            need.append(pkg)
    if need:
        print(json.dumps({"installing": need}), flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", *need])


_ensure_deps()

import httpx  # noqa: E402
from PIL import Image  # noqa: E402

# --------------------------------------------------------------------------- #
# Config / Supabase plumbing (self-contained — no heavy runner import on the
# common OCR-only path).
# --------------------------------------------------------------------------- #
SUPABASE_URL = (os.getenv("SUPABASE_URL") or os.getenv("url") or "").rstrip("/")
SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_SECRET_KEY") or os.getenv("secret_key")
BUCKET = os.getenv("SUPABASE_MEDIA_BUCKET", "stock-call-media")
DATA_TABLE = os.getenv("SUPABASE_DATA_TABLE", "app_records")
TARGET_TABLE = "stock_analysis_calls"
MODEL = os.getenv("VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
_STATUS_RE = re.compile(r"/status/(\d+)")
ENRICH = False

_LOG_LOCK = threading.Lock()


def log_event(event) -> None:
    with _LOG_LOCK:
        if isinstance(event, str):
            print(event, flush=True)
        else:
            print(json.dumps(event, ensure_ascii=False, default=str), flush=True)


def pct(frm: float, to: float, is_buy: bool = True) -> float:
    move = (to - frm) / frm * 100.0
    return round(move if is_buy else -move, 2)


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
    url = f"{SUPABASE_URL}/storage/v1/object/{BUCKET}/{path.lstrip('/')}"
    r = httpx.get(url, headers=_h(), timeout=60)
    r.raise_for_status()
    return r.content


# --------------------------------------------------------------------------- #
# Vision-language extractor
# --------------------------------------------------------------------------- #
_PROMPT = (
    "This is a single stock-recommendation card from an Indian business-news TV "
    "channel (Hindi + English). Read it and return ONLY a JSON object with these "
    "keys:\n"
    '  "stock": company/stock name exactly as printed, in English (string or null)\n'
    '  "analyst": the expert/analyst whose pick this is — usually a Hindi name, '
    'e.g. from "<name> की पसंद/राय" take just <name> (string or null)\n'
    '  "action": "Buy" or "Sell" (खरीदें=Buy, बेचें=Sell) (string or null)\n'
    '  "entry": current price / CMP as a number (number or null)\n'
    '  "stop_loss": stop-loss as a number (number or null)\n'
    '  "targets": target price(s) as an array of numbers (e.g. [123] or [123,140])\n'
    '  "duration_months": holding period in months as [min,max]; if a single '
    "value use [n,n]; if absent use null\n"
    '  "date": date shown on the card as "YYYY-MM-DD" (string or null)\n'
    '  "theme": sector/theme if shown (string or null)\n'
    "Rules: numbers must be plain (no commas, no ₹/Rs). Use null or [] when a "
    "field is not present. Do NOT guess values that are not on the card. Output "
    "the JSON object and nothing else."
)

_MAX_SIDE = int(os.getenv("VLM_MAX_SIDE", "1024"))
_vlm: dict = {}


def _load_vlm() -> None:
    import torch
    from transformers import AutoProcessor

    log_event({"loading_model": MODEL})
    try:
        from transformers import AutoModelForImageTextToText as _ModelCls
    except ImportError:  # older transformers
        from transformers import Qwen2_5_VLForConditionalGeneration as _ModelCls

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    attn = "sdpa"
    try:
        import flash_attn  # noqa: F401

        attn = "flash_attention_2"
    except Exception:  # noqa: BLE001
        pass

    kwargs = {"torch_dtype": dtype, "attn_implementation": attn}
    kwargs["device_map"] = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = _ModelCls.from_pretrained(MODEL, **kwargs)
    except Exception:  # noqa: BLE001 - flash-attn not installed/usable → sdpa
        kwargs["attn_implementation"] = "sdpa"
        model = _ModelCls.from_pretrained(MODEL, **kwargs)
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL)
    _vlm["model"] = model
    _vlm["processor"] = processor
    _vlm["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    log_event({"model_ready": True, "device": _vlm["device"], "attn": kwargs["attn_implementation"], "dtype": str(dtype)})


def _prep_image(content: bytes):
    img = Image.open(io.BytesIO(content)).convert("RGB")
    w, h = img.size
    scale = _MAX_SIDE / max(w, h)
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    return img


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


def _num(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    m = re.search(r"-?\d[\d,]*(?:\.\d+)?", str(v))
    return float(m.group().replace(",", "")) if m else None


def _normalize(d: dict) -> dict:
    targets = d.get("targets")
    if not isinstance(targets, list):
        targets = [targets] if targets is not None else []
    targets = [t for t in (_num(x) for x in targets) if t is not None and t > 0]

    dur = d.get("duration_months")
    if isinstance(dur, list) and dur:
        dur = [int(_num(x)) for x in dur if _num(x) is not None] or None
        if dur and len(dur) == 1:
            dur = [dur[0], dur[0]]
    elif _num(dur) is not None:
        dur = [int(_num(dur)), int(_num(dur))]
    else:
        dur = None

    action = (d.get("action") or "").strip().lower()
    action = "Sell" if action.startswith("sell") else ("Buy" if action.startswith("buy") else None)

    def s(key):
        v = d.get(key)
        v = str(v).strip() if v is not None else ""
        return v or None

    return {
        "stock": s("stock"),
        "analyst": s("analyst"),
        "action": action,
        "entry": _num(d.get("entry")),
        "stop_loss": _num(d.get("stop_loss")),
        "targets": targets,
        "duration_months": dur,
        "date": s("date"),
        "theme": s("theme"),
    }


def extract_batch(images: list) -> list[dict]:
    """Run the VLM on a list of PIL images; return one normalized ocr dict each."""
    import torch

    model, processor = _vlm["model"], _vlm["processor"]
    messages = [
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": _PROMPT}]}]
        for _ in images
    ]
    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
    inputs = processor(text=texts, images=list(images), padding=True, return_tensors="pt").to(_vlm["device"])
    with torch.inference_mode():
        gen = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen)]
    decoded = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return [_normalize(_parse_json(t)) for t in decoded]


# --------------------------------------------------------------------------- #
# Meta + row building (engine-agnostic)
# --------------------------------------------------------------------------- #
_META: dict = {}


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
        from spl_midcap_speedrun import enrich_card  # heavy import, only when needed

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
        except Exception:  # noqa: BLE001
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


# --------------------------------------------------------------------------- #
# Main: stream every frame through the GPU in batches of N
# --------------------------------------------------------------------------- #
def _iter_frames(jobs: list[str]):
    for job_id in jobs:
        for name in _storage_list(f"video_frames/{job_id}/", 300):
            if name.endswith(".png"):
                yield job_id, name


def _chunks(it, n):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def main() -> None:
    if not SUPABASE_URL or not SERVICE_KEY:
        raise SystemExit("Set SUPABASE_URL/url and SUPABASE_SERVICE_ROLE_KEY/secret_key first.")
    global _META, ENRICH
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=3, help="images per GPU forward pass")
    p.add_argument("--limit", type=int, default=None, help="process only the first N job dirs (testing)")
    p.add_argument("--model", default=None, help="override VLM_MODEL (e.g. Qwen/Qwen2.5-VL-3B-Instruct)")
    p.add_argument("--enrich", action="store_true", help="also resolve NSE symbol + mark-to-market via Yahoo")
    args = p.parse_args()
    ENRICH = args.enrich
    if args.model:
        globals()["MODEL"] = args.model

    _META = _build_meta()
    jobs = _storage_list("video_frames/")
    if args.limit:
        jobs = jobs[: args.limit]
    log_event({"jobs_with_frames": len(jobs), "known_jobs": len(_META), "batch": args.batch, "enrich": ENRICH, "model": MODEL})

    _load_vlm()

    total, frames_done, errors, dupes = 0, 0, 0, 0
    seen: set = set()
    pending: list[dict] = []
    dl_pool = ThreadPoolExecutor(max_workers=max(args.batch, 4))  # I/O only — no torch

    for chunk in _chunks(_iter_frames(jobs), args.batch):
        # Download + decode this batch's images in parallel (pure I/O).
        def _fetch(ref):
            job_id, name = ref
            try:
                return ref, _prep_image(_storage_get(f"video_frames/{job_id}/{name}"))
            except Exception as exc:  # noqa: BLE001
                return ref, exc

        fetched = list(dl_pool.map(_fetch, chunk))
        refs, imgs = [], []
        for ref, val in fetched:
            if isinstance(val, Exception):
                errors += 1
                continue
            refs.append(ref)
            imgs.append(val)
        if not imgs:
            continue

        try:
            ocrs = extract_batch(imgs)
        except Exception as exc:  # noqa: BLE001
            log_event({"batch_failed": str(exc), "n": len(imgs)})
            errors += len(imgs)
            continue

        for (job_id, name), ocr in zip(refs, ocrs):
            frames_done += 1
            job = _META.get(job_id, {})
            cmeta = job.get("cards", {}).get(name, {})
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
                row = _to_row(rec)
            except Exception as exc:  # noqa: BLE001
                errors += 1
                log_event({"row_error": str(exc), "frame": name})
                continue
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
        if frames_done % 30 == 0:
            log_event({"frames": frames_done, "rows": total, "dupes": dupes, "errors": errors})

    if pending:
        _upsert(pending)
        total += len(pending)
    dl_pool.shutdown(wait=False)
    log_event({"DONE": True, "rows_upserted": total, "frames": frames_done, "dupes": dupes, "errors": errors})


if __name__ == "__main__":
    main()
