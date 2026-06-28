"""Standalone EC2 speed-run for Zee Business #SPLMidcapStocks.

This file intentionally DOES NOT import this repo's ``app.*`` modules. Paste
only this Python file onto a high-CPU EC2 box, set Supabase env vars, install
system tools, and run it.

EC2 system setup, Ubuntu example:

    sudo apt-get update
    sudo apt-get install -y ffmpeg tesseract-ocr tesseract-ocr-hin python3-pip

Run:

    export url='https://YOUR_PROJECT.supabase.co'
    export secret_key='YOUR_SUPABASE_SERVICE_ROLE_KEY'
    python3 spl_midcap_speedrun.py --start 2023-04-01 --concurrency 16

What this script does:
  - searches Nitter for ZeeBusiness #splmidcapstocks videos month-by-month
  - downloads videos with yt-dlp to local pod disk only
  - samples frames with OpenCV
  - OCRs stock call cards with Tesseract
  - keeps the representative frame that contains the Hindi analyst line
  - stores analyst name in Hindi on the individual stock call row
  - uploads call-card PNGs to Supabase Storage; it does not upload source videos
  - writes video_jobs, stock_calls, and enriched_calls into app_records
  - clears enriched_calls before rebuilding unless --no-clear-enriched is used

Notes:
  - Nitter instances are flaky. Pass --nitter "https://instance1,https://instance2"
    if your preferred mirrors work better.
  - Optional proxy rotation: pass --proxy-list "http://user:pass@ip1:port,http://..."
    or set PROXY_LIST. Proxies are used for Nitter and yt-dlp/X downloads,
    not Supabase writes.
  - This is a speed-run script, not a web server. It prints progress to stdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote, urljoin

os.environ.setdefault("PYTHONWARNINGS", "ignore::DeprecationWarning")
warnings.filterwarnings("ignore", category=DeprecationWarning)
LOG_LOCK = threading.Lock()


def log_event(event) -> None:
    with LOG_LOCK:
        if isinstance(event, str):
            print(event, flush=True)
        else:
            print(json.dumps(event, ensure_ascii=False, default=str), flush=True)

PY_DEPS = {
    "httpx": "httpx",
    "bs4": "beautifulsoup4",
    "yt_dlp": "yt-dlp",
    "cv2": "opencv-python-headless",
    "numpy": "numpy",
    "pytesseract": "pytesseract",
}


def ensure_python_deps() -> None:
    missing = []
    for module, package in PY_DEPS.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    if missing:
        log_event({"installing_python_packages": missing})
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


ensure_python_deps()

import cv2  # noqa: E402
import httpx  # noqa: E402
import numpy as np  # noqa: E402
import pytesseract  # noqa: E402
from bs4 import BeautifulSoup  # noqa: E402
from yt_dlp import YoutubeDL  # noqa: E402


DEFAULT_NITTER = [
    "https://nitter.tiekoetter.com",
    "https://xcancel.com",
    "https://nitter.space",
]
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/125 Safari/537.36"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_day(value: str) -> date:
    return date.fromisoformat(value)


def month_windows(start: date, end: date) -> list[tuple[date, date]]:
    out = []
    cur = date(start.year, start.month, 1)
    while cur <= end:
        nxt = date(cur.year + (cur.month // 12), (cur.month % 12) + 1, 1)
        out.append((max(start, cur), min(end, nxt - timedelta(days=1))))
        cur = nxt
    return out


def stable_id(*parts: str, size: int = 12) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:size]


def clean_num(text: str) -> float | None:
    text = text.replace(",", "").strip()
    m = re.search(r"\d+(?:\.\d+)?", text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def pct(frm: float, to: float, is_buy: bool = True) -> float:
    move = (to - frm) / frm * 100.0
    return round(move if is_buy else -move, 2)


def split_csv(value: str | None) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def rotated_proxy(cfg: "Config", key: str) -> str | None:
    if not cfg.proxies:
        return None
    idx = int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16) % len(cfg.proxies)
    return cfg.proxies[idx]


@dataclass
class Config:
    supabase_url: str
    service_key: str
    bucket: str
    table: str
    nitter_instances: list[str]
    proxies: list[str]
    start: date
    end: date
    limit: int
    concurrency: int
    frame_fps: float
    max_frames: int
    min_card_frames: int
    soft_scan: bool
    soft_scan_fps: float
    soft_max_frames: int
    deep_window: float
    deep_step: float
    clear_enriched: bool
    cleanup_videos: bool
    skip_processed: bool
    cleanup_duplicates: bool
    existing_video_jobs: bool
    urls_file: str | None
    skip_discovery: bool
    workdir: Path


class Supabase:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.base = cfg.supabase_url.rstrip("/")
        self.key = cfg.service_key
        self.client = httpx.Client(timeout=120.0, headers={"User-Agent": USER_AGENT})

    def headers(self, prefer: str | None = None, content_type: str = "application/json") -> dict[str, str]:
        h = {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": content_type,
        }
        if prefer:
            h["Prefer"] = prefer
        return h

    def table_url(self) -> str:
        return f"{self.base}/rest/v1/{self.cfg.table}"

    def list_record_rows(self, collection: str) -> list[dict]:
        rows: list[dict] = []
        page_size = 1000
        for start in range(0, 1000000, page_size):
            h = self.headers()
            h["Range-Unit"] = "items"
            h["Range"] = f"{start}-{start + page_size - 1}"
            r = self.client.get(
                self.table_url(),
                headers=h,
                params={
                    "select": "id,payload,updated_at",
                    "collection": f"eq.{collection}",
                    "order": "updated_at.desc",
                },
            )
            r.raise_for_status()
            batch = r.json()
            rows.extend(batch)
            if len(batch) < page_size:
                break
        return rows

    def list_records(self, collection: str) -> list[dict]:
        return [item.get("payload") or {} for item in self.list_record_rows(collection)]

    def delete_collection(self, collection: str) -> None:
        r = self.client.delete(
            self.table_url(),
            headers=self.headers(),
            params={"collection": f"eq.{collection}"},
        )
        r.raise_for_status()

    def delete_records(self, collection: str, ids: list[str]) -> None:
        ids = [str(x) for x in ids if x]
        if not ids:
            return
        for start in range(0, len(ids), 100):
            chunk = ids[start : start + 100]
            encoded = ",".join(chunk)
            r = self.client.delete(
                self.table_url(),
                headers=self.headers(),
                params={"collection": f"eq.{collection}", "id": f"in.({encoded})"},
            )
            r.raise_for_status()

    def upsert_records(self, collection: str, rows: list[dict], id_key: str = "id") -> None:
        if not rows:
            return
        payload = []
        for idx, row in enumerate(rows):
            rid = str(row.get(id_key) or row.get("call_id") or row.get("tweet_id") or f"row_{idx:08d}")
            payload.append({"collection": collection, "id": rid, "payload": row})
        for start in range(0, len(payload), 400):
            r = self.client.post(
                self.table_url(),
                headers=self.headers("resolution=merge-duplicates"),
                params={"on_conflict": "collection,id"},
                json=payload[start : start + 400],
            )
            r.raise_for_status()

    def upload_file(self, local: Path, remote_path: str, content_type: str) -> str:
        url = f"{self.base}/storage/v1/object/{self.cfg.bucket}/{remote_path.lstrip('/')}"
        with local.open("rb") as fh:
            r = self.client.post(
                url,
                headers={
                    "apikey": self.key,
                    "Authorization": f"Bearer {self.key}",
                    "Content-Type": content_type,
                    "x-upsert": "true",
                },
                content=fh,
            )
        r.raise_for_status()
        return remote_path


def discover_month(cfg: Config, lo: date, hi: date) -> list[str]:
    query = f"from:ZeeBusiness #splmidcapstocks since:{lo.isoformat()} until:{(hi + timedelta(days=1)).isoformat()}"
    urls: list[str] = []
    seen: set[str] = set()
    for instance in cfg.nitter_instances:
        cursor_url = f"{instance.rstrip('/')}/search?f=tweets&q={quote(query)}"
        for page_idx in range(8):
            proxy = rotated_proxy(cfg, f"{instance}|{lo}|{page_idx}")
            try:
                with httpx.Client(
                    timeout=30.0,
                    headers={"User-Agent": USER_AGENT},
                    follow_redirects=True,
                    proxy=proxy,
                ) as client:
                    r = client.get(cursor_url)
                    r.raise_for_status()
                soup = BeautifulSoup(r.text, "html.parser")
            except Exception as exc:  # noqa: BLE001
                log_event({"nitter_error": instance, "proxy": bool(proxy), "error": str(exc)})
                break

            for item in soup.select(".timeline-item"):
                content = item.select_one(".tweet-content")
                text = content.get_text(" ", strip=True).lower() if content else ""
                if "splmidcapstocks" not in text:
                    continue
                link = item.select_one("a.tweet-link")
                if not link:
                    continue
                href = link.get("href", "")
                m = re.search(r"/ZeeBusiness/status/(\d+)", href, re.I)
                if not m:
                    continue
                video_url = f"https://x.com/ZeeBusiness/status/{m.group(1)}/video/1"
                if video_url not in seen:
                    seen.add(video_url)
                    urls.append(video_url)
                    if len(urls) >= cfg.limit:
                        return urls

            more = soup.select_one("div.show-more a, a.load-more")
            if not more or not more.get("href"):
                break
            cursor_url = urljoin(instance.rstrip("/") + "/", more["href"])
            time.sleep(1.0 if not cfg.proxies else 0.25)
        if urls:
            return urls
    return urls


def normalize_video_url(url: str | None) -> str | None:
    if not url:
        return None
    m = re.search(r"/(?:ZeeBusiness|zeebusiness)/status/(\d+)", url, re.I)
    if not m:
        m = re.search(r"(?:twitter|x)\.com/i/status/(\d+)", url, re.I)
    if not m:
        return None
    return f"https://x.com/ZeeBusiness/status/{m.group(1)}/video/1"


def add_unique_urls(target: list[str], seen: set[str], source_urls: list[str]) -> int:
    added = 0
    for raw in source_urls:
        url = normalize_video_url(raw)
        if not url or url in seen:
            continue
        seen.add(url)
        target.append(url)
        added += 1
    return added


def urls_from_file(path: str) -> list[str]:
    src = Path(path)
    if not src.exists():
        raise SystemExit(f"URLs file not found: {path}")
    urls = []
    for line in src.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            urls.append(line)
    return urls


def urls_from_existing_video_jobs(supa: Supabase) -> list[str]:
    urls = []
    for row in supa.list_records("video_jobs"):
        raw = row.get("url") or row.get("tweet_url") or row.get("source_url")
        url = normalize_video_url(raw)
        if url:
            urls.append(url)
    return urls


def row_completeness_score(row: dict) -> int:
    score = 0
    for key in (
        "image_url",
        "source_url",
        "analyst",
        "stock",
        "stock_full_name",
        "entry_date",
        "entry_price",
        "target_price",
        "video_timestamp",
    ):
        value = row.get(key)
        if value not in (None, "", [], {}):
            score += 1
    return score


def enriched_duplicate_key(row: dict) -> tuple:
    source_url = normalize_video_url(row.get("source_url"))
    if source_url:
        ts = row.get("video_timestamp")
        try:
            ts_key = round(float(ts), 1)
        except (TypeError, ValueError):
            ts_key = None
        stock = re.sub(r"[^a-z0-9]", "", str(row.get("stock") or row.get("stock_full_name") or "").lower())
        return (
            "source",
            source_url,
            ts_key,
            stock,
            row.get("entry_date"),
            row.get("target_price"),
            row.get("reco"),
        )
    return ("call", row.get("call_id") or stable_id(json.dumps(row, sort_keys=True, default=str)))


def processed_source_urls(enriched_rows: list[dict]) -> set[str]:
    urls = set()
    for row in enriched_rows:
        url = normalize_video_url(row.get("source_url"))
        if url:
            urls.add(url)
    return urls


def cleanup_duplicate_enriched_calls(supa: Supabase) -> tuple[list[dict], int]:
    rows = supa.list_record_rows("enriched_calls")
    keep_by_key: dict[tuple, dict] = {}
    delete_ids: list[str] = []
    for item in rows:
        payload = item.get("payload") or {}
        key = enriched_duplicate_key(payload)
        current = keep_by_key.get(key)
        if current is None:
            keep_by_key[key] = item
            continue
        current_score = row_completeness_score(current.get("payload") or {})
        incoming_score = row_completeness_score(payload)
        if incoming_score > current_score:
            delete_ids.append(str(current.get("id")))
            keep_by_key[key] = item
        else:
            delete_ids.append(str(item.get("id")))
    supa.delete_records("enriched_calls", delete_ids)
    kept = [item.get("payload") or {} for item in keep_by_key.values()]
    return kept, len(delete_ids)


def cleanup_duplicate_video_jobs(supa: Supabase) -> tuple[list[str], int]:
    rows = supa.list_record_rows("video_jobs")
    keep_by_url: dict[str, dict] = {}
    delete_ids: list[str] = []

    def score(payload: dict) -> tuple:
        status_rank = {"done": 4, "extracting": 3, "ocr": 3, "downloading": 2, "queued": 1, "failed": 0}
        return (
            status_rank.get(str(payload.get("status") or ""), 0),
            int(payload.get("calls_added") or 0),
            int(payload.get("cards_found") or 0),
            row_completeness_score(payload),
        )

    for item in rows:
        payload = item.get("payload") or {}
        url = normalize_video_url(payload.get("url"))
        if not url:
            continue
        current = keep_by_url.get(url)
        if current is None:
            keep_by_url[url] = item
            continue
        if score(payload) > score(current.get("payload") or {}):
            delete_ids.append(str(current.get("id")))
            keep_by_url[url] = item
        else:
            delete_ids.append(str(item.get("id")))
    supa.delete_records("video_jobs", delete_ids)
    urls = list(keep_by_url.keys())
    return urls, len(delete_ids)


# OCR geometry copied into this standalone file.
PANEL_X = (0.02, 0.47)
NAME_X = (0.02, 0.50)
ANALYST_DY = (-0.255, -0.195)
NAME_DY = (-0.190, -0.100)
ACTION_DY = (-0.100, -0.040)
VALUES_DY = (-0.015, 0.105)
ENTRY_RE = re.compile(r"@\s*(\d[\d,]*\.?\d*)")
DATE_RE = re.compile(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})")
PRICE_RE = re.compile(r"\d{2,}\.\d{2}")
BANNER_WORDS = re.compile(r"wealth|creation|pick|target|duration|stop|loss|buy|sell|खरीद|बेच", re.I)


def crop(frame, x0, x1, y0, y1):
    h, w = frame.shape[:2]
    y0 = max(0.0, y0)
    y1 = min(1.0, y1)
    return frame[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)]


def prep(sub, fx=4.0):
    gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)
    if gray.mean() < 127:
        gray = 255 - gray
    return cv2.copyMakeBorder(gray, 18, 18, 18, 18, cv2.BORDER_CONSTANT, value=255)


def ocr(frame, x0, x1, y0, y1, *, psm=7, lang="eng", whitelist=None, fx=4.0) -> str:
    sub = crop(frame, x0, x1, y0, y1)
    if sub.size == 0:
        return ""
    cfg = f"--psm {psm} --oem 1"
    if whitelist:
        cfg += f" -c tessedit_char_whitelist={whitelist}"
    try:
        return pytesseract.image_to_string(prep(sub, fx), lang=lang, config=cfg)
    except Exception:
        return pytesseract.image_to_string(prep(sub, fx), config=cfg)


def find_band_y(frame) -> float | None:
    coarse = (0.42, 0.45, 0.48, 0.51, 0.54, 0.57)
    fine = [i / 200 for i in range(84, 116, 2)]
    for y in list(coarse) + [x for x in fine if x not in coarse]:
        text = ocr(frame, PANEL_X[0], PANEL_X[1], y, y + 0.04, psm=6, fx=3.0).lower()
        if "target" in text and ("stop" in text or "loss" in text or "duration" in text):
            return y
    return None


def quick_card_signal(frame) -> tuple[bool, str]:
    text = ocr(frame, 0.02, 0.56, 0.18, 0.76, psm=6, lang="eng", fx=2.0).lower()
    if not text:
        return False, ""
    has_target = "target" in text or "tgt" in text
    has_risk = "stop" in text or "loss" in text or "duration" in text
    has_action = "buy" in text or "sell" in text
    has_numbers = len(re.findall(r"\d{2,}", text)) >= 2
    has_entry = "@" in text or "cmp" in text
    ok = has_target and has_numbers and (has_risk or has_action or has_entry)
    return ok, re.sub(r"\s+", " ", text).strip()[:160]


def valid_stock_name(name: str | None) -> bool:
    if not name:
        return False
    name = name.strip()
    if len(name) < 3 or len(name) > 60:
        return False
    if BANNER_WORDS.search(name):
        return False
    letters = sum(ch.isalpha() for ch in name)
    return letters >= 3


def clean_stock_name(text: str) -> str:
    text = text.split("@")[0]
    text = re.sub(r"[^A-Za-z0-9 .&'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" .-&|")
    text = re.sub(r"^(?:\d+\s+)+", "", text)
    return text.strip(" .-&|")


def read_name_entry(frame, by: float) -> tuple[str | None, float | None]:
    raw = ocr(frame, NAME_X[0], NAME_X[1], by + NAME_DY[0], by + NAME_DY[1], psm=6)
    best = None
    best_score = -1
    entry = None
    for line in raw.splitlines():
        m = ENTRY_RE.search(line)
        if m and entry is None:
            entry = clean_num(m.group(1))
        cand = clean_stock_name(line)
        if not valid_stock_name(cand):
            continue
        score = sum(ch.islower() for ch in cand)
        if score > best_score:
            best, best_score = cand, score
    return best, entry


def read_analyst_hindi(frame, by: float) -> str | None:
    raw = ocr(frame, PANEL_X[0], 0.54, by + ANALYST_DY[0], by + ANALYST_DY[1], psm=13, lang="hin+eng", fx=4.0)
    raw = re.sub(r"[^\u0900-\u097F\s]", " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    raw = re.sub(
        r"(?:की|का|के)?\s*(?:राय|रास|राश|राथ|पसंद|पसन्द|पसद|पसं).*$",
        "",
        raw,
    ).strip()
    raw = re.sub(r"(?:विशेषज्ञ|एनालिस्ट|एक्सपर्ट|सलाह|बाजार).*", "", raw).strip()
    raw = re.sub(r"^(?:है|ह)\s+", "", raw).strip()
    return raw or None


def read_action(frame, by: float) -> str:
    sub = crop(frame, 0.10, 0.42, by + ACTION_DY[0], by + ACTION_DY[1])
    if sub.size == 0:
        return "UNKNOWN"
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    green = float(cv2.inRange(hsv, (35, 60, 60), (85, 255, 255)).mean())
    red = float((cv2.inRange(hsv, (0, 60, 60), (10, 255, 255)) + cv2.inRange(hsv, (170, 60, 60), (180, 255, 255))).mean())
    if max(green, red) < 5:
        return "UNKNOWN"
    return "BUY" if green >= red else "SELL"


def parse_values(text: str) -> tuple[float | None, list[float]]:
    nums = [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", text)]
    if not nums:
        return None, []
    # In this card band, first number is often stop loss when present, later
    # numbers are target(s). If only one number appears, it is usually target.
    if "--" in text or len(nums) == 1:
        return None, nums
    return nums[0], nums[1:]


def read_values(frame, by: float) -> tuple[float | None, list[float]]:
    text = ocr(frame, PANEL_X[0], PANEL_X[1], by + VALUES_DY[0], by + VALUES_DY[1], psm=6, lang="eng")
    return parse_values(text)


def read_timeframe(frame, by: float) -> tuple[str, int | None]:
    for d in (0.045, 0.06, 0.075, 0.09, 0.105):
        text = ocr(frame, 0.33, 0.48, by + d, by + d + 0.05, psm=7, lang="eng+hin", fx=5.0)
        low = text.lower()
        m = re.search(r"(\d{1,2})\s*[-–—]\s*(\d{1,2})", text)
        if not m:
            continue
        hi = int(m.group(2))
        if re.search(r"साल|year|\byr\b", low):
            return "LONGTERM", hi * 12
        if re.search(r"हफ्त|सप्ताह|week", low):
            return "SWING", max(1, round(hi / 4))
        if re.search(r"दिन|\bday", low):
            return "SWING", max(1, round(hi / 30))
        return "POSITIONAL", hi
    return "UNKNOWN", None


def read_date(frame) -> str | None:
    for y in (0.77, 0.79, 0.81, 0.83):
        text = ocr(frame, 0.80, 0.99, y, y + 0.05, psm=7, whitelist="0123456789/-", fx=3.0)
        m = DATE_RE.search(text)
        if not m:
            continue
        d, mo, yr = (int(x) for x in m.groups())
        if yr < 100:
            yr += 2000
        if 1 <= d <= 31 and 1 <= mo <= 12 and 2015 <= yr <= 2035:
            return f"{yr:04d}-{mo:02d}-{d:02d}"
    return None


def read_current_price(frame, by: float) -> float | None:
    for d in (0.085, 0.10, 0.115, 0.13, 0.145):
        text = ocr(frame, PANEL_X[0], 0.30, by + d, by + d + 0.05, psm=7, whitelist="0123456789.", fx=4.0)
        m = PRICE_RE.search(text)
        if m:
            return clean_num(m.group(0))
    return None


def extract_card(frame) -> dict | None:
    by = find_band_y(frame)
    if by is None:
        return None
    stock, entry = read_name_entry(frame, by)
    stop_loss, targets = read_values(frame, by)
    current_price = read_current_price(frame, by)
    levels = [x for x in [entry, stop_loss, *targets] if x]
    if current_price and levels:
        ref = sorted(levels)[len(levels) // 2]
        if not (0.3 * ref <= current_price <= 3.0 * ref):
            current_price = None
    timeframe, horizon = read_timeframe(frame, by)
    return {
        "stock": stock,
        "analyst": read_analyst_hindi(frame, by),
        "entry": entry,
        "current_price": current_price,
        "targets": targets,
        "stop_loss": stop_loss,
        "action": read_action(frame, by),
        "timeframe": timeframe,
        "horizon_months": horizon,
        "entry_date": read_date(frame),
    }


def norm_name(name: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def vote(values: list, default=None):
    vals = [x for x in values if x not in (None, "", [])]
    if not vals:
        return default
    hashable = [tuple(x) if isinstance(x, list) else x for x in vals]
    winner = Counter(hashable).most_common(1)[0][0]
    return list(winner) if isinstance(winner, tuple) else winner


def merge_cards(records: list[dict], min_support: int) -> list[dict]:
    clusters: list[dict] = []
    for rec in records:
        key = norm_name(rec.get("stock"))
        if not key:
            continue
        placed = False
        for cluster in clusters:
            a, b = key, cluster["key"]
            overlap = len(set(a) & set(b)) / max(1, len(set(a) | set(b)))
            if a == b or overlap >= 0.72:
                cluster["items"].append(rec)
                placed = True
                break
        if not placed:
            clusters.append({"key": key, "items": [rec]})

    merged = []
    for cluster in clusters:
        items = cluster["items"]
        if len(items) < min_support:
            continue

        def q(rec):
            score = 0
            for key in ("stock", "targets", "entry", "stop_loss", "timeframe", "entry_date"):
                if rec.get(key) not in (None, "", [], "UNKNOWN"):
                    score += 1
            if rec.get("analyst"):
                score += 10
            return score, -rec["_ts"]

        rep = max(items, key=q)
        merged.append(
            {
                "stock": vote([x.get("stock") for x in items]),
                "analyst": vote([x.get("analyst") for x in items]),
                "entry": vote([x.get("entry") for x in items]),
                "current_price": vote([x.get("current_price") for x in items]),
                "targets": vote([x.get("targets") for x in items]) or [],
                "stop_loss": Counter(x.get("stop_loss") for x in items).most_common(1)[0][0],
                "action": vote([x.get("action") for x in items], "UNKNOWN"),
                "timeframe": vote([x.get("timeframe") for x in items], "UNKNOWN"),
                "horizon_months": vote([x.get("horizon_months") for x in items]),
                "entry_date": vote([x.get("entry_date") for x in items]),
                "_ts": rep["_ts"],
                "_first_ts": min(x["_ts"] for x in items),
                "_support": len(items),
            }
        )
    merged.sort(key=lambda x: x["_first_ts"])
    return merged


def download_video(url: str, outdir: Path, proxy: str | None = None, job_id: str | None = None) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    progress_last = {"ts": 0.0}

    def progress_hook(status: dict) -> None:
        now = time.time()
        state = status.get("status")
        label = job_id or stable_id(url)
        if state == "downloading" and now - progress_last["ts"] >= 10:
            progress_last["ts"] = now
            downloaded = status.get("downloaded_bytes") or 0
            total = status.get("total_bytes") or status.get("total_bytes_estimate") or 0
            row = {
                "job": label,
                "phase": "download",
                "downloaded_mb": round(downloaded / 1024 / 1024, 1),
            }
            if total:
                row["total_mb"] = round(total / 1024 / 1024, 1)
                row["pct"] = round(downloaded / total * 100, 1)
            log_event(row)
        elif state == "finished":
            log_event({"job": label, "phase": "download_finished", "filename": Path(status.get("filename", "")).name})

    opts = {
        "outtmpl": str(outdir / "%(id)s.%(ext)s"),
        "format": "bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [progress_hook],
    }
    if proxy:
        opts["proxy"] = proxy
    log_event({"job": job_id or stable_id(url), "phase": "download_start", "url": url, "proxy": bool(proxy)})
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = Path(ydl.prepare_filename(info))
        if path.suffix != ".mp4":
            candidate = path.with_suffix(".mp4")
            if candidate.exists():
                path = candidate
    return path


def iter_sampled_frames(video_path: Path, fps: float, max_frames: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(source_fps / fps))
    idx = 0
    yielded = 0
    while yielded < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            yield idx / source_fps, frame
            yielded += 1
        idx += 1
    cap.release()


def soft_scan_candidates(video_path: Path, cfg: Config, job_id: str) -> tuple[list[float], int]:
    hits: list[float] = []
    checked = 0
    for ts, frame in iter_sampled_frames(video_path, cfg.soft_scan_fps, cfg.soft_max_frames):
        checked += 1
        ok, snippet = quick_card_signal(frame)
        if ok:
            hits.append(ts)
            log_event({"job": job_id, "phase": "soft_hit", "ts": round(ts, 2), "text": snippet})
        if checked % 25 == 0:
            log_event({"job": job_id, "phase": "soft_scan", "checked": checked, "hits": len(hits)})
    return hits, checked


def expanded_deep_timestamps(hits: list[float], window: float, step: float) -> list[float]:
    if not hits:
        return []
    out: set[float] = set()
    step = max(step, 0.5)
    slots = int(window / step)
    for hit in hits:
        for i in range(-slots, slots + 1):
            out.add(round(max(0.0, hit + i * step), 2))
    return sorted(out)


def frame_at(video_path: Path, ts: float):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, ts) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def yahoo_search(name: str) -> tuple[str, str] | None:
    if not name:
        return None
    queries = [name]
    words = name.split()
    for n in (3, 2, 1):
        if len(words) > n:
            queries.append(" ".join(words[:n]))
    with httpx.Client(timeout=20.0, headers={"User-Agent": USER_AGENT}) as client:
        for q in queries:
            try:
                r = client.get("https://query2.finance.yahoo.com/v1/finance/search", params={"q": q, "quotesCount": 8, "newsCount": 0})
                r.raise_for_status()
                quotes = r.json().get("quotes", [])
            except Exception:
                continue
            nse = [x for x in quotes if str(x.get("symbol", "")).upper().endswith(".NS")]
            pick = nse[0] if nse else None
            if pick:
                return str(pick["symbol"]).upper().removesuffix(".NS"), pick.get("longname") or pick.get("shortname") or name
    return None


def yahoo_history(symbol: str, start: date, end: date) -> list[dict]:
    p1 = int(datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp()) - 7 * 86400
    p2 = int(datetime(end.year, end.month, end.day, tzinfo=timezone.utc).timestamp()) + 2 * 86400
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}.NS"
    try:
        with httpx.Client(timeout=30.0, headers={"User-Agent": USER_AGENT}) as client:
            r = client.get(url, params={"period1": p1, "period2": p2, "interval": "1d"})
            r.raise_for_status()
            result = r.json()["chart"]["result"][0]
    except Exception:
        return []
    stamps = result.get("timestamp") or []
    q = result["indicators"]["quote"][0]
    bars = []
    for i, ts in enumerate(stamps):
        o, h, low, close = q["open"][i], q["high"][i], q["low"][i], q["close"][i]
        if None in (o, h, low, close):
            continue
        day = datetime.fromtimestamp(ts, tz=timezone.utc).date()
        if start <= day <= end:
            bars.append({"day": day, "open": o, "high": h, "low": low, "close": close})
    return bars


def enrich_card(card: dict, image_url: str, source_url: str, ts: float) -> dict:
    name = card.get("stock") or ""
    action = card.get("action") or "BUY"
    is_buy = action != "SELL"
    resolved = yahoo_search(name)
    symbol, full_name = resolved if resolved else (None, name or None)
    notes = [] if resolved else [f"unresolved symbol for {name!r}"]
    entry_day = None
    try:
        entry_day = date.fromisoformat(card["entry_date"]) if card.get("entry_date") else None
    except ValueError:
        pass
    target_price = float(card["targets"][0]) if card.get("targets") else None
    stop_loss = card.get("stop_loss")
    entry_price = card.get("entry")
    if entry_price is None and symbol and entry_day:
        bars = yahoo_history(symbol, entry_day, entry_day + timedelta(days=7))
        if bars:
            entry_price = bars[0]["close"]
            notes.append("entry price from NSE close")
    expected = pct(entry_price, target_price, is_buy) if entry_price and target_price else None
    horizon = card.get("horizon_months") or 6
    target_day = entry_day + timedelta(days=horizon * 30) if entry_day else None
    row = {
        "call_id": card["call_id"],
        "image_url": image_url,
        "source_url": source_url,
        "video_timestamp": round(ts, 2),
        "analyst": card.get("analyst"),
        "analyst_company": None,
        "stock": symbol,
        "stock_full_name": full_name,
        "entry_date": entry_day.isoformat() if entry_day else None,
        "target_date": target_day.isoformat() if target_day else None,
        "entry_price": round(entry_price, 2) if entry_price else None,
        "stop_loss": stop_loss,
        "target_price": target_price,
        "expected_return_pct": expected,
        "reco": "Sell" if action == "SELL" else "Buy",
        "status": None,
        "current_price": None,
        "actual_return_pct": None,
        "annualized_pct": None,
        "success": None,
        "source": None,
        "platform": "Zee Business",
        "program": "SPL Midcap",
        "theme": None,
        "enriched_at": utcnow(),
        "notes": notes,
    }
    if not (symbol and entry_day and entry_price):
        row["status"] = "incomplete" if entry_day else "open"
        return row

    today = date.today()
    scan_end = min(target_day or today, today)
    bars = yahoo_history(symbol, entry_day + timedelta(days=1), scan_end)
    exit_price = None
    close_day = None
    success = None
    for b in bars:
        hit_target = target_price is not None and (b["high"] >= target_price if is_buy else b["low"] <= target_price)
        hit_stop = stop_loss is not None and (b["low"] <= stop_loss if is_buy else b["high"] >= stop_loss)
        if hit_target:
            exit_price, close_day, success = target_price, b["day"], True
            break
        if hit_stop:
            exit_price, close_day, success = stop_loss, b["day"], False
            break
    if close_day:
        row["status"] = f"closed on {close_day.strftime('%d/%m/%Y')}"
    elif target_day and today >= target_day:
        last = bars[-1] if bars else None
        exit_price = last["close"] if last else entry_price
        close_day = last["day"] if last else target_day
        success = (exit_price >= entry_price) if is_buy else (exit_price <= entry_price)
        row["status"] = f"closed on {close_day.strftime('%d/%m/%Y')}"
    else:
        last = bars[-1] if bars else None
        exit_price = last["close"] if last else card.get("current_price") or entry_price
        close_day = today
        success = None
        row["status"] = "open"
    if exit_price:
        row["current_price"] = round(exit_price, 2)
        actual = pct(entry_price, exit_price, is_buy)
        row["actual_return_pct"] = actual
        held = max((close_day - entry_day).days, 1)
        row["annualized_pct"] = round(actual * 365.0 / held, 2)
    row["success"] = success
    return row


def process_video(cfg: Config, supa: Supabase, url: str) -> tuple[dict, list[dict], list[dict]]:
    job_id = stable_id(url)
    job_dir = cfg.workdir / "video_cache" / job_id
    frame_dir = cfg.workdir / "video_frames" / job_id
    frame_dir.mkdir(parents=True, exist_ok=True)
    started = utcnow()
    proxy = rotated_proxy(cfg, url)
    log_event({"job": job_id, "phase": "start", "url": url})
    video_path = download_video(url, job_dir, proxy=proxy, job_id=job_id)
    ocr_started = time.time()
    log_event({"job": job_id, "phase": "ocr_start", "video_file": video_path.name, "soft_scan": cfg.soft_scan})
    records = []
    done = 0
    soft_checked = 0
    if cfg.soft_scan:
        hits, soft_checked = soft_scan_candidates(video_path, cfg, job_id)
        deep_ts = expanded_deep_timestamps(hits, cfg.deep_window, cfg.deep_step)
        log_event(
            {
                "job": job_id,
                "phase": "deep_plan",
                "soft_checked": soft_checked,
                "soft_hits": len(hits),
                "deep_frames": len(deep_ts),
            }
        )
        frame_iter = ((ts, frame_at(video_path, ts)) for ts in deep_ts[: cfg.max_frames])
    else:
        frame_iter = iter_sampled_frames(video_path, cfg.frame_fps, cfg.max_frames)

    for ts, frame in frame_iter:
        if frame is None:
            continue
        done += 1
        parsed = extract_card(frame)
        if parsed and valid_stock_name(parsed.get("stock")) and parsed.get("targets"):
            parsed["_ts"] = ts
            records.append(parsed)
        if done % 10 == 0:
            elapsed = max(time.time() - ocr_started, 0.001)
            log_event(
                {
                    "job": job_id,
                    "phase": "deep_ocr",
                    "frames": done,
                    "candidate_reads": len(records),
                    "fps": round(done / elapsed, 2),
                }
            )

    merged = merge_cards(records, cfg.min_card_frames)
    log_event(
        {
            "job": job_id,
            "phase": "merge",
            "soft_checked": soft_checked,
            "frames": done,
            "candidate_reads": len(records),
            "cards": len(merged),
            "seconds": round(time.time() - ocr_started, 1),
        }
    )
    cards = []
    calls = []
    enriched = []
    for idx, parsed in enumerate(merged):
        ts = parsed["_ts"]
        call_id = f"vcall_{job_id}_{idx}"
        parsed["call_id"] = call_id
        frame = frame_at(video_path, ts)
        image_url = None
        if frame is not None:
            name = f"{int(ts)}_{idx}.png"
            local_frame = frame_dir / name
            cv2.imwrite(str(local_frame), frame)
            supa.upload_file(local_frame, f"video_frames/{job_id}/{name}", "image/png")
            image_url = f"/video/frames/{job_id}/{name}"

        cards.append(
            {
                "timestamp": round(ts, 2),
                "image_url": image_url or "",
                "call_id": call_id,
                "analyst": parsed.get("analyst"),
                "analyst_company": None,
                "stock": parsed.get("stock"),
                "action": parsed.get("action") or "UNKNOWN",
                "entry": parsed.get("entry"),
                "current_price": parsed.get("current_price"),
                "targets": parsed.get("targets") or [],
                "stop_loss": parsed.get("stop_loss"),
                "timeframe": parsed.get("timeframe") or "UNKNOWN",
                "entry_date": parsed.get("entry_date"),
                "horizon_months": parsed.get("horizon_months"),
            }
        )
        calls.append(
            {
                "id": call_id,
                "tweet_id": call_id,
                "handle": "ZeeBusiness",
                "raw_text": f"{parsed.get('stock')} | {parsed.get('analyst') or ''} | T:{parsed.get('targets')}",
                "tweet_url": url,
                "tweet_created_at": None,
                "is_call": True,
                "ticker": None,
                "company": parsed.get("stock"),
                "sector": "OTHER",
                "market_cap": "SMALL",
                "action": parsed.get("action") or "UNKNOWN",
                "entry": parsed.get("entry"),
                "targets": parsed.get("targets") or [],
                "stop_loss": parsed.get("stop_loss"),
                "timeframe": parsed.get("timeframe") or "UNKNOWN",
                "confidence": 0.9,
                "notes": ["source=video uploader=ZeeBusiness", f"analyst_hindi={parsed.get('analyst') or ''}"],
                "status": "pending",
                "scraped_at": utcnow(),
                "reviewed_at": None,
                "source": "video",
                "scrape_run_id": job_id,
            }
        )
        enriched.append(enrich_card(parsed, image_url, url, ts))

    job = {
        "id": job_id,
        "url": url,
        "status": "done",
        "message": f"Done - {len(cards)} card(s) found.",
        "error": None,
        "uploader": "ZeeBusiness",
        "video_file": None,
        "template_id": "zbusiness",
        "template_name": "ZBusiness analyst call card",
        "frames_total": soft_checked + done,
        "frames_done": done,
        "cards_found": len(cards),
        "calls_added": len(calls),
        "started_at": started,
        "finished_at": utcnow(),
        "cards": cards,
    }
    if cfg.cleanup_videos:
        try:
            video_path.unlink(missing_ok=True)
        except OSError:
            pass
    return job, calls, enriched


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=parse_day, default=date(2023, 4, 1))
    p.add_argument("--end", type=parse_day, default=datetime.now(timezone.utc).date())
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--concurrency", type=int, default=int(os.getenv("RUNPOD_CONCURRENCY", "16")))
    p.add_argument("--frame-fps", type=float, default=1.0, help="Full-scan FPS when --no-soft-scan is used")
    p.add_argument("--max-frames", type=int, default=1200)
    p.add_argument("--min-card-frames", type=int, default=2)
    p.add_argument("--no-soft-scan", action="store_true", help="Disable sparse preflight scan and run full extraction on sampled frames")
    p.add_argument("--soft-scan-fps", type=float, default=float(os.getenv("RUNPOD_SOFT_SCAN_FPS", "0.2")), help="Cheap preflight scan FPS; 0.2 means one frame every 5 seconds")
    p.add_argument("--soft-max-frames", type=int, default=int(os.getenv("RUNPOD_SOFT_MAX_FRAMES", "240")), help="Max frames checked by soft scan before deep OCR")
    p.add_argument("--deep-window", type=float, default=float(os.getenv("RUNPOD_DEEP_WINDOW", "6")), help="Seconds around each soft hit to deep OCR")
    p.add_argument("--deep-step", type=float, default=float(os.getenv("RUNPOD_DEEP_STEP", "2")), help="Seconds between deep OCR frames around each hit")
    p.add_argument("--bucket", default=os.getenv("SUPABASE_MEDIA_BUCKET", "stock-call-media"))
    p.add_argument("--table", default=os.getenv("SUPABASE_DATA_TABLE", "app_records"))
    p.add_argument("--nitter", default=",".join(DEFAULT_NITTER))
    p.add_argument("--proxy-list", default=os.getenv("PROXY_LIST", ""), help="Comma-separated http/socks proxies for Nitter and yt-dlp")
    p.add_argument("--proxy-file", default=os.getenv("PROXY_FILE", ""), help="File with one proxy per line")
    p.add_argument("--workdir", default="./spl_speedrun_work")
    p.add_argument("--existing-video-jobs", action="store_true", help="Process ZeeBusiness URLs already stored in Supabase video_jobs")
    p.add_argument("--urls-file", default=os.getenv("URLS_FILE", ""), help="File containing one ZeeBusiness X/Twitter/Nitter status URL per line")
    p.add_argument("--skip-discovery", action="store_true", help="Do not call Nitter; use --existing-video-jobs and/or --urls-file only")
    p.add_argument("--skip-processed", action=argparse.BooleanOptionalAction, default=True, help="Skip videos that already have enriched_calls rows")
    p.add_argument("--cleanup-duplicates", action=argparse.BooleanOptionalAction, default=True, help="Delete duplicate enriched_calls/video_jobs rows before queueing")
    p.add_argument("--clear-enriched", action="store_true", help="Delete all enriched_calls before processing. Off by default for resumable runs.")
    p.add_argument("--no-clear-enriched", action="store_true", help="Compatibility no-op; enriched_calls are preserved unless --clear-enriched is passed")
    p.add_argument("--keep-source-videos", action="store_true")
    args = p.parse_args()
    supabase_url = os.getenv("SUPABASE_URL") or os.getenv("url")
    service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_SECRET_KEY") or os.getenv("secret_key")
    if not supabase_url or not service_key:
        raise SystemExit("Set SUPABASE_URL/url and SUPABASE_SERVICE_ROLE_KEY/secret_key first.")
    proxies = split_csv(args.proxy_list)
    if args.proxy_file:
        proxy_path = Path(args.proxy_file)
        if proxy_path.exists():
            proxies.extend(
                line.strip()
                for line in proxy_path.read_text().splitlines()
                if line.strip() and not line.strip().startswith("#")
            )
    return Config(
        supabase_url=supabase_url,
        service_key=service_key,
        bucket=args.bucket,
        table=args.table,
        nitter_instances=[x.strip().rstrip("/") for x in args.nitter.split(",") if x.strip()],
        proxies=proxies,
        start=args.start,
        end=args.end,
        limit=args.limit,
        concurrency=args.concurrency,
        frame_fps=args.frame_fps,
        max_frames=args.max_frames,
        min_card_frames=args.min_card_frames,
        soft_scan=not args.no_soft_scan,
        soft_scan_fps=args.soft_scan_fps,
        soft_max_frames=args.soft_max_frames,
        deep_window=args.deep_window,
        deep_step=args.deep_step,
        clear_enriched=args.clear_enriched and not args.no_clear_enriched,
        cleanup_videos=not args.keep_source_videos,
        skip_processed=args.skip_processed,
        cleanup_duplicates=args.cleanup_duplicates,
        existing_video_jobs=args.existing_video_jobs,
        urls_file=args.urls_file or None,
        skip_discovery=args.skip_discovery,
        workdir=Path(args.workdir),
    )


def main() -> None:
    cfg = parse_args()
    cfg.workdir.mkdir(parents=True, exist_ok=True)
    supa = Supabase(cfg)
    log_event(
        {
            "start": cfg.start.isoformat(),
            "end": cfg.end.isoformat(),
            "concurrency": cfg.concurrency,
            "proxies": len(cfg.proxies),
            "nitter_instances": len(cfg.nitter_instances),
            "existing_video_jobs": cfg.existing_video_jobs,
            "urls_file": bool(cfg.urls_file),
            "skip_discovery": cfg.skip_discovery,
            "soft_scan": cfg.soft_scan,
            "soft_scan_fps": cfg.soft_scan_fps,
            "soft_max_frames": cfg.soft_max_frames,
            "deep_window": cfg.deep_window,
            "deep_step": cfg.deep_step,
            "skip_processed": cfg.skip_processed,
            "cleanup_duplicates": cfg.cleanup_duplicates,
            "clear_enriched": cfg.clear_enriched,
        }
    )

    enriched_rows: list[dict] = []
    existing_video_job_urls: list[str] | None = None
    if cfg.clear_enriched:
        log_event({"clear": "enriched_calls"})
        supa.delete_collection("enriched_calls")
    elif cfg.cleanup_duplicates:
        enriched_rows, removed_enriched = cleanup_duplicate_enriched_calls(supa)
        existing_video_job_urls, removed_video_jobs = cleanup_duplicate_video_jobs(supa)
        log_event(
            {
                "app_records_loaded": True,
                "enriched_calls": len(enriched_rows),
                "duplicate_enriched_calls_deleted": removed_enriched,
                "video_jobs": len(existing_video_job_urls),
                "duplicate_video_jobs_deleted": removed_video_jobs,
            }
        )
    elif cfg.skip_processed:
        enriched_rows = supa.list_records("enriched_calls")
        log_event({"app_records_loaded": True, "enriched_calls": len(enriched_rows)})

    processed_urls = processed_source_urls(enriched_rows) if cfg.skip_processed and not cfg.clear_enriched else set()

    urls = []
    seen = set()

    if cfg.urls_file:
        file_urls = urls_from_file(cfg.urls_file)
        added = add_unique_urls(urls, seen, file_urls)
        log_event({"urls_file": cfg.urls_file, "found": len(file_urls), "added": added, "total_urls": len(urls)})

    if cfg.existing_video_jobs:
        existing_urls = existing_video_job_urls if existing_video_job_urls is not None else urls_from_existing_video_jobs(supa)
        added = add_unique_urls(urls, seen, existing_urls)
        log_event({"existing_video_jobs": True, "found": len(existing_urls), "added": added, "total_urls": len(urls)})

    if not cfg.skip_discovery:
        windows = month_windows(cfg.start, cfg.end)
        for idx, (lo, hi) in enumerate(windows, 1):
            found = discover_month(cfg, lo, hi)
            added = add_unique_urls(urls, seen, found)
            log_event(
                {
                    "discover": f"{idx}/{len(windows)}",
                    "month": f"{lo}..{hi}",
                    "found": len(found),
                    "added": added,
                    "total_urls": len(urls),
                }
            )

    if processed_urls:
        before = len(urls)
        urls = [url for url in urls if url not in processed_urls]
        seen = set(urls)
        log_event(
            {
                "skip_processed": True,
                "already_enriched_urls": len(processed_urls),
                "skipped_urls": before - len(urls),
                "remaining_urls": len(urls),
            }
        )

    if not urls:
        log_event(
            {
                "warning": "No videos queued. If Nitter is blocked from this RunPod IP, rerun with --existing-video-jobs --skip-discovery, --urls-file, or PROXY_LIST/PROXY_FILE.",
                "done": True,
            }
        )
        return

    all_jobs, all_calls, all_enriched = [], [], []
    from concurrent.futures import ThreadPoolExecutor, as_completed

    log_event(
        {
            "queue": len(urls),
            "workers": cfg.concurrency,
            "storage": "uploads call-card PNG frames only; source videos stay local and are deleted after OCR",
        }
    )
    with ThreadPoolExecutor(max_workers=cfg.concurrency) as pool:
        futures = {pool.submit(process_video, cfg, supa, url): url for url in urls}
        for i, fut in enumerate(as_completed(futures), 1):
            url = futures[fut]
            try:
                job, calls, enriched = fut.result()
                all_jobs.append(job)
                all_calls.extend(calls)
                all_enriched.extend(enriched)
                log_event({"processed": f"{i}/{len(urls)}", "url": url, "cards": len(calls)})
                supa.upsert_records("video_jobs", [job])
                supa.upsert_records("stock_calls", calls)
                supa.upsert_records("enriched_calls", enriched, id_key="call_id")
            except Exception as exc:  # noqa: BLE001
                log_event({"failed": url, "error": str(exc)})

            if i % 10 == 0:
                log_event({"checkpoint": f"{i}/{len(urls)}", "jobs_done": len(all_jobs), "calls": len(all_calls)})

    supa.upsert_records("video_jobs", all_jobs)
    supa.upsert_records("stock_calls", all_calls)
    supa.upsert_records("enriched_calls", all_enriched, id_key="call_id")
    log_event(
        {
            "done": True,
            "videos": len(all_jobs),
            "stock_calls": len(all_calls),
            "enriched_calls": len(all_enriched),
            "rows_with_hindi_analyst": sum(1 for row in all_enriched if row.get("analyst")),
            "rows_with_source_x_link": sum(1 for row in all_enriched if row.get("source_url")),
        }
    )


if __name__ == "__main__":
    main()
