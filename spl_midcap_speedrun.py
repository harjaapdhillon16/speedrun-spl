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
  - downloads videos with yt-dlp
  - samples frames with OpenCV
  - OCRs stock call cards with Tesseract
  - keeps the representative frame that contains the Hindi analyst line
  - stores analyst name in Hindi on the individual stock call row
  - uploads call-card PNGs to Supabase Storage
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
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote, urljoin


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
        print({"installing_python_packages": missing}, flush=True)
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
    clear_enriched: bool
    cleanup_videos: bool
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

    def list_records(self, collection: str) -> list[dict]:
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
                    "select": "payload",
                    "collection": f"eq.{collection}",
                    "order": "updated_at.asc",
                },
            )
            r.raise_for_status()
            batch = r.json()
            rows.extend(item.get("payload") or {} for item in batch)
            if len(batch) < page_size:
                break
        return rows

    def delete_collection(self, collection: str) -> None:
        r = self.client.delete(
            self.table_url(),
            headers=self.headers(),
            params={"collection": f"eq.{collection}"},
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
                print({"nitter_error": instance, "proxy": bool(proxy), "error": str(exc)}, flush=True)
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
    for i in range(84, 116):
        y = i / 200
        text = ocr(frame, PANEL_X[0], PANEL_X[1], y, y + 0.04, psm=6, fx=3.0).lower()
        if "target" in text and ("stop" in text or "loss" in text or "duration" in text):
            return y
    return None


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


def download_video(url: str, outdir: Path, proxy: str | None = None) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    opts = {
        "outtmpl": str(outdir / "%(id)s.%(ext)s"),
        "format": "bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
    }
    if proxy:
        opts["proxy"] = proxy
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
    video_path = download_video(url, job_dir, proxy=proxy)
    records = []
    done = 0
    for ts, frame in iter_sampled_frames(video_path, cfg.frame_fps, cfg.max_frames):
        done += 1
        parsed = extract_card(frame)
        if parsed and valid_stock_name(parsed.get("stock")) and parsed.get("targets"):
            parsed["_ts"] = ts
            records.append(parsed)
        if done % 50 == 0:
            print({"job": job_id, "frames": done, "candidate_reads": len(records)}, flush=True)

    merged = merge_cards(records, cfg.min_card_frames)
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
        "frames_total": done,
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
    p.add_argument("--frame-fps", type=float, default=1.0)
    p.add_argument("--max-frames", type=int, default=1200)
    p.add_argument("--min-card-frames", type=int, default=2)
    p.add_argument("--bucket", default=os.getenv("SUPABASE_MEDIA_BUCKET", "stock-call-media"))
    p.add_argument("--table", default=os.getenv("SUPABASE_DATA_TABLE", "app_records"))
    p.add_argument("--nitter", default=",".join(DEFAULT_NITTER))
    p.add_argument("--proxy-list", default=os.getenv("PROXY_LIST", ""), help="Comma-separated http/socks proxies for Nitter and yt-dlp")
    p.add_argument("--proxy-file", default=os.getenv("PROXY_FILE", ""), help="File with one proxy per line")
    p.add_argument("--workdir", default="./spl_speedrun_work")
    p.add_argument("--no-clear-enriched", action="store_true")
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
        clear_enriched=not args.no_clear_enriched,
        cleanup_videos=not args.keep_source_videos,
        workdir=Path(args.workdir),
    )


def main() -> None:
    cfg = parse_args()
    cfg.workdir.mkdir(parents=True, exist_ok=True)
    supa = Supabase(cfg)
    print(
        {
            "start": cfg.start.isoformat(),
            "end": cfg.end.isoformat(),
            "concurrency": cfg.concurrency,
            "proxies": len(cfg.proxies),
            "nitter_instances": len(cfg.nitter_instances),
        },
        flush=True,
    )
    if cfg.clear_enriched:
        print("[clear] enriched_calls", flush=True)
        supa.delete_collection("enriched_calls")

    urls = []
    seen = set()
    windows = month_windows(cfg.start, cfg.end)
    for idx, (lo, hi) in enumerate(windows, 1):
        found = discover_month(cfg, lo, hi)
        for url in found:
            if url not in seen:
                seen.add(url)
                urls.append(url)
        print({"discover": f"{idx}/{len(windows)}", "month": f"{lo}..{hi}", "found": len(found), "total_urls": len(urls)}, flush=True)

    all_jobs, all_calls, all_enriched = [], [], []
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=cfg.concurrency) as pool:
        futures = {pool.submit(process_video, cfg, supa, url): url for url in urls}
        for i, fut in enumerate(as_completed(futures), 1):
            url = futures[fut]
            try:
                job, calls, enriched = fut.result()
                all_jobs.append(job)
                all_calls.extend(calls)
                all_enriched.extend(enriched)
                print({"processed": f"{i}/{len(urls)}", "url": url, "cards": len(calls)}, flush=True)
            except Exception as exc:  # noqa: BLE001
                print({"failed": url, "error": str(exc)}, flush=True)

            if i % 10 == 0:
                supa.upsert_records("video_jobs", all_jobs)
                supa.upsert_records("stock_calls", all_calls)
                supa.upsert_records("enriched_calls", all_enriched, id_key="call_id")

    supa.upsert_records("video_jobs", all_jobs)
    supa.upsert_records("stock_calls", all_calls)
    supa.upsert_records("enriched_calls", all_enriched, id_key="call_id")
    print(
        {
            "done": True,
            "videos": len(all_jobs),
            "stock_calls": len(all_calls),
            "enriched_calls": len(all_enriched),
            "rows_with_hindi_analyst": sum(1 for row in all_enriched if row.get("analyst")),
            "rows_with_source_x_link": sum(1 for row in all_enriched if row.get("source_url")),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
