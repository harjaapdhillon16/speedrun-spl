# RunPod SPL Midcap Speedrun

## Accurate re-OCR → `stock_analysis_calls` (EasyOCR, recommended)

The original Tesseract whole-frame parser was inaccurate (wrong company on most
cards, targets polluted with duration/quote numbers, garbled Hindi analyst
names). `ocr_stock_analysis_fill.py` re-reads the card frames **already in
Supabase Storage** with EasyOCR (Devanagari + English) + a layout-aware parser,
market-enriches via the same `enrich_card`, and fills the dedicated
`stock_analysis_calls` table. No video download / ffmpeg needed — it's pure
image→text, so it's fast and fully parallel across CPUs.

Prereq: create the table once in the Supabase SQL editor using
`stock_analysis_calls_schema.sql` (repo root).

```bash
export url='https://YOUR_PROJECT.supabase.co'
export secret_key='YOUR_SUPABASE_SERVICE_ROLE_KEY'

bash run_ocr_fill.sh                # background, uses all CPUs
# bash run_ocr_fill.sh --workers 24 # or pin the worker count
tail -f ./ocr_fill.log
# stop: kill "$(cat ./ocr_fill.pid)"
```

The first launch pip-installs `easyocr`+`torch` (large) and downloads the model
weights, so it takes a few minutes before rows start appearing. The run is
idempotent (row id keyed on source_url+stock+entry_date), so re-running fills
gaps and corrects rows in place. Progress prints as JSON lines
(`{"progress": "...", "rows": N}`), ending with `{"DONE": true, ...}`.

To process just a few jobs as a test: `python ocr_stock_analysis_fill.py --workers 4 --limit 20`.

---

## Original video pipeline

Use **Pods** in RunPod.

Do not use Serverless, Public endpoints, or Clusters for this job. This is a one-off batch job that downloads videos, runs ffmpeg/OpenCV/Tesseract OCR, uploads frames, and writes rows to Supabase.

## Recommended Pod

- Type: **Pods**
- Your selected option: **Runpod Ubuntu 20.04, Compute-Optimized, 32 vCPU, 64 GB RAM**
- Concurrency for this pod: **16**
- CPU frequency: choose **5 GHz**
- Network volume: okay
- Disk: **300-500 GB is enough** when source videos are cleaned up. 3000 GB works but costs more than needed.
- Template/Image: Runpod Ubuntu 20.04 is fine

This workload is mostly CPU/Tesseract/ffmpeg/network, so the 32 vCPU CPU pod is better value than the H200 GPU pod.

Source videos are downloaded only to the pod's local disk for OCR. The script uploads only extracted call-card PNG frames to Supabase Storage, then deletes the local source video by default.

## Files

- `spl_midcap_speedrun.py` - standalone Python runner, no project imports
- `runpod_bootstrap.sh` - installs system packages and runs the runner
- `run_background.sh` - starts the runner in the background and writes logs/pid in this directory
- `.env.example` - env vars to set manually in the RunPod terminal

## Run

Upload this whole folder to RunPod, then in the folder:

```bash
export url='https://YOUR_PROJECT.supabase.co'
export secret_key='YOUR_SUPABASE_SERVICE_ROLE_KEY'

bash runpod_bootstrap.sh --start 2023-04-01
```

For a test month:

```bash
bash runpod_bootstrap.sh --start 2024-02-01 --end 2024-02-29 --concurrency 8
```

The runner is resumable by default. It preserves `enriched_calls`, removes duplicates, and skips videos whose source URL already exists in `enriched_calls`. Use `--clear-enriched` only when you intentionally want a destructive rebuild.

## If Nitter Is Blocked

If the run prints `403 Forbidden` for every Nitter instance and then `videos: 0`, the pod is fine but Nitter discovery is blocked from that RunPod IP.

Fast fallback: use ZeeBusiness video URLs already present in Supabase `video_jobs` and skip Nitter:

```bash
git pull
source .env
bash runpod_bootstrap.sh --existing-video-jobs --skip-discovery --start 2023-04-01 --concurrency 16
```

Run the same resumable job in the background:

```bash
git pull
source .env
bash ./run_background.sh
tail -f ./spl_midcap_run.log
```

Stop a background run:

```bash
kill "$(cat ./spl_midcap_run.pid)"
```

Manual fallback: put one ZeeBusiness status/video URL per line in a file:

```bash
cat > /workspace/zee_urls.txt <<'EOF'
https://x.com/ZeeBusiness/status/1234567890/video/1
EOF

source .env
bash runpod_bootstrap.sh --urls-file /workspace/zee_urls.txt --skip-discovery --concurrency 16
```

These fallback modes still download the X videos, OCR frames, upload images, and write Supabase rows.

While running, progress is printed as JSON lines. You should see `download_start`, `download`, `ocr_start`, frame counters, `merge`, and `processed` messages. The red Python 3.8 deprecation warnings from `yt-dlp` are not job failures.

## Speed Tuning

By default the runner now does a sparse soft scan first. It checks one frame every 5 seconds with cheap English OCR, then runs full Hindi/name/price extraction only near matching stock-call card frames.

Recommended full run on the 32 vCPU pod:

```bash
source .env
bash runpod_bootstrap.sh --existing-video-jobs --skip-discovery --start 2023-04-01 --concurrency 16
```

Faster but slightly riskier:

```bash
bash runpod_bootstrap.sh --existing-video-jobs --skip-discovery --start 2023-04-01 --concurrency 20 --soft-scan-fps 0.1 --deep-window 5 --deep-step 2
```

Safer but slower:

```bash
bash runpod_bootstrap.sh --existing-video-jobs --skip-discovery --start 2023-04-01 --concurrency 12 --soft-scan-fps 0.5 --deep-window 8 --deep-step 1
```

Fallback to the old full-scan behavior:

```bash
bash runpod_bootstrap.sh --existing-video-jobs --skip-discovery --no-soft-scan --frame-fps 1
```

Useful resume controls:

- `--skip-processed` is on by default; it skips any video URL already present in `enriched_calls.source_url`.
- `--cleanup-duplicates` is on by default; it deletes duplicate `enriched_calls` and duplicate `video_jobs` rows before queueing.
- `--clear-enriched` is destructive and should only be used for a full rebuild.

## IP Rotation

Nitter can block or rate-limit a pod IP. The script supports proxy rotation for Nitter discovery and yt-dlp/X video downloads.

Use proxies you control:

```bash
export PROXY_LIST='http://user:pass@proxy1:port,http://user:pass@proxy2:port'
bash runpod_bootstrap.sh --start 2023-04-01
```

Or put one proxy per line:

```bash
export PROXY_FILE='/workspace/proxies.txt'
bash runpod_bootstrap.sh --start 2023-04-01
```

Without proxies, the script still rotates Nitter mirrors and throttles Nitter pagination, but that is not true IP rotation.

## Output

The script writes to Supabase `app_records` collections:

- `video_jobs`
- `stock_calls`
- `enriched_calls`

It uploads card images to the Supabase Storage bucket:

- `stock-call-media/video_frames/...`

Analyst names are stored in Hindi as extracted from the card image.
