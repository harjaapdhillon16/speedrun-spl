# RunPod SPL Midcap Speedrun

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

## Files

- `spl_midcap_speedrun.py` - standalone Python runner, no project imports
- `runpod_bootstrap.sh` - installs system packages and runs the runner
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

The script clears `enriched_calls` by default. Add `--no-clear-enriched` if you do not want that.

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
