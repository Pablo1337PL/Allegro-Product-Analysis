"""Full overnight pipeline: scrape -> classify -> detail -> process -> parsed_offers -> ml.

Usage:
    python run_pipeline.py                 # full run
    python run_pipeline.py --skip-scrape    # skip listing scrape (reuse existing offers)
    python run_pipeline.py --skip-details   # skip detail scrape
    python run_pipeline.py --skip-ml        # skip parsed_offers build + similarity/clustering
    python run_pipeline.py --max-pages 5    # cap listing scrape for testing
    python run_pipeline.py --continue-on-error  # keep going past a failed step
"""
import argparse
import logging
import os
import subprocess
import sys
import time

import requests
from decouple import config

from scraper.ollama_common import OLLAMA_MODEL, OLLAMA_URL

log = logging.getLogger(__name__)

PYTHON = sys.executable  # conda automl python

# The listing-scrape step is handled specially (see _run_listing_step) — a
# scraper/thumbnail_embedder.py --watch process runs alongside scraper/scheduler.py,
# embedding each page's thumbnails as they're written instead of waiting for the
# whole listing scrape to finish.
STEPS = [
    {
        "name": "Listing scraper (+ parallel thumbnail CLIP embedding)",
        "cmd": [PYTHON, "scraper/scheduler.py"],
        "skip_flag": "skip_scrape",
        "parallel_with": [PYTHON, "scraper/thumbnail_embedder.py", "--watch"],
    },
    {
        "name": "is_bike classification (KMeans k=5 on thumbnail CLIP embeddings)",
        "cmd": [PYTHON, "scraper/bike_classifier.py"],
        "skip_flag": None,  # always run — detail scrape below is bike-only
    },
    {
        "name": "Detail scraper (with Ollama spec extraction)",
        "cmd": [PYTHON, "scraper/detail_scheduler.py"],
        "skip_flag": "skip_details",
    },
    {
        "name": "Feature processor (CLIP, zero-shot, condition, segmentation)",
        "cmd": [PYTHON, "processor/pipeline.py"],
        "skip_flag": None,  # always run
    },
    {
        "name": "Build parsed_offers table",
        "cmd": [PYTHON, "scraper/parsed_offers_builder.py"],
        "skip_flag": "skip_ml",
    },
    {
        "name": "Build similarity index + clusters",
        "cmd": [PYTHON, "ml/build_index.py"],
        "skip_flag": "skip_ml",
    },
]


def _extra_args(args: argparse.Namespace, step: dict) -> list[str]:
    if step["cmd"][-1] == "scraper/scheduler.py" and args.max_pages:
        return ["--max-pages", str(args.max_pages)]
    return []


def preflight_checks() -> None:
    """Abort early if something is obviously broken."""
    tags_url = OLLAMA_URL.rsplit("/api/", 1)[0] + "/api/tags"

    try:
        resp = requests.get(tags_url, timeout=5)
    except requests.ConnectionError:
        raise SystemExit("ERROR: Ollama is not running. Start with: ollama serve")

    models = [m["name"] for m in resp.json().get("models", [])]
    if OLLAMA_MODEL not in models and f"{OLLAMA_MODEL}:latest" not in models:
        raise SystemExit(
            f"ERROR: Model {OLLAMA_MODEL} not pulled. Run: ollama pull {OLLAMA_MODEL}"
        )

    db_path = config("DB_PATH", default="db/bikes.db")
    if os.path.exists(db_path) and not os.access(db_path, os.W_OK):
        raise SystemExit(f"ERROR: {db_path} is not writable (check ownership)")


def _run_step_with_parallel(cmd: list[str], parallel_cmd: list[str]) -> int:
    """Run cmd to completion while parallel_cmd runs alongside it; stop
    parallel_cmd (SIGTERM, giving it time to finish its current batch and do a
    final drain) once cmd finishes."""
    log.info("Starting parallel process: %s", " ".join(parallel_cmd))
    parallel_proc = subprocess.Popen(parallel_cmd)
    try:
        result = subprocess.run(cmd, check=False)
    finally:
        parallel_proc.terminate()
        try:
            parallel_proc.wait(timeout=600)
        except subprocess.TimeoutExpired:
            log.warning("Parallel process did not exit after SIGTERM — killing it")
            parallel_proc.kill()
            parallel_proc.wait()
    return result.returncode


def run_pipeline(args: argparse.Namespace) -> None:
    start = time.time()
    for step in STEPS:
        if step["skip_flag"] and getattr(args, step["skip_flag"], False):
            log.info("Skipping: %s", step["name"])
            continue
        log.info("Starting: %s", step["name"])
        cmd = step["cmd"] + _extra_args(args, step)
        if step.get("parallel_with"):
            returncode = _run_step_with_parallel(cmd, step["parallel_with"])
        else:
            returncode = subprocess.run(cmd, check=False).returncode
        if returncode != 0:
            log.error("Step failed: %s (exit %d)", step["name"], returncode)
            if not args.continue_on_error:
                sys.exit(returncode)
    elapsed = time.time() - start
    log.info("Pipeline complete in %.1f minutes", elapsed / 60)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(description="Run the full overnight pipeline")
    parser.add_argument("--skip-scrape", dest="skip_scrape", action="store_true",
                        help="Skip listing scrape (reuse existing offers)")
    parser.add_argument("--skip-details", dest="skip_details", action="store_true",
                        help="Skip detail scrape")
    parser.add_argument("--skip-ml", dest="skip_ml", action="store_true",
                        help="Skip parsed_offers build + similarity/clustering")
    parser.add_argument("--max-pages", type=int, default=None,
                        help="Cap listing scrape page count (testing)")
    parser.add_argument("--continue-on-error", dest="continue_on_error", action="store_true",
                        help="Keep running subsequent steps after a step fails")
    parser.add_argument("--skip-preflight", dest="skip_preflight", action="store_true",
                        help="Skip Ollama/DB preflight checks")
    args = parser.parse_args()

    if not args.skip_preflight:
        preflight_checks()

    run_pipeline(args)


if __name__ == "__main__":
    main()
