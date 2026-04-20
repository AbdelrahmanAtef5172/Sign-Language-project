"""
Sign Language Dataset — URL Validator (No Downloads)
=====================================================
Checks every URL in urls.csv and saves results.
Nothing is downloaded — only a small CSV is created.

RESUME FEATURE: If the script crashes or stops, just run it again.
It will automatically skip URLs already checked and continue from where it stopped.

Output
------
metadata/
  valid_urls.txt     ← URLs that work
  failed_urls.txt    ← URLs that are broken (with reason)
  validation_log.csv ← full results for every URL

Usage
-----
    pip install requests tqdm
    python validate_urls.py

    # Test on first 50 URLs:
    python validate_urls.py --limit 50
"""

import csv
import time
import argparse
import requests
from pathlib import Path
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────

CSV_PATH    = "urls.csv"
META_DIR    = Path("metadata")
VALID_FILE  = META_DIR / "valid_urls.txt"
FAILED_FILE = META_DIR / "failed_urls.txt"
LOG_FILE    = META_DIR / "validation_log.csv"

TIMEOUT = 10
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ASL-Validator/1.0)"}

# ── Setup ─────────────────────────────────────────────────────────────────────

META_DIR.mkdir(parents=True, exist_ok=True)

# ── Load already-checked URLs from log (resume feature) ───────────────────────

def load_already_checked() -> set:
    """Read validation_log.csv and return set of URLs already processed."""
    checked = set()
    if LOG_FILE.exists():
        with open(LOG_FILE, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("url"):
                    checked.add(row["url"].strip())
    return checked

# ── Validate one URL ──────────────────────────────────────────────────────────

def validate(url: str) -> tuple[bool, str]:
    """Send a HEAD request. Returns (is_valid, reason)."""
    try:
        r = requests.head(url, timeout=TIMEOUT, headers=HEADERS, allow_redirects=True)
        if r.status_code < 400:
            return True, f"HTTP {r.status_code}"
        return False, f"HTTP {r.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Connection error"
    except requests.exceptions.Timeout:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)

# ── Main ──────────────────────────────────────────────────────────────────────

def run(limit=None):
    # Load all URLs
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_urls = [row["url"].strip() for row in reader if row.get("url", "").strip()]

    if limit:
        all_urls = all_urls[:limit]

    # Resume: find URLs not yet checked
    already_checked = load_already_checked()
    remaining = [u for u in all_urls if u not in already_checked]

    total    = len(all_urls)
    skipped  = len(already_checked)
    to_check = len(remaining)

    print(f"Total URLs      : {total}")
    print(f"Already checked : {skipped}  <- resuming from here")
    print(f"Remaining       : {to_check}")
    print()

    if to_check == 0:
        print("All URLs already validated! Check metadata/ for results.")
        return

    stats = {"valid": 0, "failed": 0}

    # Append to existing files (don't overwrite previous results)
    log_exists = LOG_FILE.exists() and LOG_FILE.stat().st_size > 0

    with open(VALID_FILE,  "a", encoding="utf-8") as vf, \
         open(FAILED_FILE, "a", encoding="utf-8") as ff, \
         open(LOG_FILE,    "a", newline="", encoding="utf-8") as lf:

        log = csv.writer(lf)
        if not log_exists:
            log.writerow(["url", "status", "reason", "timestamp"])

        for url in tqdm(remaining, unit="url", desc="Validating"):
            is_valid, reason = validate(url)
            ts = time.strftime("%Y-%m-%dT%H:%M:%S")

            if is_valid:
                vf.write(url + "\n")
                vf.flush()
                stats["valid"] += 1
            else:
                ff.write(f"{url}\t{reason}\n")
                ff.flush()
                stats["failed"] += 1

            log.writerow([url, "valid" if is_valid else "failed", reason, ts])
            lf.flush()

    print(f"\nDone!")
    print(f"  Valid   : {stats['valid']}")
    print(f"  Failed  : {stats['failed']}")
    print(f"\nResults saved in metadata/")

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Only check first N URLs (for testing)")
    args = parser.parse_args()
    run(limit=args.limit)