"""
data_fetch.py
--------------
Pulls REAL vulnerability records from the NVD (National Vulnerability
Database) REST API v2.0 -- https://services.nvd.nist.gov/rest/json/cves/2.0
-- which is U.S. government public-domain data (NIST), free of copyright
and PII concerns, and therefore safe to redistribute on GitHub alongside
this code.

For every CVE published in the configured date window, we keep:
  - cve_id
  - the English-language textual description (model INPUT)
  - the primary CVSS v3.x base score and NVD's own `baseSeverity` label
    (LOW / MEDIUM / HIGH / CRITICAL, as officially defined by the CVSS
    v3.1 specification -- FIRST.org) -- this becomes the classification
    LABEL.
  - published date

CVEs that have no CVSS v3.x metric at all (a small minority, mostly very
old or not-yet-scored entries) are dropped; this is a transparent,
documented inclusion criterion, not a fabrication of any kind -- every
remaining row is a real NVD record with a real NVD-assigned severity.

Usage
-----
    python -m ciphertune.data_fetch

Optional: set the environment variable NVD_API_KEY to raise the NVD
rate limit from 5 requests/30s to 50 requests/30s (free, instant sign-up
at https://nvd.nist.gov/developers/request-an-api-key). The script works
without a key, just more slowly.
"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from . import config as C

NVD_ENDPOINT = "https://services.nvd.nist.gov/rest/json/cves/2.0"


def _request_page(pub_start: str, pub_end: str, start_index: int, api_key: str = None,
                   max_retries: int = 5, initial_backoff: float = 10.0) -> dict:
    params = {
        "pubStartDate": pub_start,
        "pubEndDate": pub_end,
        "resultsPerPage": C.RESULTS_PER_PAGE,
        "startIndex": start_index,
    }
    headers = {"apiKey": api_key} if api_key else {}

    for attempt in range(1, max_retries + 1):
        resp = requests.get(NVD_ENDPOINT, params=params, headers=headers, timeout=60)
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
            wait = initial_backoff * (2 ** (attempt - 1))
            print(f"[data_fetch]   HTTP {resp.status_code} on attempt {attempt}/{max_retries}; "
                  f"retrying in {wait:.0f}s …")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()


def _extract_primary_cvss_v3(cve_obj: dict):
    metrics = cve_obj.get("metrics", {})
    for key in ("cvssMetricV31", "cvssMetricV30"):
        entries = metrics.get(key, [])
        primary = next((e for e in entries if e.get("type") == "Primary"), None)
        chosen = primary or (entries[0] if entries else None)
        if chosen is not None:
            data = chosen["cvssData"]
            return float(data["baseScore"]), data.get("baseSeverity", "").upper(), data["version"]
    return None, None, None


def _extract_english_description(cve_obj: dict):
    for d in cve_obj.get("descriptions", []):
        if d.get("lang") == "en" and d.get("value"):
            return d["value"]
    return None


def _date_windows(start_str: str, end_str: str, max_days: int):
    """
    Splits [start_str, end_str] into a sequence of (window_start, window_end)
    string pairs, each spanning at most `max_days` days, in NVD's required
    "yyyy-MM-ddTHH:mm:ss.SSS" format. Required because the NVD API rejects
    (with an HTTP 404, not a 400) any pubStartDate/pubEndDate pair spanning
    more than 120 consecutive days.
    """
    fmt = "%Y-%m-%dT%H:%M:%S.%f"
    start_dt = datetime.strptime(start_str, fmt)
    end_dt = datetime.strptime(end_str, fmt)
    if start_dt >= end_dt:
        raise ValueError("NVD_PUBLISHED_START must be earlier than NVD_PUBLISHED_END")

    windows = []
    cur = start_dt
    step = timedelta(days=max_days)
    while cur < end_dt:
        window_end = min(cur + step, end_dt)
        windows.append((cur.strftime(fmt)[:-3], window_end.strftime(fmt)[:-3]))
        cur = window_end
    return windows


def fetch_all_raw(out_dir: str = C.RAW_JSON_DIR, api_key: str = None):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    windows = _date_windows(C.NVD_PUBLISHED_START, C.NVD_PUBLISHED_END, C.NVD_MAX_WINDOW_DAYS)
    print(f"[data_fetch] full range split into {len(windows)} window(s) of <= "
          f"{C.NVD_MAX_WINDOW_DAYS} days each (NVD's hard 120-day limit per query).")

    page_num = 0
    total_fetched = 0
    for w_idx, (w_start, w_end) in enumerate(windows):
        print(f"[data_fetch] window {w_idx + 1}/{len(windows)}: {w_start} -> {w_end}")
        start_index = 0
        total_results = None

        while total_results is None or start_index < total_results:
            page = _request_page(w_start, w_end, start_index, api_key=api_key)
            total_results = page["totalResults"]
            out_path = Path(out_dir) / f"page_{page_num:04d}.json"
            with open(out_path, "w") as f:
                json.dump(page, f)
            n_in_page = len(page.get("vulnerabilities", []))
            total_fetched += n_in_page
            print(f"[data_fetch]   page {page_num}: got {n_in_page} records "
                  f"(startIndex={start_index}, totalResults={total_results})")
            start_index += C.RESULTS_PER_PAGE
            page_num += 1

            if total_fetched >= C.TARGET_DATASET_SIZE * 3:
                print("[data_fetch] reached the configured fetch cap; stopping early "
                      "(remove this guard in data_fetch.py for the full date range).")
                return page_num

            # NVD rate limit: 5 req/30s (no key) or 50 req/30s (with key)
            time.sleep(0.6 if api_key else 6.5)

    return page_num


def build_processed_csv(raw_dir: str = C.RAW_JSON_DIR, out_csv: str = C.PROCESSED_CSV):
    rows = []
    for path in sorted(Path(raw_dir).glob("page_*.json")):
        with open(path) as f:
            page = json.load(f)
        for item in page.get("vulnerabilities", []):
            cve = item["cve"]
            score, severity, cvss_version = _extract_primary_cvss_v3(cve)
            if score is None or severity not in C.SEVERITY_LABELS:
                continue
            description = _extract_english_description(cve)
            if not description or len(description.strip()) < 15:
                continue
            rows.append({
                "cve_id": cve["id"],
                "description": description.strip(),
                "cvss_score": score,
                "cvss_version": cvss_version,
                "severity_label": severity,
                "published_date": cve.get("published", "")[:10],
            })

    df = pd.DataFrame(rows).drop_duplicates(subset="cve_id").reset_index(drop=True)

    if len(df) > C.TARGET_DATASET_SIZE:
        df = df.sample(n=C.TARGET_DATASET_SIZE, random_state=C.SEED).reset_index(drop=True)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    # token-length sanity check used to justify config.MAX_TOKEN_LENGTH
    word_counts = df["description"].str.split().apply(len)
    print(f"[data_fetch] wrote {len(df)} labeled records to {out_csv}")
    print(f"[data_fetch] class balance:\n{df['severity_label'].value_counts()}")
    print(f"[data_fetch] description word-count stats: "
          f"mean={word_counts.mean():.1f}, median={word_counts.median():.0f}, "
          f"p95={word_counts.quantile(0.95):.0f}, max={word_counts.max()}")
    return df


if __name__ == "__main__":
    api_key = os.environ.get("NVD_API_KEY")
    if api_key:
        print("[data_fetch] using NVD_API_KEY from environment (higher rate limit).")
    else:
        print("[data_fetch] no NVD_API_KEY set; using the slower unauthenticated rate limit. "
              "This is fine, just slower (a few minutes).")
    fetch_all_raw(api_key=api_key)
    build_processed_csv()
