"""
Fetch latest TLE/GP data from Space-Track for Idea 1 dataset bootstrap.

Run from the project root:

    cd "H:\\课题\\Intention recognition"
    python ".\\idea 1\\scripts\\fetch_space_track_tle.py"

Default output:

    .\\idea 1\\tle_data\\YYYY-MM-DDTHH-MM-SSZ\\
      manifest.json
      summary.md
      starlink_latest.csv
      starlink_latest.json
      starlink_latest.tle
      china_leo_high_value_candidates_latest.csv
      china_leo_high_value_candidates_latest.json
      china_leo_high_value_candidates_latest.tle

Useful argparse options:

    --output-dir PATH
        Change the base output directory. Each run still creates a timestamped
        subfolder under PATH.

    --china-config PATH
        Use a JSON file with {"object_name_prefixes": [...]} to override the
        built-in China LEO candidate name prefixes.

    --leo-min-mean-motion FLOAT
        Local LEO filter. Default is 11.25 rev/day, roughly period <= 128 min.

    --exclude-regex REGEX
        Local object-name exclusion regex. Default removes debris, rocket
        bodies, adapters, and similar non-payload objects.

    --max-starlink N
        Limit Starlink records returned by Space-Track. Useful for testing.

    --max-per-china-prefix N
        Limit records returned for each China prefix query. Useful for testing.

    --delay-seconds FLOAT
        Delay between Space-Track API calls. Keep this conservative to respect
        Space-Track rate limits.

Example small test:

    python ".\\idea 1\\scripts\\fetch_space_track_tle.py" --max-starlink 20 --max-per-china-prefix 5


"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

BASE_URL = "https://www.space-track.org"

DEFAULT_CHINA_PREFIXES = [
    "TIANHE",
    "WENTIAN",
    "MENGTIAN",
    "TIANGONG",
    "SHENZHOU",
    "TIANZHOU",
    "GAOFEN",
    "GF-",
    "YAOGAN",
    "JILIN",
    "ZIYUAN",
    "ZI YUAN",
    "HAIYANG",
    "HJ-",
    "HUANJING",
    "L-SAR",
    "LUOJIA",
    "QILU",
    "QIANFAN",
    "SPACESAIL",
    "SPACE SAIL",
    "G60",
    "G60 STARLINK",
    "SSST",
]

DEFAULT_EXCLUDE_RE = r"(DEB|R/B|ROCKET BODY|AKM|PLATFORM|ADAPTER)"

CSV_COLUMNS = [
    "group",
    "query_tag",
    "norad_cat_id",
    "object_name",
    "object_id",
    "epoch",
    "mean_motion",
    "eccentricity",
    "inclination",
    "raan",
    "arg_of_pericenter",
    "mean_anomaly",
    "period",
    "apogee",
    "perigee",
    "tle_line1",
    "tle_line2",
]


class SpaceTrackError(RuntimeError):
    """Raised when Space-Track login or query fails."""


@dataclass(frozen=True)
class QueryResult:
    tag: str
    url: str
    records: list[dict[str, Any]]


class SpaceTrackClient:
    def __init__(self, identity: str, password: str, delay_seconds: float = 2.2) -> None:
        self.identity = identity
        self.password = password
        self.delay_seconds = delay_seconds
        self.session = requests.Session()
        self._last_request = 0.0

    def __enter__(self) -> "SpaceTrackClient":
        self.login()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.session.close()

    def login(self) -> None:
        response = self.session.post(
            f"{BASE_URL}/ajaxauth/login",
            data={"identity": self.identity, "password": self.password},
            timeout=60,
        )
        if response.status_code != 200:
            raise SpaceTrackError(
                f"Space-Track login failed: HTTP {response.status_code}"
            )
        text_lower = response.text.lower()
        if (
                "login failed" in text_lower
                or "invalid" in text_lower
                or "account locked" in text_lower
                or "challenge" in text_lower
        ):
            # Space-Track may still return 200 for an HTML login/challenge page.
            text = response.text[:300].replace("\n", " ")
            raise SpaceTrackError(
                "Space-Track login did not reach an authenticated session. "
                f"Response begins with: {text}"
            )

    def query_gp_json(
            self,
            criteria: list[tuple[str, str]],
            *,
            orderby: str | None = None,
            limit: int | None = None,
            tag: str,
    ) -> QueryResult:
        segments = ["basicspacedata", "query", "class", "gp"]
        for key, value in criteria:
            segments.append(key)
            segments.append(value)
        if orderby:
            segments.extend(["orderby", orderby])
        if limit is not None:
            segments.extend(["limit", str(limit)])
        segments.extend(["format", "json", "emptyresult", "show"])
        url = build_url(segments)

        self._respect_rate_limit()
        response = self.session.get(url, timeout=120)
        if response.status_code != 200:
            raise SpaceTrackError(
                f"Space-Track query failed for {tag}: HTTP {response.status_code}; {url}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            text = response.text[:500].replace("\n", " ")
            raise SpaceTrackError(
                f"Space-Track query for {tag} did not return JSON. "
                f"Response begins with: {text}"
            ) from exc

        if isinstance(payload, dict) and payload.get("error"):
            raise SpaceTrackError(f"Space-Track query error for {tag}: {payload}")
        if not isinstance(payload, list):
            raise SpaceTrackError(
                f"Unexpected Space-Track payload for {tag}: {type(payload).__name__}"
            )
        return QueryResult(tag=tag, url=url, records=payload)

    def _respect_rate_limit(self) -> None:
        now = time.monotonic()
        wait = self.delay_seconds - (now - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()


def build_url(path_segments: list[str]) -> str:
    safe_segments = [quote(str(segment), safe=",~-.") for segment in path_segments]
    return f"{BASE_URL}/{'/'.join(safe_segments)}"


def load_china_prefixes(config_path: Path | None) -> list[str]:
    if config_path is None:
        return DEFAULT_CHINA_PREFIXES
    with config_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    prefixes = payload.get("object_name_prefixes")
    if not isinstance(prefixes, list) or not all(isinstance(x, str) for x in prefixes):
        raise ValueError("Config must contain string list: object_name_prefixes")
    return prefixes


def normalize_record(
        raw: dict[str, Any],
        *,
        group: str,
        query_tag: str,
) -> dict[str, Any]:
    line1 = get_first(raw, "TLE_LINE1", "tle_line1")
    line2 = get_first(raw, "TLE_LINE2", "tle_line2")
    mean_motion = get_first(raw, "MEAN_MOTION", "mean_motion")
    if mean_motion in (None, "") and line2:
        mean_motion = parse_mean_motion_from_tle_line2(line2)

    return {
        "group": group,
        "query_tag": query_tag,
        "norad_cat_id": get_first(raw, "NORAD_CAT_ID", "norad_cat_id"),
        "object_name": get_first(raw, "OBJECT_NAME", "object_name"),
        "object_id": get_first(raw, "OBJECT_ID", "object_id"),
        "epoch": get_first(raw, "EPOCH", "epoch"),
        "mean_motion": mean_motion,
        "eccentricity": get_first(raw, "ECCENTRICITY", "eccentricity"),
        "inclination": get_first(raw, "INCLINATION", "inclination"),
        "raan": get_first(raw, "RA_OF_ASC_NODE", "raan", "RAAN"),
        "arg_of_pericenter": get_first(
            raw,
            "ARG_OF_PERICENTER",
            "arg_of_pericenter",
        ),
        "mean_anomaly": get_first(raw, "MEAN_ANOMALY", "mean_anomaly"),
        "period": get_first(raw, "PERIOD", "period"),
        "apogee": get_first(raw, "APOGEE", "apogee"),
        "perigee": get_first(raw, "PERIGEE", "perigee"),
        "tle_line1": line1,
        "tle_line2": line2,
    }


def get_first(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return ""


def parse_mean_motion_from_tle_line2(line2: str) -> float | str:
    try:
        return float(line2[52:63].strip())
    except (TypeError, ValueError):
        return ""


def is_leo(record: dict[str, Any], min_mean_motion: float) -> bool:
    value = record.get("mean_motion", "")
    try:
        return float(value) >= min_mean_motion
    except (TypeError, ValueError):
        return True


def filter_records(
        records: list[dict[str, Any]],
        *,
        min_mean_motion: float,
        exclude_re: re.Pattern[str],
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for record in records:
        name = str(record.get("object_name", ""))
        if exclude_re.search(name):
            continue
        if not is_leo(record, min_mean_motion):
            continue
        filtered.append(record)
    return filtered


def dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_id: dict[str, dict[str, Any]] = {}
    no_id: list[dict[str, Any]] = []
    for record in records:
        norad = str(record.get("norad_cat_id", "")).strip()
        if not norad:
            no_id.append(record)
            continue
        previous = best_by_id.get(norad)
        if previous is None or str(record.get("epoch", "")) > str(previous.get("epoch", "")):
            best_by_id[norad] = record
    return sorted(
        [*best_by_id.values(), *no_id],
        key=lambda item: str(item.get("object_name", "")),
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in CSV_COLUMNS})


def write_tle(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fp:
        for record in records:
            name = str(record.get("object_name", "")).strip()
            line1 = str(record.get("tle_line1", "")).strip()
            line2 = str(record.get("tle_line2", "")).strip()
            if not line1 or not line2:
                continue
            if name:
                fp.write(f"{name}\n")
            fp.write(f"{line1}\n{line2}\n")


def write_summary(
        path: Path,
        *,
        created_at: str,
        starlink: list[dict[str, Any]],
        china: list[dict[str, Any]],
        queries: list[QueryResult],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Space-Track TLE Fetch Summary",
        "",
        f"- Created at UTC: {created_at}",
        f"- Starlink records: {len(starlink)}",
        f"- China LEO high-value candidate records: {len(china)}",
        "",
        "## Epoch Ranges",
        "",
        f"- Starlink: {epoch_range(starlink)}",
        f"- China candidates: {epoch_range(china)}",
        "",
        "## China Candidate Counts by Query Tag",
        "",
    ]
    counts: dict[str, int] = {}
    for record in china:
        tag = str(record.get("query_tag", ""))
        counts[tag] = counts.get(tag, 0) + 1
    if counts:
        for tag, count in sorted(counts.items()):
            lines.append(f"- {tag}: {count}")
    else:
        lines.append("- No records")

    lines.extend(["", "## Query URLs", ""])
    for query in queries:
        lines.append(f"- {query.tag}: {query.url}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def epoch_range(records: list[dict[str, Any]]) -> str:
    epochs = sorted(
        str(record.get("epoch", "")).strip()
        for record in records
        if str(record.get("epoch", "")).strip()
    )
    if not epochs:
        return "N/A"
    return f"{epochs[0]} to {epochs[-1]}"


def fetch_starlink(
        client: SpaceTrackClient,
        *,
        min_mean_motion: float,
        exclude_re: re.Pattern[str],
        max_records: int | None,
) -> tuple[list[dict[str, Any]], QueryResult]:
    criteria = [
        ("OBJECT_NAME", "STARLINK~~"),
        ("DECAY_DATE", "null-val"),
    ]
    result = client.query_gp_json(
        criteria,
        orderby="NORAD_CAT_ID desc",
        limit=max_records,
        tag="starlink",
    )
    records = [
        normalize_record(raw, group="starlink", query_tag="STARLINK")
        for raw in result.records
    ]
    records = filter_records(
        records,
        min_mean_motion=min_mean_motion,
        exclude_re=exclude_re,
    )
    return dedupe_records(records), result


def fetch_china_candidates(
        client: SpaceTrackClient,
        *,
        prefixes: list[str],
        min_mean_motion: float,
        exclude_re: re.Pattern[str],
        max_per_prefix: int | None,
) -> tuple[list[dict[str, Any]], list[QueryResult]]:
    all_records: list[dict[str, Any]] = []
    queries: list[QueryResult] = []
    for prefix in prefixes:
        tag = f"china_prefix_{prefix}"
        criteria = [
            ("OBJECT_NAME", f"{prefix}~~"),
            ("DECAY_DATE", "null-val"),
        ]
        result = client.query_gp_json(
            criteria,
            orderby="NORAD_CAT_ID desc",
            limit=max_per_prefix,
            tag=tag,
        )
        queries.append(result)
        for raw in result.records:
            all_records.append(
                normalize_record(
                    raw,
                    group="china_leo_high_value_candidate",
                    query_tag=prefix,
                )
            )
    filtered = filter_records(
        all_records,
        min_mean_motion=min_mean_motion,
        exclude_re=exclude_re,
    )
    return dedupe_records(filtered), queries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch current Starlink and China LEO target TLE data from Space-Track.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="H:\\课题\\Intention recognition\\idea 1\\tle_data",
        help="Directory where outputs are written.",
    )
    parser.add_argument(
        "--china-config",
        type=Path,
        default=None,
        help="Optional JSON config with object_name_prefixes.",
    )
    parser.add_argument(
        "--leo-min-mean-motion",
        type=float,
        default=11.25,
        help="Local LEO filter. Mean motion >= 11.25 rev/day is roughly period <= 128 min.",
    )
    parser.add_argument(
        "--exclude-regex",
        default=DEFAULT_EXCLUDE_RE,
        help="Regex for local exclusion by object name.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=2.2,
        help="Delay between Space-Track API requests.",
    )
    parser.add_argument(
        "--max-starlink",
        type=int,
        default=None,
        help="Optional Space-Track limit for Starlink records.",
    )
    parser.add_argument(
        "--max-per-china-prefix",
        type=int,
        default=None,
        help="Optional Space-Track limit for each China prefix query.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    run_dir = args.output_dir / created_at
    run_dir.mkdir(parents=True, exist_ok=True)

    identity = "laplace3226@gmail.com"
    password = "W.F.Laplace3226_0"
    china_prefixes = load_china_prefixes(args.china_config)
    exclude_re = re.compile(args.exclude_regex, flags=re.IGNORECASE)

    all_queries: list[QueryResult] = []
    with SpaceTrackClient(
            identity=identity,
            password=password,
            delay_seconds=args.delay_seconds,
    ) as client:
        starlink, starlink_query = fetch_starlink(
            client,
            min_mean_motion=args.leo_min_mean_motion,
            exclude_re=exclude_re,
            max_records=args.max_starlink,
        )
        all_queries.append(starlink_query)

        china, china_queries = fetch_china_candidates(
            client,
            prefixes=china_prefixes,
            min_mean_motion=args.leo_min_mean_motion,
            exclude_re=exclude_re,
            max_per_prefix=args.max_per_china_prefix,
        )
        all_queries.extend(china_queries)

    write_json(run_dir / "starlink_latest.json", starlink)
    write_csv(run_dir / "starlink_latest.csv", starlink)
    write_tle(run_dir / "starlink_latest.tle", starlink)

    write_json(run_dir / "china_leo_high_value_candidates_latest.json", china)
    write_csv(run_dir / "china_leo_high_value_candidates_latest.csv", china)
    write_tle(run_dir / "china_leo_high_value_candidates_latest.tle", china)

    manifest = {
        "created_at_utc": created_at,
        "space_track_base_url": BASE_URL,
        "china_prefixes": china_prefixes,
        "leo_min_mean_motion": args.leo_min_mean_motion,
        "exclude_regex": args.exclude_regex,
        "queries": [{"tag": query.tag, "url": query.url} for query in all_queries],
        "outputs": {
            "starlink_records": len(starlink),
            "china_candidate_records": len(china),
        },
    }
    write_json(run_dir / "manifest.json", manifest)
    write_summary(
        run_dir / "summary.md",
        created_at=created_at,
        starlink=starlink,
        china=china,
        queries=all_queries,
    )

    print(f"Wrote Space-Track data to: {run_dir}")
    print(f"Starlink records: {len(starlink)}")
    print(f"China LEO high-value candidate records: {len(china)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SpaceTrackError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
