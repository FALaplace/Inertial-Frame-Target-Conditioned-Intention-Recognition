from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EARTH_RADIUS_KM = 6378.137
REGION_RADIUS_MARGIN_KM = 120.0
ACCESS_ELEVATION_THRESHOLD_DEG = 10.0

Vector3 = tuple[float, float, float]

FEATURE_NAMES = [
    "time_offset_s",
    "distance_km",
    "relative_speed_km_s",
    "range_rate_km_s",
    "closing_speed_km_s",
    "actor_radius_km",
    "counterpart_radius_km",
    "altitude_gap_km",
    "plane_angle_deg",
    "surface_distance_km",
    "elevation_deg",
    "access_flag",
    "counterpart_is_region",
    "time_since_maneuver_s",
    "maneuver_flag",
    "matching_score",
    "orbit_position_sigma",
    "orbit_velocity_sigma",
    "maneuver_time_sigma",
]


def load_samples(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, list):
        raise ValueError("Sample file must contain a JSON list.")
    return payload


def build_feature_dataset(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [build_feature_sample(sample, index) for index, sample in enumerate(samples)]


def build_feature_sample(sample: dict[str, Any], sample_index: int = 0) -> dict[str, Any]:
    actor_trajectory = sample["actor_trajectory"]
    counterpart_trajectory = sample["counterpart_trajectory"]
    if len(actor_trajectory) != len(counterpart_trajectory):
        raise ValueError(f"Sample {sample_index} has mismatched trajectory lengths.")
    if not actor_trajectory:
        raise ValueError(f"Sample {sample_index} has empty trajectories.")

    matching_score = float(sample.get("matching_score", 0.0))
    error_summary = sample.get("error_summary", {}) or {}
    maneuver_times = parse_maneuver_times(sample.get("maneuver_events", []))
    counterpart_is_region = infer_counterpart_is_region(counterpart_trajectory)
    start_time = parse_time(actor_trajectory[0]["t"])

    sequence = []
    for actor_point, counterpart_point in zip(actor_trajectory, counterpart_trajectory):
        t = parse_time(actor_point["t"])
        actor_r = vector3(actor_point["r"])
        actor_v = vector3(actor_point["v"])
        counterpart_r = vector3(counterpart_point["r"])
        counterpart_v = vector3(counterpart_point["v"])

        relative_r = vector_sub(actor_r, counterpart_r)
        relative_v = vector_sub(actor_v, counterpart_v)
        distance = vector_norm(relative_r)
        relative_speed = vector_norm(relative_v)
        range_rate = dot(relative_r, relative_v) / distance if distance > 0.0 else 0.0
        actor_radius = vector_norm(actor_r)
        counterpart_radius = vector_norm(counterpart_r)
        time_since_maneuver, maneuver_flag = maneuver_features(t, maneuver_times)

        surface_distance = 0.0
        elevation = 0.0
        access_flag = 0.0
        if counterpart_is_region:
            surface_distance = EARTH_RADIUS_KM * angle_between(actor_r, counterpart_r)
            elevation = elevation_angle_deg(actor_r, counterpart_r)
            access_flag = 1.0 if elevation >= ACCESS_ELEVATION_THRESHOLD_DEG else 0.0

        sequence.append(
            [
                (t - start_time).total_seconds(),
                distance,
                relative_speed,
                range_rate,
                -range_rate,
                actor_radius,
                counterpart_radius,
                actor_radius - counterpart_radius,
                plane_angle_deg(actor_r, actor_v, counterpart_r, counterpart_v),
                surface_distance,
                elevation,
                access_flag,
                1.0 if counterpart_is_region else 0.0,
                time_since_maneuver,
                maneuver_flag,
                matching_score,
                sigma_value(error_summary, "orbit_position_sigma"),
                sigma_value(error_summary, "orbit_velocity_sigma"),
                sigma_value(error_summary, "maneuver_time_sigma"),
            ]
        )

    return {
        "sample_index": sample_index,
        "feature_names": FEATURE_NAMES,
        "sequence": sequence,
        "matching_score": matching_score,
        "y": sample.get("y"),
    }


def infer_counterpart_is_region(trajectory: list[dict[str, Any]]) -> bool:
    mean_radius = sum(vector_norm(vector3(point["r"])) for point in trajectory) / len(trajectory)
    return mean_radius <= EARTH_RADIUS_KM + REGION_RADIUS_MARGIN_KM


def parse_maneuver_times(events: list[dict[str, Any]]) -> list[datetime]:
    times = []
    for event in events:
        value = event.get("t_maneuver")
        if isinstance(value, str):
            times.append(parse_time(value))
    return sorted(times)


def maneuver_features(t: datetime, maneuver_times: list[datetime]) -> tuple[float, float]:
    if not maneuver_times:
        return 0.0, 0.0
    previous = [item for item in maneuver_times if item <= t]
    if not previous:
        return -1.0, 0.0
    elapsed = (t - previous[-1]).total_seconds()
    return elapsed, 1.0 if elapsed == 0.0 else 0.0


def sigma_value(error_summary: dict[str, Any], key: str) -> float:
    value = error_summary.get(key)
    return 0.0 if value is None else float(value)


def parse_time(value: str) -> datetime:
    t = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if t.tzinfo is None:
        return t.replace(tzinfo=timezone.utc)
    return t.astimezone(timezone.utc)


def vector3(value: list[float]) -> Vector3:
    return (float(value[0]), float(value[1]), float(value[2]))


def vector_sub(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vector_norm(v: Vector3) -> float:
    return math.sqrt(dot(v, v))


def dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vector3, b: Vector3) -> Vector3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def angle_between(a: Vector3, b: Vector3) -> float:
    norm_a = vector_norm(a)
    norm_b = vector_norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return math.pi
    cos_angle = max(-1.0, min(1.0, dot(a, b) / (norm_a * norm_b)))
    return math.acos(cos_angle)


def plane_angle_deg(actor_r: Vector3, actor_v: Vector3, counterpart_r: Vector3, counterpart_v: Vector3) -> float:
    actor_h = cross(actor_r, actor_v)
    counterpart_h = cross(counterpart_r, counterpart_v)
    return math.degrees(angle_between(actor_h, counterpart_h))


def elevation_angle_deg(actor_r: Vector3, region_r: Vector3) -> float:
    line_of_sight = vector_sub(actor_r, region_r)
    los_norm = vector_norm(line_of_sight)
    region_norm = vector_norm(region_r)
    if los_norm == 0.0 or region_norm == 0.0:
        return -90.0
    up = (region_r[0] / region_norm, region_r[1] / region_norm, region_r[2] / region_norm)
    sin_elev = max(-1.0, min(1.0, dot(line_of_sight, up) / los_norm))
    return math.degrees(math.asin(sin_elev))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def demo() -> None:
    sample = {
        "actor_trajectory": [
            {"t": "2026-01-01T00:00:00+00:00", "r": [7000, 0, 0], "v": [0, 7.5, 0]},
            {"t": "2026-01-01T00:05:00+00:00", "r": [6990, 100, 0], "v": [-0.1, 7.5, 0]},
        ],
        "counterpart_trajectory": [
            {"t": "2026-01-01T00:00:00+00:00", "r": [6378.137, 0, 0], "v": [0, 0.46, 0]},
            {"t": "2026-01-01T00:05:00+00:00", "r": [6377, 120, 0], "v": [-0.01, 0.46, 0]},
        ],
        "maneuver_events": [],
        "error_summary": {},
        "matching_score": 0.5,
        "y": None,
    }
    result = build_feature_sample(sample)
    assert result["feature_names"] == FEATURE_NAMES
    assert len(result["sequence"]) == 2
    assert len(result["sequence"][0]) == len(FEATURE_NAMES)


if __name__ == "__main__":
    demo()
    idea_dir = Path(__file__).resolve().parents[1]
    input_path = idea_dir / "generated_dataset" / "sgp4_first_pass" / "unlabeled_x_samples.json"
    output_path = idea_dir / "generated_dataset" / "sgp4_first_pass" / "analytic_features.json"
    write_json(output_path, build_feature_dataset(load_samples(input_path)))
    print(f"Analytic features written: {output_path}")
