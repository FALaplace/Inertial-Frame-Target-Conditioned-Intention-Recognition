from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


LABEL_RULE_VERSION = "simple_v0"

LABELS = {
    "none": "C0_no_clear_intention",
    "pass": "C1_natural_pass",
    "approach": "C2_close_approach",
    "flyby": "C3_high_speed_flyby",
    "coverage": "C6_region_coverage",
}


def load_feature_samples(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, list):
        raise ValueError("Feature file must contain a JSON list.")
    return payload


def label_dataset(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [label_sample(sample) for sample in samples]


def label_sample(sample: dict[str, Any]) -> dict[str, Any]:
    evidence = summarize_sample(sample)
    label = assign_label(evidence)
    output = dict(sample)
    output["pseudo_label"] = label
    output["label_rule_version"] = LABEL_RULE_VERSION
    output["label_evidence"] = evidence
    return output


def summarize_sample(sample: dict[str, Any]) -> dict[str, float]:
    names = sample["feature_names"]
    index = {name: i for i, name in enumerate(names)}
    sequence = sample["sequence"]
    min_distance_index = min(
        range(len(sequence)),
        key=lambda i: sequence[i][index["distance_km"]],
    )
    min_distance_row = sequence[min_distance_index]
    is_region = max(row[index["counterpart_is_region"]] for row in sequence) >= 0.5

    min_surface_distance = 0.0
    if is_region:
        min_surface_distance = min(row[index["surface_distance_km"]] for row in sequence)

    return {
        "matching_score": float(sample.get("matching_score", 0.0)),
        "counterpart_is_region": 1.0 if is_region else 0.0,
        "min_distance_km": min_distance_row[index["distance_km"]],
        "relative_speed_at_min_distance_km_s": min_distance_row[index["relative_speed_km_s"]],
        "min_surface_distance_km": min_surface_distance,
        "max_access_flag": max(row[index["access_flag"]] for row in sequence),
    }


def assign_label(evidence: dict[str, float]) -> str:
    matching_score = evidence["matching_score"]
    if evidence["counterpart_is_region"] >= 0.5:
        if evidence["max_access_flag"] >= 0.5 and matching_score >= 0.4:
            return LABELS["coverage"]
        if evidence["min_surface_distance_km"] <= 600.0 or matching_score >= 0.4:
            return LABELS["pass"]
        return LABELS["none"]

    min_distance = evidence["min_distance_km"]
    speed_at_min = evidence["relative_speed_at_min_distance_km_s"]
    if min_distance <= 700.0 and speed_at_min >= 12.0:
        return LABELS["flyby"]
    if min_distance <= 700.0:
        return LABELS["approach"]
    if min_distance <= 1200.0 or matching_score >= 0.4:
        return LABELS["pass"]
    return LABELS["none"]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def demo() -> None:
    sample = {
        "feature_names": [
            "distance_km",
            "relative_speed_km_s",
            "surface_distance_km",
            "access_flag",
            "counterpart_is_region",
        ],
        "sequence": [
            [650.0, 7.5, 0.0, 0.0, 0.0],
            [500.0, 8.0, 0.0, 0.0, 0.0],
        ],
        "matching_score": 0.3,
    }
    assert label_sample(sample)["pseudo_label"] == LABELS["approach"]


if __name__ == "__main__":
    demo()
    idea_dir = Path(__file__).resolve().parents[1]
    input_path = idea_dir / "generated_dataset" / "sgp4_first_pass" / "analytic_features.json"
    output_path = idea_dir / "generated_dataset" / "sgp4_first_pass" / "pseudo_labeled_features.json"
    labeled = label_dataset(load_feature_samples(input_path))
    write_json(output_path, labeled)
    counts = Counter(sample["pseudo_label"] for sample in labeled)
    print(f"Pseudo labels written: {output_path}")
    print(dict(sorted(counts.items())))
