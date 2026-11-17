"""Extract physical evidence for Phase 3 cases without loading the 1 GB dataset.

Run this preparation step in the ``intention-ir`` environment. The resulting
JSON contains physical values only and can be plotted in the plotting runtime.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from analytic_layer import (
    ACCESS_ELEVATION_THRESHOLD_DEG,
    FEATURE_NAMES,
    build_feature_sample,
    infer_counterpart_is_region,
    parse_time,
)
from data_processing import _counterfactual_actor, _validate_time_grid, iter_json_samples
from orbit_core import gmst_angle


class PhysicalCaseData:
    """Reconstruct the same no-impulse trajectory used by the formal builder."""

    def __init__(self, project_root: Path | None = None):
        self.project_root = Path(project_root or Path(__file__).resolve().parents[1])
        frozen = self.project_root / "dataset"
        self.dataset_path = frozen / "x_samples_frozen_18000_v1.json"
        self.provenance_path = frozen / "sample_provenance_frozen.json"
        self.labels_path = frozen / "labels_main_18000_v1.json"
        self.provenance = self._index(self.provenance_path)
        self.labels = self._index(self.labels_path)

    @staticmethod
    def _index(path):
        rows = json.loads(path.read_text(encoding="utf-8"))
        result = {int(row["sample_index"]): row for row in rows}
        if len(rows) != len(result):
            raise ValueError(f"Duplicate sample_index in {path}")
        return result

    def read_samples(self, indices):
        """Use the existing one-record-per-line reader; retain requested rows only."""
        requested = {int(index) for index in indices}
        if not requested:
            return {}
        if any(index < 0 or index not in self.provenance for index in requested):
            raise ValueError("Every index must belong to the frozen provenance file.")
        samples = {}
        for index, sample in iter_json_samples(self.dataset_path):
            if index in requested:
                samples[index] = sample
                if len(samples) == len(requested):
                    break
        missing = requested - samples.keys()
        if missing:
            raise ValueError(f"Dataset lacks sample indices {sorted(missing)}")
        return samples

    @staticmethod
    def _geographic(trajectory):
        """Spherical geocentric coordinates using the project's GMST convention."""
        r = np.asarray([row["r"] for row in trajectory], dtype=np.float64)
        radius = np.linalg.norm(r, axis=1)
        if np.any(radius <= 0):
            raise ValueError("Geographic coordinates need nonzero position vectors.")
        latitude = np.degrees(np.arcsin(np.clip(r[:, 2] / radius, -1.0, 1.0)))
        theta = np.asarray([gmst_angle(parse_time(row["t"])) for row in trajectory])
        longitude = (np.degrees(np.arctan2(r[:, 1], r[:, 0]) - theta) + 180.0) % 360.0 - 180.0
        return latitude.tolist(), longitude.tolist()

    @staticmethod
    def _series(sample, index):
        # Use the existing analytic definitions in raw float64 physical units.
        feature = build_feature_sample(sample, index)
        values = np.asarray(feature["sequence"], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite physical values at sample {index}")
        names = ("distance_km", "range_rate_km_s", "relative_speed_km_s",
                 "surface_distance_km", "elevation_deg", "access_flag")
        series = {name: values[:, FEATURE_NAMES.index(name)].tolist() for name in names}
        latitude, longitude = PhysicalCaseData._geographic(sample["actor_trajectory"])
        series["actor_latitude_deg"] = latitude
        series["actor_longitude_deg"] = longitude
        return series

    def build(self, sample, index):
        """Return identity, times, events, factual and reconstructed CF evidence."""
        index = int(index)
        identity, label = self.provenance[index], self.labels[index]
        if identity["sample_id"] != label["sample_id"]:
            raise ValueError(f"Provenance/label sample_id mismatch at {index}")
        y = np.asarray([sample["y"][f"c{k}"] for k in range(8)], dtype=np.float64)
        expected = np.asarray([label["y"][f"c{k}"] for k in range(8)], dtype=np.float64)
        if not np.allclose(y, expected, atol=1e-9, rtol=0):
            raise ValueError(f"Raw/sidecar reference-label mismatch at {index}")
        actor, counterpart = sample["actor_trajectory"], sample["counterpart_trajectory"]
        _validate_time_grid(actor, counterpart, index)
        times = [parse_time(point["t"]) for point in actor]
        offsets = np.asarray([(t - times[0]).total_seconds() for t in times])
        if len(times) < 2 or np.any(np.diff(offsets) <= 0):
            raise ValueError(f"Time grid must be strictly increasing at {index}")
        for trajectory in (actor, counterpart):
            rv = np.asarray([[*row["r"], *row["v"]] for row in trajectory], dtype=np.float64)
            if rv.shape != (len(times), 6) or not np.isfinite(rv).all():
                raise ValueError(f"Invalid raw trajectory at {index}")
        counterpart_type = "ground_region" if infer_counterpart_is_region(counterpart) else "satellite"
        if counterpart_type != identity["counterpart_type"] or counterpart_type != label["counterpart_type"]:
            raise ValueError(f"Physical/provenance/label counterpart type mismatch at {index}")

        cf_actor, cf_valid = _counterfactual_actor(actor, sample.get("maneuver_events", []), index)
        cf_sample = dict(sample, actor_trajectory=cf_actor, maneuver_events=[])
        maneuvers = []
        for event in sample.get("maneuver_events", []):
            event_time = parse_time(event["t_maneuver"])
            event_offset = (event_time - times[0]).total_seconds()
            applied_index = int(np.argmin(np.abs(offsets - event_offset)))
            maneuvers.append({
                "t_utc": event_time.isoformat(),
                "time_offset_s": event_offset,
                "delta_v_km_s": [float(x) for x in event["delta_v"]],
                "delta_v_norm_km_s": float(np.linalg.norm(event["delta_v"])),
                "recorded_delta_v_norm_km_s": event.get("delta_v_norm_km_s"),
                "applied_grid_offset_s": float(offsets[applied_index]),
                "applied_grid_time_utc": times[applied_index].isoformat(),
            })
        region_latitude, region_longitude = self._geographic(counterpart)
        result = {
            "sample_index": index,
            "sample_id": identity["sample_id"],
            "actor_norad_cat_id": identity["actor_norad_cat_id"],
            "counterpart": identity.get("counterpart_id", identity.get("counterpart")),
            "counterpart_type": counterpart_type,
            "split": identity["split"],
            "y": y.tolist(),
            "class_mask": [1, 1, 0, 0, 0, 1, 1, 1] if counterpart_type == "ground_region" else [1, 1, 1, 1, 1, 1, 0, 0],
            "timestamps_utc": [t.isoformat() for t in times],
            "time_offset_s": offsets.tolist(),
            "factual": self._series(sample, index),
            "counterfactual": self._series(cf_sample, index),
            "counterfactual_valid": bool(cf_valid),
            "maneuvers": maneuvers,
            "counterpart_latitude_deg": region_latitude,
            "counterpart_longitude_deg": region_longitude,
            "metadata": {
                "dataset_path": str(self.dataset_path),
                "provenance_path": str(self.provenance_path),
                "labels_path": str(self.labels_path),
                "raw_sample_canonical_sha256": hashlib.sha256(json.dumps(sample, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
                "identity_check": "Frozen sample_index maps raw record to matching provenance and label sample_id; embedded raw label and inferred counterpart type checked. Raw records do not embed actor/counterpart identifiers.",
                "counterfactual_source": "data_processing._counterfactual_actor: subtract declared impulse at nearest recorded time, then orbit_core.apply_impulse_maneuver RK4 two-body propagation. No stored original-TLE counterfactual trajectory is present.",
                "physical_source": "analytic_layer.build_feature_sample on unnormalized raw position (km) and velocity (km/s), with float64 calculations.",
                "range_rate_definition": "dot(actor_r-counterpart_r, actor_v-counterpart_v) / distance; negative means closing, positive means separating.",
                "access_elevation_threshold_deg": ACCESS_ELEVATION_THRESHOLD_DEG,
                "ground_geometry_scope": "The ground counterpart is a spherical-Earth fixed point, not a polygon or sensor footprint. Elevation/access describe this point; spatial coverage area or footprint shift cannot be established from these fields.",
                "geographic_coordinates": "Spherical geocentric latitude; longitude rotates the stored inertial state by orbit_core.gmst_angle. Uses the project's frame convention, not a new precision geodesy product.",
            },
        }
        if not cf_valid and result["factual"] != result["counterfactual"]:
            raise ValueError(f"No-maneuver sample {index} should have identical physical tracks.")
        return result

    def load(self, indices):
        return {index: self.build(sample, index) for index, sample in self.read_samples(indices).items()}

    def export(self, indices, output_path):
        output_path = Path(output_path)
        if output_path.exists():
            raise FileExistsError(f"Preserve existing preparation file: {output_path}")
        results = self.load(indices)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        return results


if __name__ == "__main__":
    # IDE entry point: median-margin candidates chosen only from reference labels.
    data = PhysicalCaseData()
    candidate_indices = [17985, 17330, 16806, 16832, 17541, 14938, 17873, 17967, 17877, 17000,
                         12976, 12978, 12970, 12968, 12980, 12971, 12966, 12979, 12967, 12969]
    output = data.project_root / "experiment_data/baseline_test_analysis_20260917/preparation/physical_cases.json"
    result = data.export(candidate_indices, output)
    for index in (17985, 12976):
        item = result[index]
        assert item["split"] == "test" and len(item["time_offset_s"]) == 181
        assert item["counterfactual_valid"] and len(item["maneuvers"]) == 1
        print(index, item["sample_id"], item["counterpart_type"], item["maneuvers"])
    print(f"Saved physical evidence for {len(result)} candidates: {output}")
