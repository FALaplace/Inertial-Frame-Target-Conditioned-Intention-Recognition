"""Prepare two fixed matching-score scenes; run in the intention-ir environment.

Only source states and candidate validity are exported. No scores or rankings are
computed here, and no source dataset is modified. States retain their original
TEME numerical axes; the ground model is the project's spherical GMST convention.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from project_paths import resolve_project_path

import numpy as np
from sgp4.alpha5 import from_alpha5
from sgp4.api import Satrec

from data_processing import _counterfactual_actor, _validate_time_grid, iter_json_samples
from orbit_core import (
    EARTH_RADIUS_KM, GroundPointPropagator, OrbitObject, Sgp4OrbitPropagator,
    default_sensitive_regions, parse_epoch,
)


SCENES = {"spacecraft": 17985, "regions": 12976}
MAX_ABSOLUTE_TLE_AGE_DAYS = 10.0


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class MatchingValidationPrepare:
    def __init__(self, project_root=None):
        self.root = Path(project_root or Path(__file__).resolve().parents[1])
        self.frozen = self.root / "dataset"
        self.tle_dir = self.root / "tle_data/2026-09-08T16-26-29Z"

    @staticmethod
    def states(trajectory):
        result = np.asarray([[*point["r"], *point["v"]] for point in trajectory], dtype=np.float64)
        if result.shape != (181, 6) or not np.isfinite(result).all():
            raise ValueError("Expected 181 finite position/velocity states.")
        return result

    def choose_output(self):
        base = self.root / "experiment_data/matching_score_validation_20261019"
        output = base
        version = 1
        while output.exists() and any(output.rglob("*")):
            version += 1
            output = base.with_name(base.name + f"_v{version}")
        return output

    def spacecraft_pool(self, records, times, actor_id, source_hash):
        pool, states, ids = [], [], []
        seen = set()
        propagator = Sgp4OrbitPropagator()
        for raw_index, item in enumerate(records):
            identity = str(item.get("norad_cat_id", ""))
            row = {"scene": "spacecraft", "candidate_id": identity,
                   "candidate_name": item.get("object_name", ""), "counterpart_type": "satellite",
                   "raw_candidate_index": raw_index, "valid_candidate_index": "",
                   "valid": False, "exclusion_reason": "", "source_path": str(self.tle_dir / "china_leo_high_value_candidates_latest.json"),
                   "source_sha256": source_hash, "norad_cat_id": identity,
                   "epoch_utc": item.get("epoch", ""), "latitude_deg": "", "longitude_deg": "",
                   "spherical_altitude_km": "", "earth_radius_km": "",
                   "tle_age_at_start_days": "", "tle_abs_age_at_start_days": "",
                   "tle_abs_age_max_over_window_days": "", "future_epoch_at_start": ""}
            try:
                if not identity or identity in seen:
                    raise ValueError("missing_or_duplicate_norad_id")
                seen.add(identity)
                if identity == actor_id:
                    raise ValueError("actor_self_pair")
                epoch = parse_epoch(item["epoch"])
                age = (times[0] - epoch).total_seconds() / 86400.0
                max_age = max(abs((t - epoch).total_seconds()) / 86400.0 for t in (times[0], times[-1]))
                row.update(epoch_utc=epoch.isoformat(), tle_age_at_start_days=age,
                           tle_abs_age_at_start_days=abs(age), tle_abs_age_max_over_window_days=max_age,
                           future_epoch_at_start=age < 0)
                if max_age > MAX_ABSOLUTE_TLE_AGE_DAYS:
                    raise ValueError("absolute_tle_age_exceeds_10_days_within_window")
                lines = [item["tle_line1"], item["tle_line2"]]
                for number, line in enumerate(lines, start=1):
                    if len(line) != 69 or line[0] != str(number):
                        raise ValueError("invalid_tle_line_format")
                    checksum = sum(int(c) if c.isdigit() else int(c == "-") for c in line[:68]) % 10
                    if not line[68].isdigit() or checksum != int(line[68]):
                        raise ValueError("invalid_tle_checksum")
                    if from_alpha5(line[2:7]) != int(identity):
                        raise ValueError("tle_norad_mismatch")
                satellite = Satrec.twoline2rv(*lines)
                values = self.states([p.to_json() for p in propagator.propagate(OrbitObject.from_csv_row(item), times)])
                if np.any(np.linalg.norm(values[:, :3], axis=1) < satellite.radiusearthkm):
                    raise ValueError("state_below_sgp4_earth_radius")
                row.update(valid=True, valid_candidate_index=len(ids))
                ids.append(identity)
                states.append(values)
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                row["exclusion_reason"] = str(error)
            pool.append(row)
        if len(records) != 478 or len(seen) != 478 or not states:
            raise ValueError("The fixed spacecraft source must contain 478 unique candidates.")
        return pool, np.asarray(states), np.asarray(ids, dtype=str)

    def region_pool(self, times, source_hash):
        pool, states, ids = [], [], []
        propagator = GroundPointPropagator()
        regions = default_sensitive_regions()
        if len(regions) != 22 or len({region.name for region in regions}) != 22:
            raise ValueError("The fixed ground-center source must contain 22 unique centers.")
        for raw_index, region in enumerate(regions):
            values = self.states([p.to_json() for p in propagator.propagate(region, times)])
            np.testing.assert_allclose(np.linalg.norm(values[:, :3], axis=1), EARTH_RADIUS_KM, atol=1e-9, rtol=0)
            pool.append({"scene": "regions", "candidate_id": region.name, "candidate_name": region.name,
                         "counterpart_type": "ground_region", "raw_candidate_index": raw_index,
                         "valid_candidate_index": raw_index, "valid": True, "exclusion_reason": "",
                         "source_path": str(self.root / "scripts/orbit_core.py") + "::default_sensitive_regions",
                         "source_sha256": source_hash, "norad_cat_id": "", "epoch_utc": "",
                         "latitude_deg": region.latitude_deg, "longitude_deg": region.longitude_deg,
                         "spherical_altitude_km": 0.0, "earth_radius_km": EARTH_RADIUS_KM,
                         "tle_age_at_start_days": "", "tle_abs_age_at_start_days": "",
                         "tle_abs_age_max_over_window_days": "", "future_epoch_at_start": ""})
            ids.append(region.name)
            states.append(values)
        return pool, np.asarray(states), np.asarray(ids, dtype=str)

    def run(self):
        output = self.choose_output()
        dataset = self.frozen / "x_samples_frozen_18000_v1.json"
        provenance_path = self.frozen / "sample_provenance_frozen.json"
        freeze_path = self.frozen / "freeze_manifest.json"
        counterpart_path = self.tle_dir / "china_leo_high_value_candidates_latest.json"
        starlink_path = self.tle_dir / "starlink_latest.json"
        source_paths = [dataset, provenance_path, freeze_path, counterpart_path, starlink_path,
                        self.tle_dir / "manifest.json", self.root / "scripts/orbit_core.py",
                        self.root / "scripts/data_processing.py", Path(__file__).resolve(),
                        self.root / "generated_dataset/rareclass_expansion_v1/tle_audit_20260909/新快照库存与传播审计.md"]
        source_hashes = {str(path): sha256(path) for path in source_paths}
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        frozen_outputs = {resolve_project_path(name, self.root): value for name, value in freeze["outputs"].items()}
        for path in (dataset, provenance_path):
            if source_hashes[str(path)] != frozen_outputs[path]["sha256"]:
                raise ValueError(f"Frozen source hash mismatch: {path}")
        provenance = {int(row["sample_index"]): row for row in json.loads(provenance_path.read_text(encoding="utf-8"))}
        starlink = {str(row["norad_cat_id"]): row for row in json.loads(starlink_path.read_text(encoding="utf-8"))}
        counterparts = json.loads(counterpart_path.read_text(encoding="utf-8"))
        samples = {}
        for index, sample in iter_json_samples(dataset):
            if index in SCENES.values():
                samples[index] = sample
            if len(samples) == len(SCENES):
                break
        if set(samples) != set(SCENES.values()):
            raise ValueError("Missing fixed scene in frozen dataset.")
        arrays, scenes, all_pool = {}, {}, []
        for scene, index in SCENES.items():
            sample, identity = samples[index], provenance[index]
            actor = sample["actor_trajectory"]
            _validate_time_grid(actor, sample["counterpart_trajectory"], index)
            times = [parse_epoch(point["t"]) for point in actor]
            times_s = np.asarray([t.timestamp() for t in times], dtype=np.float64)
            if len(times) != 181 or not np.all(np.diff(times_s) == 60.0):
                raise ValueError("The fixed scene must span 180 minutes at 60-second intervals.")
            factual = self.states(actor)
            cf_points, cf_valid = _counterfactual_actor(actor, sample["maneuver_events"], index)
            _validate_time_grid(actor, cf_points, index)
            counterfactual = self.states(cf_points)
            if not cf_valid:
                raise ValueError("Both fixed scenes require a valid maneuver counterfactual.")
            event_time = parse_epoch(sample["maneuver_events"][0]["t_maneuver"])
            before = times_s < event_time.timestamp()
            np.testing.assert_array_equal(factual[before], counterfactual[before])
            event_index = int(np.argmin(np.abs(times_s - event_time.timestamp())))
            np.testing.assert_array_equal(factual[event_index, :3], counterfactual[event_index, :3])
            actor_id = str(identity["actor_norad_cat_id"])
            if actor_id not in starlink or identity["split"] != "test":
                raise ValueError("Expected a frozen test actor in the retained Starlink source.")
            if parse_epoch(identity["start_utc"]) != times[0]:
                raise ValueError("Provenance window does not match the raw trajectory.")
            if scene == "spacecraft":
                pool, candidates, ids = self.spacecraft_pool(counterparts, times, actor_id, source_hashes[str(counterpart_path)])
            else:
                pool, candidates, ids = self.region_pool(times, source_hashes[str(self.root / "scripts/orbit_core.py")])
            source_counterpart_id = str(identity["counterpart_id"])
            matches = np.flatnonzero(ids == source_counterpart_id)
            if len(matches) != 1:
                raise ValueError("The original frozen counterpart is absent from the rebuilt valid pool.")
            original_target = self.states(sample["counterpart_trajectory"])
            np.testing.assert_allclose(candidates[matches[0]], original_target, atol=1e-7, rtol=0)
            arrays.update({f"{scene}_times_s": times_s, f"{scene}_actor_f": factual,
                           f"{scene}_actor_cf": counterfactual, f"{scene}_candidates": candidates,
                           f"{scene}_ids": ids})
            all_pool.extend(pool)
            scenes[scene] = {
                "sample_index": index, "sample_id": identity["sample_id"], "scene_id": identity["scenario_id"],
                "scene_key": actor_id + "|" + times[0].isoformat() + "|" + identity["scenario_id"],
                "actor_id": actor_id, "actor_name": starlink[actor_id]["object_name"],
                "counterpart_type": identity["counterpart_type"], "split": identity["split"],
                "generation_development_exposed": identity.get("generation_development_exposed"),
                "window_start_utc": times[0].isoformat(), "window_end_utc": times[-1].isoformat(),
                "sample_count": 181, "step_s": 60, "maneuver_events": sample["maneuver_events"],
                "event_grid_index": event_index, "counterfactual_valid": bool(cf_valid),
                "error_summary": sample["error_summary"], "raw_provenance": identity,
                "source_dataset_path": str(dataset), "source_dataset_sha256": source_hashes[str(dataset)],
                "source_sample_canonical_sha256": canonical_sha256(sample),
                "source_actor_trajectory_canonical_sha256": canonical_sha256(actor),
                "source_counterpart_id": source_counterpart_id,
                "candidate_source_path": pool[0]["source_path"], "candidate_source_sha256": pool[0]["source_sha256"],
                "source_candidate_count": len(pool), "valid_candidate_count": len(ids),
                "data_quality_excluded_count": len(pool) - len(ids),
                "candidate_array_order": ids.tolist(),
                "actor_tle": starlink[actor_id],
                "counterfactual_method": "data_processing._counterfactual_actor: subtract the recorded impulse at the nearest grid node; propagate with orbit_core.apply_impulse_maneuver two-body RK4; factual states retained before the event.",
                "raw_frame": "Original TEME numerical axes; frozen maneuver propagation uses those axes. Ground centers use the project's spherical GMST rotation, not a precision TEME/J2000 transformation.",
                "delta_v_frame": "Same original numerical axes as the stored frozen position/velocity vectors.",
                "uncertainty_note": "Both event time_sigma and error_summary.maneuver_time_sigma are retained unchanged; this preparation does not choose a scoring uncertainty. No separate candidate covariance is supplied.",
            }
        for path in (dataset, provenance_path):
            if sha256(path) != source_hashes[str(path)]:
                raise ValueError(f"Source changed during preparation: {path}")
        preparation = output / "preparation"
        preparation.mkdir(parents=True, exist_ok=False)
        raw_path = preparation / "raw_states.npz"
        np.savez_compressed(raw_path, **arrays)
        with np.load(raw_path, allow_pickle=False) as saved:
            assert set(saved.files) == set(arrays)
            for key, values in arrays.items():
                np.testing.assert_array_equal(saved[key], values)
        with (output / "candidate_pool.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(all_pool[0]))
            writer.writeheader()
            writer.writerows(all_pool)
        metadata = {"schema_version": 1, "purpose": "Raw fixed-scene preparation; no matching score, ranking, screening threshold or display-subset selection.",
                    "output_directory": str(output), "array_columns": ["x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s"],
                    "times_s_definition": "Unix seconds since 1970-01-01T00:00:00 UTC", "dtype": "float64; candidate IDs are Unicode strings",
                    "candidate_rule": {"max_absolute_tle_age_days": MAX_ABSOLUTE_TLE_AGE_DAYS,
                                       "age_evaluation": "Maximum absolute epoch separation over both window endpoints; no distance/score preselection.",
                                       "source": str(source_paths[-1]), "earth_surface_screen": "Spacecraft radius >= SGP4 WGS72 radius; region centers fixed on sphere R=6378.137 km."},
                    "source_sha256": source_hashes, "source_files_verified_unchanged": True,
                    "raw_states_sha256": sha256(raw_path), "candidate_pool_sha256": sha256(output / "candidate_pool.csv"),
                    "checks": ["Frozen dataset/provenance match freeze-manifest hashes", "Frozen sources unchanged after preparation",
                               "Both scenes have 181 common time nodes with 60-second spacing", "F/CF states identical strictly before the maneuver",
                               "F/CF event positions identical", "Rebuilt source counterpart agrees with frozen counterpart within 1e-7",
                               "All output arrays round-trip exactly without pickle"], "scenes": scenes}
        (preparation / "scenes.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(output), "scenes": {key: {"source_candidates": value["source_candidate_count"], "valid_candidates": value["valid_candidate_count"]} for key, value in scenes.items()}}, ensure_ascii=False, indent=2))
        return output


if __name__ == "__main__":
    MatchingValidationPrepare().run()
