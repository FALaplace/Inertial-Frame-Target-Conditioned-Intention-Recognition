"""Prepare the two archived scenes and their complete July counterpart pools.

Run this file directly in the intention-ir interpreter in PyCharm.  Archived
samples are read only.  The 181-point frozen trajectories remain the formal
inputs; a separate 37-point replay audits the original generation grid.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path

from project_paths import resolve_project_path

import numpy as np

from data_processing import iter_json_samples
from orbit_core import (
    EARTH_RADIUS_KM, GroundPointPropagator, OrbitObject,
    Sgp4OrbitPropagator, TrajectoryPoint, apply_impulse_maneuver,
    default_sensitive_regions, parse_epoch,
)


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "experiment_data/matching_score_legacy_20261019"
PREPARATION = OUTPUT / "preparation"
FROZEN = PROJECT / "dataset"
SAMPLES = FROZEN / "x_samples_frozen_18000_v1.json"
PROVENANCE = FROZEN / "sample_provenance_frozen.json"
SPLIT = PROJECT / "experiment_data/splits/actor_isolated_13000_2500_2500_v1__labels_18000_v1.json"
CATALOG = PROJECT / "tle_data/2026-07-08T12-48-43Z"
ACTORS = CATALOG / "starlink_latest.json"
COUNTERPARTS = CATALOG / "china_leo_high_value_candidates_latest.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")


def audit_source_integrity(manifest):
    freeze_path = FROZEN / "freeze_manifest.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    checks = []
    frozen_outputs = {resolve_project_path(name, PROJECT): value for name, value in freeze["outputs"].items()}
    for path in (SAMPLES, PROVENANCE, SPLIT):
        expected = frozen_outputs[path]
        actual = manifest["source_sha256"][str(path)]
        checks.append({"path": str(path), "sha256_at_preparation": actual,
                       "frozen_sha256": expected["sha256"],
                       "sha256_equal": actual == expected["sha256"],
                       "size_bytes": path.stat().st_size,
                       "frozen_size_bytes": expected["bytes"]})
    assert all(c["sha256_equal"] and c["size_bytes"] == c["frozen_size_bytes"] for c in checks)
    write_json(PREPARATION / "source_integrity_checks.json",
               {"freeze_manifest": str(freeze_path), "freeze_manifest_sha256": sha256(freeze_path),
                "all_checked_inputs_match_frozen_manifest": True, "checks": checks,
                "scope": "Prepared dataset, provenance and split bytes are checked; no claim of rehashing all project files."})


def array(points):
    return np.asarray([list(p.r_km) + list(p.v_km_s) for p in points], dtype=float)


def points(records):
    return [TrajectoryPoint(parse_epoch(p["t"]), tuple(p["r"]), tuple(p["v"]))
            for p in records]


def states_to_points(states, times):
    return [TrajectoryPoint(t, tuple(row[:3]), tuple(row[3:]))
            for row, t in zip(states, times)]


class LegacyScenePreparation:
    def __init__(self):
        self.sgp4 = Sgp4OrbitPropagator()
        self.ground = GroundPointPropagator()
        self.regions = default_sensitive_regions()
        self.actor_rows = json.loads(ACTORS.read_text(encoding="utf-8"))
        self.candidate_rows = json.loads(COUNTERPARTS.read_text(encoding="utf-8"))
        self.actor_by_id = {str(r["norad_cat_id"]): r for r in self.actor_rows}
        self.candidate_by_name = {r["object_name"]: str(r["norad_cat_id"])
                                  for r in self.candidate_rows}
        self.provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
        self.metadata = json.loads(SPLIT.read_text(encoding="utf-8"))["samples"]

    def extract_scenes(self):
        ordered = sorted((m for m in self.metadata if m["sample_index"] < 10400),
                         key=lambda m: m["sample_index"])
        selected = {}
        for key, kind in (("spacecraft", "satellite"), ("regions", "ground_region")):
            used = {s["actor_id"] for s in selected.values()}
            choice = next(m for m in ordered if m["counterpart_type"] == kind
                          and m["maneuver_count"] == 1
                          and str(m["actor_norad_cat_id"]) not in used)
            aid = str(choice["actor_norad_cat_id"])
            selected[key] = {"scene_key": key, "sample_index": choice["sample_index"],
                             "actor_id": aid,
                             "actor_name": self.actor_by_id[aid]["object_name"],
                             "records": []}
        index_to_scene = {}
        for key, scene in selected.items():
            for m in ordered:
                if str(m["actor_norad_cat_id"]) == scene["actor_id"]:
                    index_to_scene[m["sample_index"]] = key
        last = max(index_to_scene)
        for index, sample in iter_json_samples(SAMPLES):
            if index in index_to_scene:
                selected[index_to_scene[index]]["records"].append(
                    {"sample_index": index, "sample": sample})
            if index >= last:
                break
        for scene in selected.values():
            assert len(scene["records"]) == 40
            first = scene["records"][0]["sample"]
            assert len(first["maneuver_events"]) == 1
            for item in scene["records"]:
                for key in ("actor_trajectory", "maneuver_events", "error_summary"):
                    assert item["sample"][key] == first[key], (scene["actor_id"], key)
        (PREPARATION / "source_scene_records.json").write_text(
            json.dumps({"scenes": selected}, ensure_ascii=False, allow_nan=False),
            encoding="utf-8")
        return selected

    def build(self):
        PREPARATION.mkdir(parents=True, exist_ok=True)
        selected = self.extract_scenes()
        raw, historical, metadata, pool = {}, {}, {}, []
        for scene_key, scene in selected.items():
            sample = scene["records"][0]["sample"]
            factual = points(sample["actor_trajectory"])
            times = [p.t for p in factual]
            offsets = np.array([(t - times[0]).total_seconds() for t in times])
            assert len(times) == 181 and np.all(np.diff(offsets) == 60)
            actor = OrbitObject.from_csv_row(self.actor_by_id[scene["actor_id"]])
            counterfactual = self.sgp4.propagate(actor, times)
            event = sample["maneuver_events"][0]
            event_time = parse_epoch(event["t_maneuver"])
            event_index = min(range(len(times)), key=lambda i: abs((times[i] - event_time).total_seconds()))
            assert times[event_index] == event_time and event_index % 5 == 0
            replay_fine = apply_impulse_maneuver(counterfactual, event_time, tuple(event["delta_v"]))
            replay_old = apply_impulse_maneuver(counterfactual[::5], event_time, tuple(event["delta_v"]))
            factual_array, cf_array = array(factual), array(counterfactual)
            fine_delta = np.linalg.norm(factual_array[:, :3] - array(replay_fine)[:, :3], axis=1)
            old_delta = np.linalg.norm(factual_array[::5, :3] - array(replay_old)[:, :3], axis=1)
            assert np.max(fine_delta) < 1e-10
            unix_times = np.array([t.timestamp() for t in times])
            raw.update({scene_key + "_times_s": unix_times,
                        scene_key + "_time_offset_s": offsets, scene_key + "_actor_f": factual_array,
                        scene_key + "_actor_cf": cf_array})
            historical.update({scene_key + "_times_s": unix_times[::5],
                               scene_key + "_time_offset_s": offsets[::5],
                               scene_key + "_actor_f": array(replay_old),
                               scene_key + "_actor_cf": cf_array[::5]})
            valid_by_key, excluded = {}, []
            for kind in ("spacecraft", "regions"):
                arrays, ids = [], []
                items = self.candidate_rows if kind == "spacecraft" else self.regions
                for order, obj in enumerate(items):
                    if kind == "spacecraft":
                        name, cid = obj["object_name"], str(obj["norad_cat_id"])
                        lat, lon = "", ""
                    else:
                        name = cid = obj.name
                        lat, lon = obj.latitude_deg, obj.longitude_deg
                    row = {"scene_key": scene_key, "scene_type": kind, "type": kind,
                           "candidate_id": cid, "name": name,
                           "source_order": order if kind == "spacecraft" else len(self.candidate_rows) + order,
                           "valid": True, "reason": "", "latitude_deg": lat, "longitude_deg": lon}
                    try:
                        trajectory = (self.sgp4.propagate(OrbitObject.from_csv_row(obj), times)
                                      if kind == "spacecraft" else self.ground.propagate(obj, times))
                        state = array(trajectory)
                        if not np.isfinite(state).all():
                            raise ValueError("nonfinite propagated state")
                        if kind == "spacecraft" and np.min(np.linalg.norm(state[:, :3], axis=1)) <= EARTH_RADIUS_KM:
                            raise ValueError("propagated geocentric radius reaches spherical Earth")
                        arrays.append(state)
                        ids.append(cid)
                        valid_by_key[(kind, cid)] = state
                    except ValueError as exc:
                        row["valid"], row["reason"] = False, str(exc)
                        excluded.append(dict(row))
                    pool.append(row)
                raw[scene_key + "_" + kind + "_candidates"] = np.asarray(arrays)
                raw[scene_key + "_" + kind + "_ids"] = np.asarray(ids, dtype=str)
                historical[scene_key + "_" + kind + "_candidates"] = np.asarray(arrays)[:, ::5]
                historical[scene_key + "_" + kind + "_ids"] = np.asarray(ids, dtype=str)
            source_pairs, max_pair_delta = [], 0.0
            for record in scene["records"]:
                prov = self.provenance[record["sample_index"]]
                kind = "spacecraft" if prov["counterpart_type"] == "satellite" else "regions"
                cid = (self.candidate_by_name[prov["counterpart"]] if kind == "spacecraft"
                       else prov["counterpart"])
                rebuilt = valid_by_key[(kind, cid)]
                pair_delta = float(np.max(np.abs(rebuilt - array(points(record["sample"]["counterpart_trajectory"])))))
                max_pair_delta = max(max_pair_delta, pair_delta)
                assert pair_delta < 1e-9, (record["sample_index"], pair_delta)
                source_pairs.append({"sample_index": record["sample_index"], "scene_type": kind,
                                     "candidate_id": cid, "name": prov["counterpart"],
                                     "stored_matching_score": record["sample"]["matching_score"],
                                     "reference_category": "record_only_not_independent_relevance"})
            chosen = next(p for p in source_pairs if p["sample_index"] == scene["sample_index"])
            metadata[scene_key] = {
                "scene_id": "original10400_actor_" + scene["actor_id"],
                "actor_id": scene["actor_id"], "actor_name": scene["actor_name"],
                "sample_index": scene["sample_index"],
                "window_start_utc": times[0].isoformat(), "window_end_utc": times[-1].isoformat(),
                "maneuver_events": sample["maneuver_events"], "error_summary": sample["error_summary"],
                "source_counterpart_id": chosen["candidate_id"],
                "source_counterpart_type": chosen["scene_type"], "event_grid_index": event_index,
                "history_event_grid_index": event_index // 5,
                "catalog_count": {"spacecraft": len(self.candidate_rows), "regions": len(self.regions)},
                "valid_count": {k: len(raw[scene_key + "_" + k + "_ids"]) for k in ("spacecraft", "regions")},
                "quality_excluded": excluded, "source_pairs": source_pairs,
                "source_frame": "TEME-like original pipeline axes; SGP4 TEME and GMST spherical ground",
                "formal_time_grid": "181 nodes at 60 s, copied frozen factual actor",
                "historical_time_grid": "37 nodes at 300 s, July TLE plus archived impulse replay",
                "counterfactual_definition": "Same July actor TLE propagated by SGP4 without impulse; diagnostic only, not historical gain input",
                "replay_checks": {"frozen_vs_replayed60s_max_position_km": float(np.max(fine_delta)),
                                  "frozen_subsample_vs_replayed300s_max_position_km": float(np.max(old_delta)),
                                  "frozen_counterparts_vs_catalog_replay_max_abs_state_difference": max_pair_delta},
            }
            print(scene_key, "ready", metadata[scene_key]["valid_count"], flush=True)
        np.savez_compressed(PREPARATION / "raw_states.npz", **raw)
        np.savez_compressed(PREPARATION / "history_raw_states.npz", **historical)
        with (PREPARATION / "candidate_pool.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(pool[0]))
            writer.writeheader()
            writer.writerows(pool)
        sources = [SAMPLES, PROVENANCE, SPLIT, ACTORS, COUNTERPARTS, Path(__file__),
                   PROJECT / "scripts/orbit_core.py", PREPARATION / "source_scene_records.json"]
        manifest = {
            "selection_rule": "Ascending original sample_index <10400; first single-maneuver actor with satellite pair; then first distinct single-maneuver actor with ground pair. No labels, scores, or rankings used.",
            "source_sha256": {str(p): sha256(p) for p in sources}, "scenes": metadata,
            "quality_rule": "Exclude propagation exceptions, nonfinite state, or satellite geocentric radius <=6378.137 km. Report exclusions apart from screening.",
        }
        write_json(PREPARATION / "scenes.json", manifest)
        audit_source_integrity(manifest)
        return raw, historical, manifest

    def score(self, raw, historical, manifest):
        from matching_legacy_score import CandidateMatcher
        matcher = CandidateMatcher()
        result, regressions, match_checks = {"scenes": {}}, [], {}
        for scene_key, meta in manifest["scenes"].items():
            from datetime import datetime, timezone
            times = [datetime.fromtimestamp(float(t), timezone.utc) for t in raw[scene_key + "_times_s"]]
            main_actor = states_to_points(raw[scene_key + "_actor_f"], times)
            old_actor = states_to_points(historical[scene_key + "_actor_f"], times[::5])
            output, old_output = {}, {}
            for kind in ("spacecraft", "regions"):
                score_fn = matcher.score_satellite_pair_details if kind == "spacecraft" else matcher.score_region_pair_details
                output[kind], old_output[kind] = [], []
                ids = raw[scene_key + "_" + kind + "_ids"]
                for i, cid in enumerate(ids):
                    for archive, a, target_times, storage in (
                            (raw, main_actor, times, output),
                            (historical, old_actor, times[::5], old_output)):
                        target = states_to_points(archive[scene_key + "_" + kind + "_candidates"][i], target_times)
                        score = score_fn(a, target, meta["maneuver_events"], meta["error_summary"])
                        storage[kind].append({"candidate_id": str(cid), "total": score.total,
                                              "components": score.components, "metrics": score.metrics})
            result["scenes"][scene_key] = output
            main_map = {(kind, s["candidate_id"]): s for kind, rows in output.items() for s in rows}
            old_map = {(kind, s["candidate_id"]): s for kind, rows in old_output.items() for s in rows}
            old_order = sorted([(kind, s) for kind in ("spacecraft", "regions") for s in old_output[kind]],
                               key=lambda pair: pair[1]["total"], reverse=True)
            rank = {(kind, s["candidate_id"]): i+1 for i, (kind, s) in enumerate(old_order)}
            for source in meta["source_pairs"]:
                key = (source["scene_type"], source["candidate_id"])
                old, main, stored = old_map[key]["total"], main_map[key]["total"], source["stored_matching_score"]
                regressions.append({"scene_key": scene_key, **source,
                                    "replay_37_total": old, "formal_181_total": main,
                                    "replay_minus_stored": old - stored,
                                    "formal_minus_stored": main - stored,
                                    "replay_mixed_rank": rank[key], "replay_top40": rank[key] <= 40,
                                    "replay_rounded6_equal": round(old, 6) == stored})
            original = {(s["scene_type"], s["candidate_id"]) for s in meta["source_pairs"]}
            reproduced = {(kind, s["candidate_id"]) for kind, s in old_order[:40]}
            match_checks[scene_key] = {
                "source_count": len(original), "replayed_top40_count": len(reproduced),
                "intersection_count": len(original & reproduced), "exact_top40_set_match": original == reproduced,
                "stored_score_rounded6_match_count": sum(r["replay_rounded6_equal"] for r in regressions if r["scene_key"] == scene_key),
                "replay_scores": old_output,
            }
        result["scorer_sha256"] = sha256(PROJECT / "scripts/matching_legacy_score.py")
        result["formal_input"] = "Frozen 181-node facts with historical formula and parameters"
        write_json(PREPARATION / "legacy_scores.json", result)
        write_json(PREPARATION / "archive_score_regression.json", match_checks)
        with (PREPARATION / "archive_score_regression.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(regressions[0]))
            writer.writeheader()
            writer.writerows(regressions)
        for check in match_checks.values():
            assert check["exact_top40_set_match"]
            assert check["stored_score_rounded6_match_count"] == 40
        anchor_path = OUTPUT / "history/original_summary_first_record_row4407.txt"
        if anchor_path.exists():
            anchor = ast.literal_eval(next(line for line in anchor_path.read_text(encoding="utf-8").splitlines()
                                           if line.startswith("{'actor':")))
            cid = manifest["scenes"]["spacecraft"]["source_counterpart_id"]
            recovered = next(s for s in match_checks["spacecraft"]["replay_scores"]["spacecraft"]
                             if s["candidate_id"] == cid)
            checks = [{"field": "total", "archived": anchor["matching_score"], "replayed": recovered["total"]}]
            for source_key, replay_key in (("score_components", "components"), ("score_metrics", "metrics")):
                checks.extend({"field": source_key + "." + key, "archived": value,
                               "replayed": recovered[replay_key][key]}
                              for key, value in anchor[source_key].items())
            for row in checks:
                row["equal_at_archived_6_decimals"] = round(row["replayed"], 6) == row["archived"]
            assert all(row["equal_at_archived_6_decimals"] for row in checks)
            write_json(PREPARATION / "historical_detail_anchor_regression.json",
                       {"source_path": str(anchor_path), "source_sha256": sha256(anchor_path),
                        "sample_index": 0, "candidate_id": cid, "checked_value_count": len(checks),
                        "all_equal_at_archived_precision": True, "checks": checks,
                        "scope": "One archived detailed pair: total plus five components plus seventeen metrics. Other 79 pairs have independent archived totals only."})
        print("historical_regression", {k: {f:v for f,v in m.items() if f != "replay_scores"}
                                         for k,m in match_checks.items()}, flush=True)

    def run(self):
        raw, historical, manifest = self.build()
        self.score(raw, historical, manifest)


if __name__ == "__main__":
    LegacyScenePreparation().run()
