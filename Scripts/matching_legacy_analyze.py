"""Analyze legacy scores on two unchanged 60-second frozen actor scenes.

Run after matching_legacy_prepare.py, using base Anaconda (NumPy and ERFA).
Plotting is in the three matching_legacy_plot_<figure>.py scripts; this file never plots.
"""

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from matching_validation_frames import (FRAME_METADATA, actor_rtn_basis,
                                       relative_rtn, spherical_ground_track,
                                       teme_to_j2000)
from matching_validation_metrics import compute_metrics
from matching_validation_run import digest, utc, write_csv


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'experiment_data/matching_score_legacy_20261019'
COMPONENTS = {
    'spacecraft': [('prox', 'proximity', .30), ('geom', 'geometry', .20),
                   ('reach', 'reachability', .20), ('man', 'maneuver_gain', .20),
                   ('unc', 'uncertainty', .10)],
    'regions': [('ground', 'ground_track', .30), ('access', 'access', .25),
                ('revisit', 'revisit', .20), ('cov', 'coverage_gain', .15),
                ('unc', 'uncertainty', .10)],
}
PHYSICAL_PARAMETERS = {
    'shared': {'earth_radius_km': 6378.137, 'mu_km3_s2': 398600.4418},
    'spacecraft': {'dwell_distance_threshold_km': 1200.0},
    'regions': {'elevation_threshold_deg': 10.0},
}


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def rank_pool(rows, budget=40):
    """Stable score sort preserves the original satellite-then-region order."""
    for rank, row in enumerate(sorted(rows, key=lambda r: -r['total_score']), 1):
        row['mixed_rank'] = rank
        row['retained'] = rank <= budget
    for branch in COMPONENTS:
        subset = [r for r in rows if r['scene_type'] == branch]
        for rank, row in enumerate(sorted(subset, key=lambda r: -r['total_score']), 1):
            row['score_rank'] = rank
        k = sum(r['retained'] for r in subset)
        # Distance has its own stable catalogue-order tie rule, with identical K.
        for rank, row in enumerate(sorted(subset, key=lambda r: r['distance_value_km']), 1):
            row['distance_rank'] = rank
            row['distance_retained'] = rank <= k
    assert sum(r['retained'] for r in rows) == min(budget, len(rows))
    return sorted(rows, key=lambda r: r['mixed_rank'])[min(budget, len(rows)) - 1]['total_score']


def select_display(rows, branch, source_id):
    """Physical contrasts selected without score/rank or model/label information."""
    selected = {}

    def add(row, reason):
        selected.setdefault(row['candidate_id'], []).append(reason)

    # Identical catalogue trajectories stay in the full pool, but need one curve in a figure.
    anchor = next(r for r in rows if r['candidate_id'] == source_id)
    unique = {anchor['trajectory_sha256']: anchor}
    for row in sorted(rows, key=lambda r: r['candidate_id']):
        unique.setdefault(row['trajectory_sha256'], row)
    ordered = sorted(unique.values(), key=lambda r: r['candidate_id'])
    add(next(r for r in ordered if r['candidate_id'] == source_id),
        'First saved pair of this type in the preselected original actor group; not a relevance label.')
    add(min(ordered, key=lambda r: r['distance_value_km']), 'Smallest full-pool factual distance.')
    if branch == 'spacecraft':
        add(min(ordered, key=lambda r: r['legacy_geometry_plane_angle_deg']),
            'Smallest legacy angular-momentum direction angle.')
        add(max(ordered, key=lambda r: r['legacy_maneuver_gain_approach_gain_km']),
            'Largest legacy pre/post minimum-distance improvement.')
    else:
        add(max(ordered, key=lambda r: r['legacy_access_access_duration_s']),
            'Longest legacy region-center access duration.')
        add(max(ordered, key=lambda r: r['legacy_coverage_gain_coverage_gain_s']),
            'Largest legacy pre/post access-duration improvement.')
    add(max(ordered, key=lambda r: r['distance_value_km']), 'Largest minimum factual distance.')
    if len(selected) < 5:
        close = sorted(ordered, key=lambda r: r['distance_value_km'])[:max(1, len(ordered) // 4)]
        for row in sorted(close, key=lambda r: -r['relative_speed_at_ca_F_km_s']):
            if len(selected) == 5:
                break
            add(row, 'Largest closest-approach relative speed among the nearest-distance quartile.')
    if len(selected) < 5:
        for row in ordered:
            if len(selected) == 5:
                break
            add(row, 'Catalogue-identity fallback after duplicate physical selections.')
    assert len(selected) == 5
    return selected


def self_check():
    rows = [dict(candidate_id=str(i), scene_type='spacecraft' if i < 3 else 'regions',
                 total_score=.5, distance_value_km=10 - i) for i in range(6)]
    rank_pool(rows, 4)
    assert [r['candidate_id'] for r in rows if r['retained']] == ['0', '1', '2', '3']
    assert sum(r['distance_retained'] for r in rows if r['scene_type'] == 'regions') == 1
    assert next(r for r in rows if r['candidate_id'] == '5')['distance_retained']


class MatchingLegacyAnalysis:
    def __init__(self, output=OUTPUT):
        self.output = Path(output)

    def run(self):
        self_check()
        out = self.output
        source = json.loads((out / 'preparation/scenes.json').read_text(encoding='utf-8'))
        scores = json.loads((out / 'preparation/legacy_scores.json').read_text(encoding='utf-8'))
        with (out / 'preparation/candidate_pool.csv').open(encoding='utf-8-sig', newline='') as stream:
            pool = {(r['scene_key'], r['scene_type'], r['candidate_id']): r for r in csv.DictReader(stream)}
        source_records = json.loads((out / 'preparation/source_scene_records.json').read_text(encoding='utf-8'))
        arrays, all_rows, summaries, selected_rows, series_rows, coordinate_rows = {}, [], [], [], [], []
        checks, analyses, ranking_changes = {}, {}, []
        with np.load(out / 'preparation/raw_states.npz', allow_pickle=False) as raw:
            for scene_key, meta in source['scenes'].items():
                times = raw[scene_key + '_times_s']
                actor, cf = raw[scene_key + '_actor_f'], raw[scene_key + '_actor_cf']
                assert len(times) == 181 and np.allclose(np.diff(times), 60, atol=1e-6, rtol=0)
                event = meta['maneuver_events'][0]
                event_s = datetime.fromisoformat(event['t_maneuver'].replace('Z', '+00:00')).timestamp()
                metric_meta = dict(scene_id=meta['scene_id'], maneuver_time_unix_s=event_s,
                                   sigma_r_km=meta['error_summary']['orbit_position_sigma'],
                                   sigma_tm_s=meta['error_summary']['maneuver_time_sigma'])
                actor_rows, branch_series = [], {}
                for branch in COMPONENTS:
                    ids = raw[scene_key + '_' + branch + '_ids'].tolist()
                    candidates = raw[scene_key + '_' + branch + '_candidates']
                    trajectory_hashes = {identity: hashlib.sha256(candidates[i].tobytes()).hexdigest()
                                         for i, identity in enumerate(ids)}
                    metrics, series = compute_metrics(times, actor, cf, candidates, ids,
                                                       dict(metric_meta, scene_type=branch), PHYSICAL_PARAMETERS)
                    metric_by_id = {r['candidate_id']: r for r in metrics}
                    legacy = {r['candidate_id']: r for r in scores['scenes'][scene_key][branch]}
                    assert set(ids) == set(legacy)
                    branch_series[branch] = series
                    for identity in ids:
                        value = legacy[identity]
                        catalog = pool[(scene_key, branch, identity)]
                        row = dict(metric_by_id[identity], scene_key=scene_key, actor_id=meta['actor_id'],
                                   candidate_name=catalog['name'], total_score=float(value['total']),
                                   trajectory_sha256=trajectory_hashes[identity],
                                   relevance_evidence='unlabelled_catalogue_candidate')
                        row.update({'legacy_' + k: v for k, v in value['metrics'].items()})
                        for short, long_name, weight in COMPONENTS[branch]:
                            row['S_' + short] = float(value['components'][long_name])
                            row['wS_' + short] = weight * row['S_' + short]
                            assert 0 <= row['S_' + short] <= 1
                        contribution = sum(row['wS_' + short] for short, _, _ in COMPONENTS[branch])
                        assert abs(contribution - row['total_score']) < 1e-12
                        assert 0 <= row['total_score'] <= 1
                        assert not any('placeholder' in key for key in value['metrics'])
                        row['distance_value_km'] = row['d_min_F_km'] if branch == 'spacecraft' else row['ell_min_F_km']
                        for duration in ('T_dwell_F_s', 'T_access_F_s', 'T_access_CF_s'):
                            if row.get(duration) is not None:
                                assert 0 <= row[duration] <= times[-1] - times[0] + 1e-6
                        actor_rows.append(row)
                cutoff = rank_pool(actor_rows)
                old_records = source_records['scenes'][scene_key]['records']
                # The source pair identities are recovered from the explicit preparation audit.
                old_ids = {r['scene_type'] + ':' + r['candidate_id'] for r in meta['source_pairs']}
                for row in actor_rows:
                    identity_key = row['scene_type'] + ':' + row['candidate_id']
                    row['saved_in_old_top40'] = identity_key in old_ids if old_ids else None
                    if row['saved_in_old_top40']:
                        row['relevance_evidence'] = 'historical_selected_pair_only'
                all_rows.extend(actor_rows)
                for branch in COMPONENTS:
                    subset = [r for r in actor_rows if r['scene_type'] == branch]
                    kept = [r for r in subset if r['retained']]
                    distance = [r for r in subset if r['distance_retained']]
                    common = set(r['candidate_id'] for r in kept) & set(r['candidate_id'] for r in distance)
                    source_count = sum(k[0] == scene_key and k[1] == branch for k in pool)
                    summary = dict(scene=scene_key, scene_type=branch, actor_id=meta['actor_id'],
                                   source_candidates=source_count, data_quality_excluded=source_count - len(subset),
                                   candidates=len(subset), retained=len(kept), retained_fraction=len(kept) / len(subset),
                                   reduction_fraction=1 - len(kept) / len(subset), distance_budget=len(kept),
                                   same_budget_overlap=len(common), mixed_pool_count=len(actor_rows), mixed_retained=40,
                                   mixed_score_cutoff=cutoff, independent_relevant_set_available=False,
                                   spearman_score_distance=float(np.corrcoef([r['score_rank'] for r in subset],
                                                                          [r['distance_rank'] for r in subset])[0, 1]))
                    assert len(kept) == len(distance)
                    summaries.append(summary)
                    for row in subset:
                        if row['retained'] != row['distance_retained']:
                            ranking_changes.append(dict(row, selection_difference='score_only' if row['retained'] else 'distance_only'))
                branch = scene_key
                subset = [r for r in actor_rows if r['scene_type'] == branch]
                ids = raw[scene_key + '_' + branch + '_ids'].tolist()
                selection = select_display(subset, branch, meta['source_counterpart_id'])
                factual_j = teme_to_j2000(times, actor)
                cf_j = teme_to_j2000(times, cf)
                c_j = teme_to_j2000(times, raw[scene_key + '_' + branch + '_candidates'])
                basis = actor_rtn_basis(factual_j)
                c_rtn = relative_rtn(c_j[..., :3], factual_j[:, :3], basis)
                cf_rtn = relative_rtn(cf_j[:, :3], factual_j[:, :3], basis)
                norm_error = float(np.max(np.abs(np.linalg.norm(c_rtn, axis=-1)
                                                - np.linalg.norm(c_j[..., :3] - factual_j[:, :3], axis=-1))))
                rotation_error = float(np.max(np.abs(basis @ basis.swapaxes(-1, -2) - np.eye(3))))
                assert norm_error < 1e-8 and rotation_error < 1e-13
                arrays.update({branch + '_times_s': times, branch + '_ids': np.asarray(ids),
                               branch + '_actor_f': factual_j, branch + '_actor_cf': cf_j,
                               branch + '_candidates': c_j, branch + '_candidate_rtn': c_rtn,
                               branch + '_actor_cf_rtn': cf_rtn,
                               branch + '_actor_f_ground': spherical_ground_track(times, factual_j[:, :3]),
                               branch + '_actor_cf_ground': spherical_ground_track(times, cf_j[:, :3])})
                display = []
                by_id = {r['candidate_id']: r for r in subset}
                for n, (identity, reason) in enumerate(selection.items(), 1):
                    row = by_id[identity]
                    catalog = pool[(scene_key, branch, identity)]
                    label = ('T' if branch == 'spacecraft' else 'R') + str(n)
                    item = dict(candidate_id=identity, candidate_name=row['candidate_name'],
                                array_index=ids.index(identity), label=label, metric=row, component=row,
                                selection_reason=' '.join(reason))
                    if branch == 'regions':
                        item.update(latitude_deg=float(catalog['latitude_deg']), longitude_deg=float(catalog['longitude_deg']))
                    display.append(item)
                    selected_rows.append(dict(row, display_label=label, selection_reason=item['selection_reason']))
                analyses[branch] = dict(meta, valid_candidate_count=len(ids), display=display,
                                        event_offset_min=(event_s - times[0]) / 60,
                                        mixed_score_cutoff=cutoff, mixed_pool_count=len(actor_rows), mixed_retained=40)
                checks[branch] = dict(rtn_distance_max_error_km=norm_error, rtn_orthogonality_error=rotation_error,
                                     actor_f_rtn_is_zero=True, mixed_candidates_processed=len(actor_rows),
                                     type_candidates_processed=len(ids), all_components_finite=True,
                                     weighted_sum_max_tolerance=1e-12, same_actor_for_all_candidates=True,
                                     no_missing_input_placeholders_used=True, saved_pair_count=len(old_records))
                series = branch_series[branch]
                for i, identity in enumerate(ids):
                    for j, timestamp in enumerate(times):
                        row = dict(scene=branch, candidate_id=identity, utc=utc(timestamp),
                                   time_offset_s=float(timestamp - times[0]))
                        for key, values in series.items():
                            if isinstance(values, np.ndarray) and values.shape == (len(ids), len(times)):
                                value = values[i, j].item()
                                row[key] = value if isinstance(value, bool) or np.isfinite(value) else None
                        series_rows.append(row)
                objects = [('actor_F', meta['actor_id'], factual_j, np.zeros_like(cf_rtn)),
                           ('actor_CF', meta['actor_id'], cf_j, cf_rtn)]
                objects.extend(('counterpart', identity, c_j[i], c_rtn[i]) for i, identity in enumerate(ids))
                for role, identity, states, rtn in objects:
                    for j, timestamp in enumerate(times):
                        row = dict(scene=branch, object_role=role, object_id=identity, utc=utc(timestamp),
                                   time_offset_s=float(timestamp - times[0]), frame='J2000',
                                   lvlh_origin='actor_F:' + meta['actor_id'], lvlh_axes='RTN')
                        row.update(zip(('x_km', 'y_km', 'z_km', 'vx_km_s', 'vy_km_s', 'vz_km_s'), states[j].tolist()))
                        row.update(zip(('R_km', 'T_km', 'N_km'), rtn[j].tolist()))
                        coordinate_rows.append(row)
        write_csv(out / 'all_candidate_results.csv', all_rows)
        score_fields = ('scene_key', 'scene_type', 'actor_id', 'candidate_id', 'candidate_name', 'total_score',
                        'score_rank', 'mixed_rank', 'distance_rank', 'retained', 'distance_retained')
        write_csv(out / 'matching_components.csv', [{k: v for k, v in r.items() if k in score_fields or k.startswith(('S_', 'wS_'))} for r in all_rows])
        write_csv(out / 'matching_metrics.csv', [{k: v for k, v in r.items() if not k.startswith(('S_', 'wS_'))} for r in all_rows])
        write_csv(out / 'all_screening_summary.csv', summaries)
        write_csv(out / 'matching_screening_summary.csv', [r for r in summaries if r['scene'] == r['scene_type']])
        write_csv(out / 'selected_cases.csv', selected_rows)
        write_csv(out / 'ranking_changes.csv', ranking_changes)
        write_csv(out / 'matching_timeseries.csv', series_rows)
        write_csv(out / 'matching_trajectories_3d.csv', coordinate_rows)
        np.savez_compressed(out / 'analysis_states.npz', **arrays)
        dump(out / 'analysis.json', dict(parameters=PHYSICAL_PARAMETERS, scenes=analyses,
                                       scoring_version='Historical formulas and fixed parameters; frozen 60-second states',
                                       counterfactual_role='Physical diagnostic only; historical gain uses pre/post factual intervals'))
        dump(out / 'validation_checks.json', checks)
        self.stability(all_rows, selected_rows)
        self.historical_comparison(all_rows)
        self.refresh_manifest()
        print(json.dumps(summaries, ensure_ascii=False, indent=2))

    def stability(self, rows, selected):
        results, details = [], []
        # Diagnostic perturbations only; never used to choose the reported configuration.
        for scene in ('spacecraft', 'regions'):
            baseline = [dict(r) for r in rows if r['scene_key'] == scene]
            for name, factor, budget in [('baseline', 1., 40), ('geometry_0.8', .8, 40),
                                         ('geometry_1.2', 1.2, 40), ('budget_32', 1., 32), ('budget_48', 1., 48)]:
                changed = [dict(r) for r in baseline]
                for row in changed:
                    if row['scene_type'] == 'spacecraft' and factor != 1:
                        values = [('altitude_gap_km', 500., .35), ('inclination_gap_deg', 12., .25),
                                  ('plane_angle_deg', 20., .25), ('phase_gap_deg', 45., .15)]
                        geometry = sum(w * np.exp(-row['legacy_geometry_' + field] / (scale * factor))
                                       for field, scale, w in values)
                        row['total_score'] += .2 * (float(geometry) - row['S_geom'])
                cutoff = rank_pool(changed, budget)
                for branch in COMPONENTS:
                    sub = [r for r in changed if r['scene_type'] == branch]
                    base = [r for r in baseline if r['scene_type'] == branch]
                    results.append(dict(scene=scene, scene_type=branch, variant=name, mixed_budget=budget,
                                        mixed_score_cutoff=cutoff, retained=sum(r['retained'] for r in sub),
                                        candidates=len(sub), reduction_fraction=1 - sum(r['retained'] for r in sub) / len(sub),
                                        spearman_to_baseline=float(np.corrcoef([r['score_rank'] for r in base],
                                                                             [r['score_rank'] for r in sub])[0, 1])))
                wanted = {r['candidate_id'] for r in selected if r['scene_key'] == scene}
                details.extend(dict(scene=scene, variant=name, candidate_id=r['candidate_id'],
                                    scene_type=r['scene_type'], score_rank=r['score_rank'], mixed_rank=r['mixed_rank'],
                                    total_score=r['total_score'], retained=r['retained'])
                               for r in changed if r['scene_type'] == scene and r['candidate_id'] in wanted)
        write_csv(self.output / 'stability_summary.csv', results)
        write_csv(self.output / 'stability_display.csv', details)

    def historical_comparison(self, formal_rows):
        history = json.loads((self.output / 'preparation/archive_score_regression.json').read_text(encoding='utf-8'))
        rows, summaries, comparisons = [], [], []
        for scene, archive in history.items():
            replay = []
            for branch in COMPONENTS:
                for value in archive['replay_scores'][branch]:
                    metric = value['metrics']
                    row = dict(scene_key=scene, scene_type=branch, candidate_id=value['candidate_id'],
                               total_score=value['total'],
                               distance_value_km=metric['proximity_d_min_km' if branch == 'spacecraft'
                                                        else 'ground_track_min_surface_distance_km'])
                    row.update({'legacy_' + k: v for k, v in metric.items()})
                    for short, name, weight in COMPONENTS[branch]:
                        row['S_' + short] = value['components'][name]
                        row['wS_' + short] = weight * row['S_' + short]
                    replay.append(row)
            cutoff = rank_pool(replay)
            rows.extend(replay)
            for branch in COMPONENTS:
                subset = [r for r in replay if r['scene_type'] == branch]
                summaries.append(dict(scene=scene, scene_type=branch, grid_step_s=300, candidates=len(subset),
                                      retained=sum(r['retained'] for r in subset), mixed_pool_count=len(replay),
                                      mixed_score_cutoff=cutoff,
                                      same_budget_overlap=sum(r['retained'] and r['distance_retained'] for r in subset)))
            formal = [r for r in formal_rows if r['scene_key'] == scene]
            old_set = {(r['scene_type'], r['candidate_id']) for r in replay if r['retained']}
            new_set = {(r['scene_type'], r['candidate_id']) for r in formal if r['retained']}
            comparisons.append(dict(scene=scene, historical_step_s=300, formal_step_s=60,
                                    historical_top40_matches_saved=archive['exact_top40_set_match'],
                                    formal_historical_top40_overlap=len(old_set & new_set),
                                    formal_new_candidates=len(new_set - old_set),
                                    interpretation='Fixed historical formula; trajectories and sampling differ between the two archived data stages.'))
        write_csv(self.output / 'historical_all_candidate_results.csv', rows)
        write_csv(self.output / 'historical_screening_summary.csv', summaries)
        write_csv(self.output / 'time_grid_comparison.csv', comparisons)

    def refresh_manifest(self):
        out = self.output
        files = {str(p.relative_to(out)): digest(p) for p in sorted(out.rglob('*'))
                 if p.is_file() and p.name != 'manifest.json' and '__pycache__' not in str(p)}
        scripts = list((ROOT / 'scripts').glob('matching_legacy_*.py'))
        scripts.extend(ROOT / 'scripts' / name for name in ('orbit_core.py', 'data_processing.py',
                       'matching_validation_frames.py', 'matching_validation_metrics.py', 'matching_validation_run.py'))
        dump(out / 'manifest.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
                                        experiment='Legacy Matching Score case analysis on frozen 10400 subset',
                                        frame=FRAME_METADATA, files_sha256=files,
                                        script_sha256={p.name: digest(p) for p in sorted(scripts)},
                                        execution_order=['matching_legacy_prepare.py (intention-ir)',
                                                         'matching_legacy_analyze.py (base)',
                                                         'matching_legacy_plot_spacecraft.py (base, OUTPUT_MODE=savefig)',
                                                         'matching_legacy_plot_regions.py (base, OUTPUT_MODE=savefig)',
                                                         'matching_legacy_plot_screening_stability.py (base, OUTPUT_MODE=savefig)'],
                                        formula_scope='Legacy reproduction, not the current paper formula',
                                        frozen_data_changed=False))


if __name__ == '__main__':
    MatchingLegacyAnalysis().run()
