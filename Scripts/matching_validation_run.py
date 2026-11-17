"""Compute/export the fixed-scene matching validation; run in base Anaconda.

Reuse prepared display states after matching_validation_calibrate.py freezes
parameters. Change OUTPUT in main for a new run; existing results are protected.
"""

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from matching_validation_frames import (FRAME_METADATA, actor_rtn_basis, relative_rtn,
                                       spherical_ground_track, teme_to_j2000)
from matching_validation_metrics import compute_metrics, score_metrics


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def utc(seconds):
    return datetime.fromtimestamp(float(seconds), timezone.utc).isoformat().replace('+00:00', 'Z')


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError('Cannot export an empty table: ' + str(path))
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def display_candidates(rows, scene, original):
    """Choose physical contrasts before using any matching-score results."""
    selected = {}

    def add(row, reason):
        selected.setdefault(row['candidate_id'], []).append(reason)

    ordered = sorted(rows, key=lambda row: row['candidate_id'])
    add(next(row for row in ordered if row['candidate_id'] == original),
        'Original frozen pair; spacecraft has a pre-existing design reference, ground is a source pair only.')
    distance = 'd_min_F_km' if scene == 'spacecraft' else 'ell_min_F_km'
    add(min(ordered, key=lambda row: row[distance]), 'Smallest factual distance in the complete valid pool.')
    if scene == 'spacecraft':
        add(min(ordered, key=lambda row: row['plane_angle_deg']), 'Smallest mean orbital-plane angle.')
        add(max(ordered, key=lambda row: row['G_d_km']), 'Largest positive post-event distance gain.')
        add(max(ordered, key=lambda row: row[distance]), 'Largest minimum separation; distance reference.')
        close = sorted(ordered, key=lambda row: row[distance])[:max(1, len(ordered) // 4)]
        if len(selected) < 5:
            add(max(close, key=lambda row: row['relative_speed_at_ca_F_km_s']),
                'Largest closest-approach relative speed within the nearest-distance quartile.')
    else:
        add(max(ordered, key=lambda row: row['T_access_F_s']), 'Longest factual region-center access.')
        add(max(ordered, key=lambda row: row['G_T_s']), 'Largest positive post-event access gain.')
        no_access = [row for row in ordered if row['T_access_F_s'] == 0]
        if no_access:
            add(max(no_access, key=lambda row: row[distance]), 'No access and largest minimum surface distance.')
        if len(selected) < 5:
            add(max(ordered, key=lambda row: row['N_pass_F']), 'Largest number of separate access intervals.')
    assert 3 <= len(selected) <= 5
    return selected


class MatchingValidationRun:
    def __init__(self, output):
        self.output = Path(output)

    def run(self):
        out = self.output
        if (out / 'matching_metrics.csv').exists():
            raise FileExistsError('Results already exist. Prepare a new version directory and set OUTPUT to it; '
                                  'copy its scoring_parameters.json before calculation.')
        source = json.loads((out / 'preparation/scenes.json').read_text(encoding='utf-8'))
        parameters = json.loads((out / 'scoring_parameters.json').read_text(encoding='utf-8'))
        raw_path = out / 'preparation/raw_states.npz'
        if digest(raw_path) != source['raw_states_sha256']:
            raise ValueError('Prepared states have changed since source verification.')
        if digest(out / 'candidate_pool.csv') != source['candidate_pool_sha256']:
            raise ValueError('Candidate-pool metadata changed since preparation.')
        with (out / 'candidate_pool.csv').open(encoding='utf-8-sig', newline='') as stream:
            pool = {(row['scene'], row['candidate_id']): row for row in csv.DictReader(stream)}
        all_metrics, all_components, selected_rows, summary, checks = [], [], [], [], {}
        arrays, analysis, timeseries_rows, trajectory_rows = {}, {'parameters': parameters, 'scenes': {}}, [], []
        ground_track_rows = []
        with np.load(raw_path, allow_pickle=False) as raw:
            for scene, meta in source['scenes'].items():
                times = raw[scene + '_times_s']
                ids = raw[scene + '_ids'].tolist()
                f = teme_to_j2000(times, raw[scene + '_actor_f'])
                cf = teme_to_j2000(times, raw[scene + '_actor_cf'])
                candidates = teme_to_j2000(times, raw[scene + '_candidates'])
                basis = actor_rtn_basis(f)
                cf_rtn = relative_rtn(cf[:, :3], f[:, :3], basis)
                c_rtn = relative_rtn(candidates[..., :3], f[:, :3], basis)
                distance_error = float(np.max(np.abs(np.linalg.norm(c_rtn, axis=-1)
                                                     - np.linalg.norm(candidates[..., :3] - f[:, :3], axis=-1))))
                assert distance_error < 1e-8
                assert np.allclose(basis @ basis.swapaxes(-1, -2), np.eye(3), rtol=0, atol=1e-14)
                event = meta['maneuver_events'][0]
                event_time = datetime.fromisoformat(event['t_maneuver'].replace('Z', '+00:00')).timestamp()
                error = meta['error_summary']
                metric_meta = dict(scene_id=meta['scene_id'], scene_type=scene,
                                   maneuver_time_unix_s=event_time,
                                   sigma_r_km=error.get('orbit_position_sigma'),
                                   sigma_tm_s=error.get('maneuver_time_sigma'))
                metrics, series = compute_metrics(times, f, cf, candidates, ids, metric_meta, parameters)
                selection = display_candidates(metrics, scene, meta['source_counterpart_id'])
                components = score_metrics(metrics, scene, parameters)
                components_by_id = {row['candidate_id']: row for row in components}
                display = []
                for row in metrics:
                    identity = row['candidate_id']
                    row.update(candidate_name=pool[(scene, identity)]['candidate_name'], actor_id=meta['actor_id'],
                               sample_index=meta['sample_index'], event_time_sigma_s=event.get('time_sigma'),
                               counterpart_covariance_available=False)
                    if identity in selection:
                        code = ('T' if scene == 'spacecraft' else 'R') + str(list(selection).index(identity) + 1)
                        display.append(dict(candidate_id=identity, candidate_name=row['candidate_name'],
                                            array_index=ids.index(identity), label=code,
                                            selection_reason=' '.join(selection[identity]),
                                            metric=row, component=components_by_id[identity]))
                        selected_rows.append(dict(scene=scene, scene_key=meta['scene_key'], sample_index=meta['sample_index'],
                                                  actor_id=meta['actor_id'], actor_name=meta['actor_name'],
                                                  window_start_utc=meta['window_start_utc'], window_end_utc=meta['window_end_utc'],
                                                  maneuver_utc=event['t_maneuver'], display_label=code,
                                                  candidate_id=identity, candidate_name=row['candidate_name'],
                                                  selection_reason=' '.join(selection[identity])))
                display.sort(key=lambda item: item['label'])
                all_metrics.extend(metrics)
                all_components.extend(components)
                arrays.update({scene + '_times_s': times, scene + '_actor_f': f, scene + '_actor_cf': cf,
                               scene + '_candidates': candidates, scene + '_ids': np.asarray(ids),
                               scene + '_actor_cf_rtn': cf_rtn, scene + '_candidate_rtn': c_rtn,
                               scene + '_actor_f_ground': spherical_ground_track(times, f[:, :3]),
                               scene + '_actor_cf_ground': spherical_ground_track(times, cf[:, :3])})
                if scene == 'regions':
                    for branch in ('f', 'cf'):
                        for timestamp, latlon in zip(times, arrays[scene + '_actor_' + branch + '_ground']):
                            ground_track_rows.append(dict(scene=scene, actor_id=meta['actor_id'], branch=branch.upper(),
                                                          utc=utc(timestamp), time_offset_s=float(timestamp-times[0]),
                                                          latitude_deg=float(latlon[0]), longitude_deg=float(latlon[1])))
                array_fields = {key: value for key, value in series.items()
                                if isinstance(value, np.ndarray) and value.shape == (len(ids), len(times))}
                for i, identity in enumerate(ids):
                    for j, timestamp in enumerate(times):
                        row = dict(scene=scene, scene_id=meta['scene_id'], candidate_id=identity,
                                   utc=utc(timestamp), time_offset_s=float(timestamp-times[0]),
                                   node_duration_s=float(series['node_duration_s'][j]),
                                   post_node_duration_s=float(series['post_node_duration_s'][j]))
                        for key, values in array_fields.items():
                            value = values[i, j].item()
                            row[key] = value if isinstance(value, bool) or np.isfinite(value) else None
                        timeseries_rows.append(row)
                objects = [('actor_F', meta['actor_id'], f, np.zeros_like(cf_rtn)),
                           ('actor_CF', meta['actor_id'], cf, cf_rtn)]
                objects.extend(('counterpart', identity, candidates[i], c_rtn[i]) for i, identity in enumerate(ids))
                for role, identity, states, rtn in objects:
                    for j, timestamp in enumerate(times):
                        row = dict(scene=scene, object_role=role, object_id=identity, utc=utc(timestamp),
                                   time_offset_s=float(timestamp-times[0]), frame='J2000',
                                   lvlh_origin='actor_F:' + meta['actor_id'], lvlh_axes='RTN',
                                   is_maneuver_node=bool(timestamp == event_time))
                        row.update(zip(('x_km','y_km','z_km','vx_km_s','vy_km_s','vz_km_s'), states[j].tolist()))
                        row.update(zip(('R_km','T_km','N_km'), rtn[j].tolist()))
                        trajectory_rows.append(row)
                retained = None if any(row['retained'] is None for row in components) else sum(row['retained'] for row in components)
                ref = components_by_id.get('43029') if scene == 'spacecraft' else None
                summary.append(dict(scene=scene, actor_id=meta['actor_id'], source_candidates=meta['source_candidate_count'],
                                    data_quality_excluded=meta['data_quality_excluded_count'], candidates=len(ids),
                                    retained=retained, retained_fraction=None if retained is None else retained/len(ids),
                                    reduction_fraction=None if retained is None else 1-retained/len(ids),
                                    independently_referenced_count=1 if ref else None,
                                    independently_referenced_retained=None if ref is None or retained is None else int(ref['retained']),
                                    distance_only_reference_retained=None if ref is None or retained is None else int(ref['distance_retained']),
                                    status='parameters_pending' if retained is None else 'calculated'))
                event_index = meta['event_grid_index']
                analysis['scenes'][scene] = dict(meta, event_offset_min=(event_time-times[0])/60, display=display,
                                               event_delta_v_j2000_km_s=(f[event_index, 3:]-cf[event_index, 3:]).tolist())
                checks[scene] = dict(rtn_distance_max_error_km=distance_error,
                                     node_duration_total_s=float(series['node_duration_s'].sum()),
                                     post_duration_total_s=float(series['post_node_duration_s'].sum()),
                                     actor_f_rtn_is_zero=True, full_pool_processed=len(ids),
                                     same_actor_for_all_candidates=True)
        write_csv(out/'matching_metrics.csv', all_metrics)
        write_csv(out/'matching_components.csv', all_components)
        write_csv(out/'selected_cases.csv', selected_rows)
        write_csv(out/'matching_screening_summary.csv', summary)
        write_csv(out/'matching_timeseries.csv', timeseries_rows)
        write_csv(out/'matching_trajectories_3d.csv', trajectory_rows)
        write_csv(out/'matching_ground_tracks.csv', ground_track_rows)
        write_csv(out/'relevance_reference.csv', [
            dict(scene='spacecraft', candidate_id='43029', reference_kind='generation_design_counterpart',
                 known_relevant_set_complete=False, source='sample_provenance_frozen.json:17985 scenario_id and selection_reason',
                 determined_at='Generation before 2026-09-10 freeze; exact creation time not retained',
                 meaning='Independent of this score; design reference, not observed intent ground truth'),
            dict(scene='regions', candidate_id=None, reference_kind='unavailable', known_relevant_set_complete=False,
                 source='No independently predefined region set retained', determined_at=None,
                 meaning='Original frozen pair suffix does not establish relevance; unlabelled candidates remain unknown')])
        np.savez_compressed(out/'analysis_states.npz', **arrays)
        (out/'analysis.json').write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding='utf-8')
        (out/'validation_checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
        manifest = dict(status=parameters['status'], created_utc=datetime.now(timezone.utc).isoformat(),
                        formula_source=parameters['formula_source'], frame=FRAME_METADATA,
                        parameters_sha256=digest(out/'scoring_parameters.json'), source_sha256=source['source_sha256'],
                        files_sha256={p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'},
                        script_sha256={p.name:digest(p) for p in (ROOT/'scripts').glob('matching_validation_*.py')},
                        frozen_source_integrity='Verified against freeze manifest before and after preparation; no writes to frozen inputs.',
                        validation=checks, source_scope='478 objects in the retained China LEO catalogue; 22 fixed project ground centers',
                        observations='Retrospective deterministic scenarios; both display actors were exposed during dataset generation development.',
                        execution_order=['matching_validation_prepare.py (intention-ir)', 'matching_validation_run.py (base)',
                                         'matching_validation_plot.py (base)'])
        (out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(dict(output=str(out), summary=summary, selected=selected_rows), ensure_ascii=False, indent=2))
        return analysis


if __name__ == '__main__':
    OUTPUT = ROOT / 'experiment_data/matching_score_validation_20261019_v2'
    MatchingValidationRun(OUTPUT).run()
