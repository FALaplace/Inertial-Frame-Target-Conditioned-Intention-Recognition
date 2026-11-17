"""Matching-score physics and scoring for fixed-actor candidate pools.

Times are UTC Unix seconds; states have columns X,Y,Z,VX,VY,VZ in km and
km/s in one common inertial frame. No files, models, or global state are changed.
Run this file directly for deterministic checks using synthetic states only.
"""

import math

import numpy as np


COMPONENTS = {
    'spacecraft': ('prox', 'geom', 'reach', 'man', 'unc'),
    'regions': ('ground', 'access', 'revisit', 'cov', 'unc'),
}


def _number(value, name, positive=False):
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError('{} must be finite and {}'.format(name, 'positive' if positive else 'nonnegative'))
    return value


def node_durations(times_unix_s, start_unix_s=None):
    """Midpoint/Voronoi node cells clipped to [start, window_end]."""
    times = np.asarray(times_unix_s, dtype=np.float64)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Times must contain at least two finite, strictly increasing nodes.')
    edges = np.r_[times[0], times[:-1] + np.diff(times) / 2, times[-1]]
    start = times[0] if start_unix_s is None else float(start_unix_s)
    if not math.isfinite(start):
        raise ValueError('Post-event start must be finite.')
    start = np.clip(start, times[0], times[-1])
    return np.maximum(0, edges[1:] - np.maximum(edges[:-1], start))


def _angles(a, b):
    numerator = np.sum(a * b, axis=-1)
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    if np.any(denominator <= 1e-12):
        raise ValueError('Angles require nonzero vectors.')
    return np.degrees(np.arccos(np.clip(numerator / denominator, -1, 1)))


def compute_metrics(times_unix_s, actor_f, actor_cf, candidates, ids, scene_meta, parameters):
    """Return (candidate summary rows, array-valued time series).

    scene_meta: scene_id, scene_type ('spacecraft'/'regions'),
    maneuver_time_unix_s (None when absent), sigma_r_km, sigma_tm_s.
    Series have shape (M,T), except timestamps/duration weights (T) and IDs (M).
    Geometry means use the same duration weights as dwell and access integrals.
    """
    times = np.asarray(times_unix_s, dtype=np.float64)
    weights = node_durations(times)
    actor_f, actor_cf = np.asarray(actor_f, dtype=np.float64), np.asarray(actor_cf, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=np.float64)
    ids = [str(value) for value in ids]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Candidate IDs must be nonempty and unique.')
    for name, states, shape in (
        ('actor_f', actor_f, (len(times), 6)),
        ('actor_cf', actor_cf, (len(times), 6)),
        ('candidates', candidates, (len(ids), len(times), 6)),
    ):
        if states.shape != shape or not np.isfinite(states).all():
            raise ValueError('{} must be finite with shape {}.'.format(name, shape))
        if np.any(np.linalg.norm(states[..., :3], axis=-1) <= 1e-12):
            raise ValueError('{} contains a zero geocentric position.'.format(name))
    scene_type = scene_meta['scene_type']
    if scene_type not in COMPONENTS:
        raise ValueError('scene_type must be spacecraft or regions.')
    event = scene_meta.get('maneuver_time_unix_s')
    if event is None and not np.array_equal(actor_f, actor_cf):
        raise ValueError('Without a maneuver, factual and counterfactual states must be identical.')
    post_weights = np.zeros_like(weights) if event is None else node_durations(times, event)
    post_nodes = np.zeros(len(times), dtype=bool) if event is None else times >= float(event)
    sigma_r = _number(scene_meta.get('sigma_r_km'), 'sigma_r_km')
    sigma_tm = _number(scene_meta.get('sigma_tm_s'), 'sigma_tm_s')
    shared, branch = parameters.get('shared', {}), parameters.get(scene_type, {})
    earth_radius = _number(shared.get('earth_radius_km'), 'earth_radius_km', positive=True)
    mu = _number(shared.get('mu_km3_s2'), 'mu_km3_s2', positive=True)
    dwell_threshold = _number(branch.get('dwell_distance_threshold_km'), 'dwell_distance_threshold_km', positive=True)
    elevation_threshold = branch.get('elevation_threshold_deg')
    if elevation_threshold is not None:
        elevation_threshold = float(elevation_threshold)
        if not math.isfinite(elevation_threshold) or not -90 <= elevation_threshold <= 90:
            raise ValueError('Elevation threshold must be in [-90,90] degrees.')
    series = {
        'time_unix_s': times,
        'time_offset_s': times - times[0],
        'node_duration_s': weights,
        'post_node_duration_s': post_weights,
        'candidate_id': ids,
    }
    for suffix, actor in (('F', actor_f), ('CF', actor_cf)):
        relative_r = actor[None, :, :3] - candidates[:, :, :3]
        relative_v = actor[None, :, 3:] - candidates[:, :, 3:]
        distance = np.linalg.norm(relative_r, axis=-1)
        range_rate = np.divide(np.sum(relative_r * relative_v, axis=-1), distance,
                               out=np.full_like(distance, np.nan), where=distance > 0)
        series['distance_{}_km'.format(suffix)] = distance
        series['range_rate_{}_km_s'.format(suffix)] = range_rate
        series['range_rate_{}_valid'.format(suffix)] = distance > 0
        series['relative_speed_{}_km_s'.format(suffix)] = np.linalg.norm(relative_v, axis=-1)
        if scene_type == 'regions':
            region_r = candidates[:, :, :3]
            radius = np.linalg.norm(region_r, axis=-1)
            sin_elevation = np.divide(np.sum(relative_r * region_r, axis=-1), distance * radius,
                                      out=np.full_like(distance, np.nan), where=distance > 0)
            elevation = np.degrees(np.arcsin(np.clip(sin_elevation, -1, 1)))
            series['ell_{}_km'.format(suffix)] = (None if earth_radius is None else
                                                   np.radians(_angles(actor[None, :, :3], region_r)) * earth_radius)
            series['elevation_{}_deg'.format(suffix)] = elevation
            series['access_{}'.format(suffix)] = (None if elevation_threshold is None else
                                                  elevation >= elevation_threshold)

    geometry = {}
    if scene_type == 'spacecraft':
        average_weights = weights / weights.sum()
        radius_a = np.dot(np.linalg.norm(actor_f[:, :3], axis=-1), average_weights)
        radius_c = np.linalg.norm(candidates[:, :, :3], axis=-1) @ average_weights
        speed_a = np.dot(np.linalg.norm(actor_f[:, 3:], axis=-1), average_weights)
        h_a, h_c = np.cross(actor_f[:, :3], actor_f[:, 3:]), np.cross(candidates[:, :, :3], candidates[:, :, 3:])
        z = np.array([0.0, 0.0, 1.0])
        inclination_a = np.dot(_angles(h_a, z), average_weights)
        inclination_c = _angles(h_c, z) @ average_weights
        plane = _angles(h_a[None], h_c) @ average_weights
        phase = _angles(actor_f[None, :, :3], candidates[:, :, :3]) @ average_weights
        dv_plane = 2 * speed_a * np.sin(np.radians(plane) / 2)
        dv_alt = None if mu is None else (
            np.abs(np.sqrt(mu / radius_a) * (np.sqrt(2 * radius_c / (radius_a + radius_c)) - 1))
            + np.abs(np.sqrt(mu / radius_c) * (1 - np.sqrt(2 * radius_a / (radius_a + radius_c))))
        )
        geometry = {
            'actor_mean_radius_km': np.full(len(ids), radius_a),
            'counterpart_mean_radius_km': radius_c,
            'actor_mean_speed_km_s': np.full(len(ids), speed_a),
            'actor_mean_inclination_deg': np.full(len(ids), inclination_a),
            'counterpart_mean_inclination_deg': inclination_c,
            'delta_mean_radius_km': np.abs(radius_c - radius_a),
            'delta_inclination_deg': np.abs(inclination_c - inclination_a),
            'plane_angle_deg': plane, 'phase_angle_deg': phase,
            'dv_alt_km_s': dv_alt, 'dv_plane_km_s': dv_plane,
            'dv_req_km_s': None if dv_alt is None else dv_alt + dv_plane,
        }
    rows = []
    for i, candidate_id in enumerate(ids):
        row = {
            'scene_id': scene_meta['scene_id'], 'scene_type': scene_type,
            'candidate_id': candidate_id, 'has_maneuver': event is not None,
            'maneuver_time_unix_s': event, 'sigma_r_km': sigma_r, 'sigma_tm_s': sigma_tm,
            'window_duration_s': float(weights.sum()), 'post_duration_s': float(post_weights.sum()),
        }
        for suffix in ('F', 'CF'):
            distance = series['distance_{}_km'.format(suffix)][i]
            closest = int(np.argmin(distance))
            rate = series['range_rate_{}_km_s'.format(suffix)][i, closest]
            row.update({
                'd_min_{}_km'.format(suffix): float(distance[closest]),
                't_ca_{}_offset_s'.format(suffix): float(times[closest] - times[0]),
                'range_rate_at_ca_{}_km_s'.format(suffix): float(rate) if np.isfinite(rate) else None,
                'relative_speed_at_ca_{}_km_s'.format(suffix): float(series['relative_speed_{}_km_s'.format(suffix)][i, closest]),
                'd_min_{}_post_km'.format(suffix): float(np.min(distance[post_nodes])) if np.any(post_nodes) else None,
                'T_dwell_{}_s'.format(suffix): float(np.dot(distance < dwell_threshold, weights)) if dwell_threshold is not None else None,
            })
            if scene_type == 'regions':
                ell, access = series['ell_{}_km'.format(suffix)], series['access_{}'.format(suffix)]
                row.update({
                    'ell_min_{}_km'.format(suffix): None if ell is None else float(np.min(ell[i])),
                    'elevation_max_{}_deg'.format(suffix): float(np.max(series['elevation_{}_deg'.format(suffix)][i])),
                    'T_access_{}_s'.format(suffix): None if access is None else float(np.dot(access[i], weights)),
                    'T_access_{}_post_s'.format(suffix): None if access is None else float(np.dot(access[i], post_weights)),
                    'N_pass_{}'.format(suffix): None if access is None else int(np.count_nonzero(access[i] & ~np.r_[False, access[i, :-1]])),
                })
        for name, values in geometry.items():
            row[name] = None if values is None else float(values[i])
        row['G_d_km'] = (0.0 if event is None else None if row['d_min_F_post_km'] is None else
                         max(0.0, row['d_min_CF_post_km'] - row['d_min_F_post_km']))
        if scene_type == 'regions':
            row['G_T_s'] = (0.0 if event is None else None if row['T_access_F_post_s'] is None else
                             max(0.0, row['T_access_F_post_s'] - row['T_access_CF_post_s']))
        rows.append(row)
    distance_field = 'd_min_F_km' if scene_type == 'spacecraft' else 'ell_min_F_km'
    if all(row[distance_field] is not None for row in rows):
        for rank, row in enumerate(sorted(rows, key=lambda row: (row[distance_field], row['candidate_id'])), 1):
            row['distance_rank'] = rank
    else:
        for row in rows:
            row['distance_rank'] = None
    return rows, series


def _transform(value, scale, increasing=False):
    value, scale = _number(value, 'physical metric'), _number(scale, 'score scale', positive=True)
    if value is None or scale is None:
        return None
    return -math.expm1(-value / scale) if increasing else math.exp(-value / scale)


def _weights(values, names):
    weights = [_number(values.get(name), 'weight ' + name) for name in names]
    known_sum = sum(value for value in weights if value is not None)
    if known_sum > 1 + 1e-12 or (None not in weights and not math.isclose(known_sum, 1, abs_tol=1e-12)):
        raise ValueError('Known weight group must be nonnegative and sum to one.')
    return weights


def _weighted_sum(values, weights):
    if any(weight is None or (weight != 0 and value is None) for value, weight in zip(values, weights)):
        return None
    return sum(0.0 if weight == 0 else value * weight for value, weight in zip(values, weights))


def score_metrics(metrics, scene_type, parameters):
    """Return five components/contributions, rankings, and equal-budget retention.

    Missing parameters propagate as None. Ranking requires all candidate totals;
    threshold retention additionally requires shared.score_threshold.
    """
    if scene_type not in COMPONENTS:
        raise ValueError('scene_type must be spacecraft or regions.')
    names = COMPONENTS[scene_type]
    shared, branch = parameters.get('shared', {}), parameters.get(scene_type, {})
    scales = branch.get('scales', {})
    outer = _weights(branch.get('weights', {}), names)
    threshold = _number(shared.get('score_threshold'), 'score_threshold')
    if threshold is not None and threshold > 1:
        raise ValueError('Score threshold must be in [0,1].')
    if scene_type == 'spacecraft':
        beta = _weights(branch.get('beta', {}), ('distance', 'dwell'))
        gamma = _weights(branch.get('gamma', {}), ('radius', 'inclination', 'plane', 'phase'))
    rows = []
    for metric in metrics:
        if metric['scene_type'] != scene_type:
            raise ValueError('Cannot mix counterpart types in one ranking.')
        values = {'unc': _transform(metric.get('sigma_r_km'), shared.get('position_sigma_scale_km'))}
        if scene_type == 'spacecraft':
            values.update({
                'prox': _weighted_sum([
                    _transform(metric.get('d_min_F_km'), scales.get('distance_km')),
                    _transform(metric.get('T_dwell_F_s'), scales.get('dwell_s'), True)], beta),
                'geom': _weighted_sum([
                    _transform(metric.get(key), scales.get(scale)) for key, scale in (
                        ('delta_mean_radius_km', 'radius_km'), ('delta_inclination_deg', 'inclination_deg'),
                        ('plane_angle_deg', 'plane_deg'), ('phase_angle_deg', 'phase_deg'))], gamma),
                'reach': _transform(metric.get('dv_req_km_s'), scales.get('delta_v_km_s')),
            })
            gain_name, gain_metric, gain_scale = 'man', 'G_d_km', 'gain_km'
        else:
            values.update({
                'ground': _transform(metric.get('ell_min_F_km'), scales.get('surface_distance_km')),
                'access': _transform(metric.get('T_access_F_s'), scales.get('access_s'), True),
                'revisit': _transform(metric.get('N_pass_F'), scales.get('pass_count'), True),
            })
            gain_name, gain_metric, gain_scale = 'cov', 'G_T_s', 'gain_s'
        gain = _transform(metric.get(gain_metric), scales.get(gain_scale), True)
        timing = _transform(metric.get('sigma_tm_s'), shared.get('maneuver_time_sigma_scale_s'))
        values[gain_name] = (0.0 if not metric['has_maneuver'] else
                            None if gain is None or timing is None else gain * timing)
        row = {key: metric[key] for key in ('scene_id', 'scene_type', 'candidate_id', 'distance_rank')}
        row.update({'score_rank': None, 'retained': None, 'distance_retained': None})
        for name, weight in zip(names, outer):
            value = values[name]
            if value is not None and not 0 <= value <= 1:
                raise ValueError('Component outside [0,1].')
            row['S_' + name] = value
            row['wS_' + name] = (None if weight is None else 0.0 if weight == 0 else
                                  None if value is None else weight * value)
        row['total_score'] = _weighted_sum([values[name] for name in names], outer)
        rows.append(row)
    if len({row['scene_id'] for row in rows}) > 1 or len({row['candidate_id'] for row in rows}) != len(rows):
        raise ValueError('One ranking requires a single scene and unique candidate IDs.')
    if rows and all(row['total_score'] is not None for row in rows):
        for rank, row in enumerate(sorted(rows, key=lambda row: (-row['total_score'], row['candidate_id'])), 1):
            row['score_rank'] = rank
            if threshold is not None:
                row['retained'] = row['total_score'] >= threshold
        if threshold is not None and all(row['distance_rank'] is not None for row in rows):
            count = sum(row['retained'] for row in rows)
            for row in rows:
                row['distance_retained'] = row['distance_rank'] <= count
    return rows


def self_check():
    """Synthetic checks only; these numbers are not experimental parameters."""
    times = np.arange(181, dtype=float) * 60
    weights = node_durations(times)
    assert weights.sum() == 10800 and weights[0] == weights[-1] == 30
    assert node_durations(times, 65.25 * 60).sum() == 10800 - 65.25 * 60
    assert node_durations(times, -10).sum() == 10800
    assert node_durations(times, 20000).sum() == 0
    angle = times * 0.001
    actor = np.column_stack((7000 * np.cos(angle), 7000 * np.sin(angle), np.zeros(181),
                             -7 * np.sin(angle), 7 * np.cos(angle), np.zeros(181)))
    candidates = np.stack((actor.copy(), actor.copy()))
    candidates[:, :, 2] += 1
    parameters = {
        'shared': {'earth_radius_km': 6378.137, 'mu_km3_s2': 398600.4418,
                   'position_sigma_scale_km': 1, 'maneuver_time_sigma_scale_s': 1, 'score_threshold': 0.1},
        'spacecraft': {'dwell_distance_threshold_km': 2,
                       'weights': dict(zip(COMPONENTS['spacecraft'], [.3, .2, .2, .2, .1])),
                       'beta': {'distance': .5, 'dwell': .5},
                       'gamma': dict(zip(('radius', 'inclination', 'plane', 'phase'), [.25] * 4)),
                       'scales': dict.fromkeys(('distance_km', 'dwell_s', 'radius_km', 'inclination_deg',
                                               'plane_deg', 'phase_deg', 'delta_v_km_s', 'gain_km'), 1)},
        'regions': {'elevation_threshold_deg': 10,
                    'weights': dict(zip(COMPONENTS['regions'], [.3, .25, .2, .15, .1])),
                    'scales': dict.fromkeys(('surface_distance_km', 'access_s', 'pass_count', 'gain_s'), 1)},
    }
    meta = {'scene_id': 'synthetic', 'scene_type': 'spacecraft', 'maneuver_time_unix_s': None,
            'sigma_r_km': 1, 'sigma_tm_s': None}
    rows, series = compute_metrics(times, actor, actor, candidates, ['B', 'A'], meta, parameters)
    assert [row['distance_rank'] for row in rows] == [2, 1]
    assert all(row['T_dwell_F_s'] == 10800 and row['G_d_km'] == 0 for row in rows)
    scored = score_metrics(rows, 'spacecraft', parameters)
    assert all(row['S_man'] == 0 and 0 <= row['total_score'] <= 1 for row in scored)
    assert [row['score_rank'] for row in scored] == [2, 1]
    assert all(math.isclose(sum(row['wS_' + name] for name in COMPONENTS['spacecraft']), row['total_score']) for row in scored)
    parameters['shared']['score_threshold'] = scored[0]['total_score']
    scored = score_metrics(rows, 'spacecraft', parameters)
    assert all(row['retained'] for row in scored)
    assert sum(row['retained'] for row in scored) == sum(row['distance_retained'] for row in scored)
    parameters['spacecraft']['scales']['distance_km'] = None
    scored = score_metrics(rows, 'spacecraft', parameters)
    assert all(row['S_prox'] is None and row['total_score'] is None and row['retained'] is None for row in scored)
    parameters['spacecraft']['dwell_distance_threshold_km'] = None
    rows, _ = compute_metrics(times, actor, actor, candidates, ['B', 'A'], meta, parameters)
    assert all(row['T_dwell_F_s'] is None for row in rows)
    parameters['spacecraft']['scales']['distance_km'] = 1
    parameters['spacecraft']['dwell_distance_threshold_km'] = 2
    candidates[1, :, 2] = 2
    rows, _ = compute_metrics(times, actor, actor, candidates, ['B', 'A'], meta, parameters)
    scored = score_metrics(rows, 'spacecraft', parameters)
    parameters['shared']['score_threshold'] = sum(row['total_score'] for row in scored) / 2
    scored = score_metrics(rows, 'spacecraft', parameters)
    assert sum(row['retained'] for row in scored) == sum(row['distance_retained'] for row in scored) == 1
    for threshold, expected_count in ((0, 2), (1, 0)):
        parameters['shared']['score_threshold'] = threshold
        scored = score_metrics(rows, 'spacecraft', parameters)
        assert sum(row['retained'] for row in scored) == sum(row['distance_retained'] for row in scored) == expected_count
    meta.update(scene_type='regions', maneuver_time_unix_s=65.25 * 60)
    candidates[:, :, :3] *= 6378.137 / np.linalg.norm(candidates[:, :, :3], axis=-1, keepdims=True)
    rows, _ = compute_metrics(times, actor, actor, candidates, ['B', 'A'], meta, parameters)
    assert all(row['T_access_F_s'] == 10800 and row['T_access_F_post_s'] == 6885 for row in rows)
    assert all(row['N_pass_F'] == 1 and row['G_T_s'] == 0 for row in rows)
    candidates[:, :, :3] *= -1
    candidates[:, 20:40, :3] *= -1
    candidates[:, 80:100, :3] *= -1
    rows, _ = compute_metrics(times, actor, actor, candidates, ['B', 'A'], meta, parameters)
    assert all(row['N_pass_F'] == 2 and row['T_access_F_s'] == 2400 and row['T_access_F_post_s'] == 1200 for row in rows)
    meta['maneuver_time_unix_s'] = None
    rows, _ = compute_metrics(times, actor, actor, candidates, ['B', 'A'], meta, parameters)
    assert all(row['S_cov'] == 0 for row in score_metrics(rows, 'regions', parameters))
    print('Matching metrics self-check passed.')


if __name__ == '__main__':
    self_check()
