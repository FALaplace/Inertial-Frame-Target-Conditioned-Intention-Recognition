"""Offline TEME-to-J2000 states and actor-centered RTN positions.

Run in the base Anaconda environment, which provides ERFA. State arrays use
km and km/s; Unix times are UTC seconds. No files or network are accessed.
"""

from datetime import datetime, timezone

import erfa
import numpy as np


TT_MINUS_UTC_S = 69.184
FRAME_METADATA = {
    "output_frame": "Geocentric mean equator and equinox J2000 (IAU 2006 frame bias)",
    "input_frame": "Stored TEME-compatible simulation coordinates; SGP4 TEME before impulse",
    "transform": "IAU 2006/2000A celestial matrix, GMST82, ERA00, TIO locator, J2000 bias",
    "velocity_transform": "v_J2000 = M v_TEME + Mdot r_TEME; centered 1-second difference",
    "ut1_minus_utc_s": 0.0,
    "ut1_assumption": "UT1=UTC; polar motion cancels in this purely celestial frame change",
    "tt_minus_utc_s": TT_MINUS_UTC_S,
    "time_validity": "2026-01-01 through 2026-12-31; TAI-UTC=37 s",
    "time_source": "https://datacenter.iers.org/data/16/bulletinc-072.txt",
    "frame_source": "https://docs.astropy.org/en/stable/api/astropy.coordinates.TEME.html",
    "j2000_source": "https://docs.astropy.org/en/stable/api/astropy.coordinates.PrecessedGeocentric.html",
    "earth_model": "Original spherical Earth, radius 6378.137 km, GMST(UTC), no polar motion",
    "lvlh_origin": "Same actor factual trajectory at each timestamp",
    "lvlh_axes": "R=unit(r_F), N=unit(r_F cross v_F), T=N cross R; Q rows R,T,N",
    "erfa_version": erfa.__version__,
}
_TIME_START = datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
_TIME_END = datetime(2027, 1, 1, tzinfo=timezone.utc).timestamp()
_J2000_BIAS = erfa.pmat06(2451545.0, 0.0)


def _times(times_unix_s):
    times = np.asarray(times_unix_s, dtype=np.float64)
    if times.ndim != 1 or not times.size or not np.isfinite(times).all():
        raise ValueError("times_unix_s must be a nonempty finite (T,) array.")
    if np.any(times < _TIME_START) or np.any(times >= _TIME_END):
        raise ValueError("This validation helper uses the verified 2026 TT-UTC offset only.")
    return times


def _dates(times):
    days = np.floor(times / 86400.0)
    utc1 = 2440587.5 + days
    utc2 = (times - days * 86400.0) / 86400.0
    return utc1, utc2, utc2 + TT_MINUS_UTC_S / 86400.0


def _teme_matrix(times):
    utc1, utc2, tt2 = _dates(times)
    celestial = erfa.c2i06a(utc1, tt2)
    angle = erfa.gmst82(utc1, utc2) - erfa.era00(utc1, utc2) - erfa.sp00(utc1, tt2)
    z_rotation = erfa.rz(angle, np.broadcast_to(np.eye(3), (len(times), 3, 3)))
    return _J2000_BIAS @ celestial.swapaxes(-1, -2) @ z_rotation


def teme_to_j2000(times_unix_s, states):
    """Convert finite (..., T, 6) [position, velocity] states, preserving shape."""
    times = _times(times_unix_s)
    states = np.asarray(states, dtype=np.float64)
    if states.ndim < 2 or states.shape[-2:] != (len(times), 6) or not np.isfinite(states).all():
        raise ValueError("states must be finite (..., T, 6), in km and km/s.")
    rotation = _teme_matrix(times)
    derivative = _teme_matrix(times + 0.5) - _teme_matrix(times - 0.5)
    position = np.einsum("tij,...tj->...ti", rotation, states[..., :3])
    velocity = np.einsum("tij,...tj->...ti", rotation, states[..., 3:])
    velocity += np.einsum("tij,...tj->...ti", derivative, states[..., :3])
    return np.concatenate((position, velocity), axis=-1)


def actor_rtn_basis(actor_j2000):
    """Return (T, 3, 3) right-handed RTN bases of one factual actor (T, 6)."""
    actor = np.asarray(actor_j2000, dtype=np.float64)
    if actor.ndim != 2 or actor.shape[1] != 6 or not len(actor) or not np.isfinite(actor).all():
        raise ValueError("actor_j2000 must be a nonempty finite (T, 6) state array.")
    radial = actor[:, :3]
    normal = np.cross(radial, actor[:, 3:])
    radius = np.linalg.norm(radial, axis=1, keepdims=True)
    angular_momentum = np.linalg.norm(normal, axis=1, keepdims=True)
    if np.any(radius <= 0.0) or np.any(angular_momentum <= 1e-12):
        raise ValueError("RTN requires nonzero radius and angular momentum.")
    radial = radial / radius
    normal = normal / angular_momentum
    transverse = np.cross(normal, radial)
    return np.stack((radial, transverse, normal), axis=1)


def relative_rtn(states_xyz, actor_xyz, Q):
    """Rotate (..., T, 3) positions about the common factual actor (T, 3)."""
    positions = np.asarray(states_xyz, dtype=np.float64)
    origin = np.asarray(actor_xyz, dtype=np.float64)
    basis = np.asarray(Q, dtype=np.float64)
    if origin.ndim != 2 or origin.shape[1] != 3 or not len(origin):
        raise ValueError("actor_xyz must have shape (T, 3).")
    if positions.ndim < 2 or positions.shape[-2:] != origin.shape or basis.shape != (len(origin), 3, 3):
        raise ValueError("Positions (..., T, 3), actor (T, 3), and Q (T, 3, 3) must align.")
    if not all(np.isfinite(value).all() for value in (positions, origin, basis)):
        raise ValueError("RTN inputs must be finite.")
    return np.einsum("tij,...tj->...ti", basis, positions - origin)


def spherical_ground_track(times_unix_s, positions_j2000):
    """Return (..., T, 2) [latitude, longitude] degrees in the original Earth model."""
    times = _times(times_unix_s)
    position = np.asarray(positions_j2000, dtype=np.float64)
    if position.ndim < 2 or position.shape[-2:] != (len(times), 3) or not np.isfinite(position).all():
        raise ValueError("positions_j2000 must be finite (..., T, 3).")
    raw = np.einsum("tji,...tj->...ti", _teme_matrix(times), position)
    radius = np.linalg.norm(raw, axis=-1)
    if np.any(radius <= 0.0):
        raise ValueError("Geographic coordinates require nonzero position vectors.")
    utc1, utc2, _ = _dates(times)
    jd = utc1 + utc2
    centuries = (jd - 2451545.0) / 36525.0
    # Preserve orbit_core.gmst_angle exactly, including its original UTC convention.
    gmst = np.radians((280.46061837 + 360.98564736629 * (jd - 2451545.0)
                      + 0.000387933 * centuries ** 2 - centuries ** 3 / 38710000.0) % 360.0)
    latitude = np.degrees(np.arcsin(np.clip(raw[..., 2] / radius, -1.0, 1.0)))
    longitude = (np.degrees(np.arctan2(raw[..., 1], raw[..., 0]) - gmst) + 180.0) % 360.0 - 180.0
    return np.stack((latitude, longitude), axis=-1)


if __name__ == "__main__":
    start = datetime(2026, 9, 8, 10, 27, tzinfo=timezone.utc).timestamp()
    times = start + np.arange(181) * 60.0
    matrix = _teme_matrix(times)
    orthogonal_error = np.max(np.abs(matrix @ matrix.swapaxes(-1, -2) - np.eye(3)))
    assert orthogonal_error < 2e-15
    assert np.allclose(np.linalg.det(matrix), 1.0, atol=2e-15, rtol=0)

    utc1, utc2, tt2 = _dates(times)
    xp, yp = 0.5 / 206264.806247, 0.3 / 206264.806247
    teme_itrs = erfa.c2tcio(np.eye(3), erfa.gmst82(utc1, utc2), erfa.pom00(xp, yp, 0.0))
    cirs_itrs = erfa.c2tcio(np.eye(3), erfa.era00(utc1, utc2), erfa.pom00(xp, yp, erfa.sp00(utc1, tt2)))
    chain = _J2000_BIAS @ erfa.c2i06a(utc1, tt2).swapaxes(-1, -2) @ cirs_itrs.swapaxes(-1, -2) @ teme_itrs
    chain_error = np.max(np.abs(matrix - chain))
    assert chain_error < 2e-15

    raw = np.tile([7000.0, 200.0, -1500.0, -1.0, 7.0, 2.0], (len(times), 1))
    transformed = teme_to_j2000(times, raw)
    before, after = raw.copy(), raw.copy()
    before[:, :3] -= raw[:, 3:] * 0.5
    after[:, :3] += raw[:, 3:] * 0.5
    numeric_velocity = (teme_to_j2000(times + 0.5, after)[:, :3]
                        - teme_to_j2000(times - 0.5, before)[:, :3])
    velocity_error = np.max(np.abs(numeric_velocity - transformed[:, 3:]))
    assert velocity_error < 1e-9

    basis = actor_rtn_basis(transformed)
    assert np.allclose(basis @ basis.swapaxes(-1, -2), np.eye(3), atol=2e-15, rtol=0)
    assert np.allclose(np.linalg.det(basis), 1.0, atol=2e-15, rtol=0)
    candidates = np.stack((raw, raw + [30, -10, 80, 0, 0, 0]))
    converted_candidates = teme_to_j2000(times, candidates)
    rtn = relative_rtn(converted_candidates[..., :3], transformed[:, :3], basis)
    distance_error = np.max(np.abs(np.linalg.norm(rtn, axis=-1)
                                  - np.linalg.norm(converted_candidates[..., :3] - transformed[:, :3], axis=-1)))
    assert distance_error < 1e-10
    assert np.array_equal(rtn[0], np.zeros((len(times), 3)))
    raw_latitude = np.degrees(np.arcsin(raw[:, 2] / np.linalg.norm(raw[:, :3], axis=1)))
    assert np.allclose(spherical_ground_track(times, transformed[:, :3])[:, 0], raw_latitude, atol=1e-12, rtol=0)
    print({"orthogonal_error": float(orthogonal_error), "full_chain_error": float(chain_error),
           "velocity_finite_difference_error_km_s": float(velocity_error),
           "rtn_distance_error_km": float(distance_error)})
