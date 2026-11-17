from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from sgp4.api import Satrec, jday


MU_EARTH_KM3_S2 = 398600.4418
EARTH_RADIUS_KM = 6378.137
EARTH_ROTATION_RAD_S = 7.2921150e-5
SECONDS_PER_DAY = 86400.0
TWO_PI = 2.0 * math.pi


Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class TrajectoryPoint:
    """One ECI state sample."""

    t: datetime
    r_km: Vector3
    v_km_s: Vector3

    def to_json(self) -> dict[str, object]:
        return {
            "t": self.t.isoformat(),
            "r": list(self.r_km),
            "v": list(self.v_km_s),
        }


@dataclass(frozen=True)
class OrbitObject:
    """A satellite-like object initialized from one TLE/GP CSV row."""

    norad_cat_id: str
    object_name: str
    group: str
    epoch: datetime
    mean_motion_rev_day: float
    eccentricity: float
    inclination_deg: float
    raan_deg: float
    arg_pericenter_deg: float
    mean_anomaly_deg: float
    tle_line1: str
    tle_line2: str

    @classmethod
    def from_csv_row(cls, row: dict[str, str]) -> "OrbitObject":
        return cls(
            norad_cat_id=row["norad_cat_id"],
            object_name=row["object_name"],
            group=row.get("group", ""),
            epoch=parse_epoch(row["epoch"]),
            mean_motion_rev_day=float(row["mean_motion"]),
            eccentricity=float(row["eccentricity"]),
            inclination_deg=float(row["inclination"]),
            raan_deg=float(row["raan"]),
            arg_pericenter_deg=float(row["arg_of_pericenter"]),
            mean_anomaly_deg=float(row["mean_anomaly"]),
            tle_line1=row.get("tle_line1", ""),
            tle_line2=row.get("tle_line2", ""),
        )


@dataclass(frozen=True)
class GroundRegion:
    """A sensitive ground point, later converted into an ECI pseudo-trajectory."""

    name: str
    latitude_deg: float
    longitude_deg: float


class TleCatalogLoader:
    """Load the existing Space-Track CSV snapshot into OrbitObject instances."""

    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path

    def load(self, limit: int | None = None) -> list[OrbitObject]:
        objects: list[OrbitObject] = []
        with self.csv_path.open("r", encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                try:
                    objects.append(OrbitObject.from_csv_row(row))
                except (KeyError, TypeError, ValueError):
                    continue
                if limit is not None and len(objects) >= limit:
                    break
        return objects


class TimeGrid:
    """Build a common UTC time grid for actor and counterpart propagation."""

    def __init__(self, horizon_minutes: int = 180, step_seconds: int = 300) -> None:
        self.horizon_minutes = horizon_minutes
        self.step_seconds = step_seconds

    def build(self, start_time: datetime) -> list[datetime]:
        total_seconds = self.horizon_minutes * 60
        count = total_seconds // self.step_seconds + 1
        return [
            start_time + timedelta(seconds=i * self.step_seconds)
            for i in range(count)
        ]


class KeplerOrbitPropagator:
    """
    Lightweight two-body propagator from TLE mean elements.

    This is intentionally a first-pass replacement for SGP4 so the data pipeline
    can run without external packages. Later, this class can be replaced with an
    SGP4-backed implementation while keeping the same public methods.
    """

    def propagate(self, obj: OrbitObject, times: Iterable[datetime]) -> list[TrajectoryPoint]:
        n_rad_s = obj.mean_motion_rev_day * TWO_PI / SECONDS_PER_DAY
        semi_major_axis_km = (MU_EARTH_KM3_S2 / (n_rad_s * n_rad_s)) ** (1.0 / 3.0)

        inc = math.radians(obj.inclination_deg)
        raan = math.radians(obj.raan_deg)
        argp = math.radians(obj.arg_pericenter_deg)
        mean_anomaly_0 = math.radians(obj.mean_anomaly_deg)

        points: list[TrajectoryPoint] = []
        for t in times:
            dt_seconds = (t - obj.epoch).total_seconds()
            mean_anomaly = normalize_angle(mean_anomaly_0 + n_rad_s * dt_seconds)
            eccentric_anomaly = solve_kepler(mean_anomaly, obj.eccentricity)
            r_pqw, v_pqw = pqw_state(
                semi_major_axis_km,
                obj.eccentricity,
                eccentric_anomaly,
                n_rad_s,
            )
            r_eci = rotate_pqw_to_eci(r_pqw, raan, inc, argp)
            v_eci = rotate_pqw_to_eci(v_pqw, raan, inc, argp)
            points.append(TrajectoryPoint(t=t, r_km=r_eci, v_km_s=v_eci))
        return points


class Sgp4OrbitPropagator:
    """Propagate TLE states with SGP4."""

    def propagate(self, obj: OrbitObject, times: Iterable[datetime]) -> list[TrajectoryPoint]:
        if not obj.tle_line1 or not obj.tle_line2:
            raise ValueError(f"Missing TLE lines for {obj.object_name}")

        satellite = Satrec.twoline2rv(obj.tle_line1, obj.tle_line2)
        points: list[TrajectoryPoint] = []
        for t in times:
            utc = t.astimezone(timezone.utc)
            jd, fr = jday(
                utc.year,
                utc.month,
                utc.day,
                utc.hour,
                utc.minute,
                utc.second + utc.microsecond / 1e6,
            )
            error, r, v = satellite.sgp4(jd, fr)
            if error != 0:
                raise ValueError(
                    f"SGP4 failed for {obj.object_name} at {utc.isoformat()} "
                    f"with error code {error}"
                )
            r_vec: Vector3 = (float(r[0]), float(r[1]), float(r[2]))
            v_vec: Vector3 = (float(v[0]), float(v[1]), float(v[2]))
            points.append(TrajectoryPoint(t=t, r_km=r_vec, v_km_s=v_vec))
        return points


class GroundPointPropagator:
    """Convert fixed latitude/longitude points into ECI pseudo-trajectories."""

    def propagate(self, region: GroundRegion, times: Iterable[datetime]) -> list[TrajectoryPoint]:
        lat = math.radians(region.latitude_deg)
        lon = math.radians(region.longitude_deg)
        x_ecef = EARTH_RADIUS_KM * math.cos(lat) * math.cos(lon)
        y_ecef = EARTH_RADIUS_KM * math.cos(lat) * math.sin(lon)
        z_ecef = EARTH_RADIUS_KM * math.sin(lat)

        points: list[TrajectoryPoint] = []
        for t in times:
            theta = gmst_angle(t)
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            x_eci = cos_t * x_ecef - sin_t * y_ecef
            y_eci = sin_t * x_ecef + cos_t * y_ecef
            z_eci = z_ecef
            r_eci = (x_eci, y_eci, z_eci)
            v_eci = (
                -EARTH_ROTATION_RAD_S * y_eci,
                EARTH_ROTATION_RAD_S * x_eci,
                0.0,
            )
            points.append(TrajectoryPoint(t=t, r_km=r_eci, v_km_s=v_eci))
        return points


def parse_epoch(value: str) -> datetime:
    epoch = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if epoch.tzinfo is None:
        return epoch.replace(tzinfo=timezone.utc)
    return epoch.astimezone(timezone.utc)


def default_sensitive_regions() -> list[GroundRegion]:
    return [
        GroundRegion("Beijing", 39.9042, 116.4074),
        GroundRegion("Shanghai", 31.2304, 121.4737),
        GroundRegion("Xi'an", 34.3416, 108.9398),
        GroundRegion("Chengdu", 30.5728, 104.0668),
        GroundRegion("Zhengzhou", 34.7466, 113.6254),
        GroundRegion("Taiwan Strait", 24.0000, 119.5000),
        GroundRegion("Taipei", 25.0330, 121.5654),
        GroundRegion("Kaohsiung", 22.6273, 120.3014),
        GroundRegion("Kyiv", 50.4501, 30.5234),
        GroundRegion("Sevastopol", 44.6167, 33.5254),
        GroundRegion("Black Sea", 43.0000, 34.0000),
        GroundRegion("South China Sea", 12.0000, 114.0000),
        GroundRegion("Scarborough Shoal", 15.1533, 117.7539),
        GroundRegion("Spratly Islands", 10.0000, 114.0000),
        GroundRegion("Guam", 13.4443, 144.7937),
        GroundRegion("Okinawa", 26.3344, 127.8056),
        GroundRegion("Korean Peninsula", 38.0000, 127.0000),
        GroundRegion("Persian Gulf", 26.5000, 51.0000),
        GroundRegion("Red Sea", 20.0000, 38.0000),
        GroundRegion("Suez Canal", 30.5852, 32.2654),
        GroundRegion("Strait of Hormuz", 26.5667, 56.2500),
        GroundRegion("Diego Garcia", -7.3133, 72.4111),
    ]


def normalize_angle(angle_rad: float) -> float:
    return angle_rad % TWO_PI


def solve_kepler(mean_anomaly: float, eccentricity: float, max_iter: int = 20) -> float:
    eccentric_anomaly = mean_anomaly if eccentricity < 0.8 else math.pi
    for _ in range(max_iter):
        f = eccentric_anomaly - eccentricity * math.sin(eccentric_anomaly) - mean_anomaly
        fp = 1.0 - eccentricity * math.cos(eccentric_anomaly)
        step = f / fp
        eccentric_anomaly -= step
        if abs(step) < 1e-12:
            break
    return eccentric_anomaly


def pqw_state(
    semi_major_axis_km: float,
    eccentricity: float,
    eccentric_anomaly: float,
    mean_motion_rad_s: float,
) -> tuple[Vector3, Vector3]:
    cos_e = math.cos(eccentric_anomaly)
    sin_e = math.sin(eccentric_anomaly)
    one_minus_e_cos = 1.0 - eccentricity * cos_e
    sqrt_one_minus_e2 = math.sqrt(1.0 - eccentricity * eccentricity)

    r_pqw = (
        semi_major_axis_km * (cos_e - eccentricity),
        semi_major_axis_km * sqrt_one_minus_e2 * sin_e,
        0.0,
    )
    v_pqw = (
        -semi_major_axis_km * mean_motion_rad_s * sin_e / one_minus_e_cos,
        semi_major_axis_km * mean_motion_rad_s * sqrt_one_minus_e2 * cos_e / one_minus_e_cos,
        0.0,
    )
    return r_pqw, v_pqw


def rotate_pqw_to_eci(vector: Vector3, raan: float, inc: float, argp: float) -> Vector3:
    x, y, z = vector
    cos_o = math.cos(raan)
    sin_o = math.sin(raan)
    cos_i = math.cos(inc)
    sin_i = math.sin(inc)
    cos_w = math.cos(argp)
    sin_w = math.sin(argp)

    r11 = cos_o * cos_w - sin_o * sin_w * cos_i
    r12 = -cos_o * sin_w - sin_o * cos_w * cos_i
    r13 = sin_o * sin_i
    r21 = sin_o * cos_w + cos_o * sin_w * cos_i
    r22 = -sin_o * sin_w + cos_o * cos_w * cos_i
    r23 = -cos_o * sin_i
    r31 = sin_w * sin_i
    r32 = cos_w * sin_i
    r33 = cos_i
    return (
        r11 * x + r12 * y + r13 * z,
        r21 * x + r22 * y + r23 * z,
        r31 * x + r32 * y + r33 * z,
    )


def gmst_angle(t: datetime) -> float:
    jd = julian_date(t)
    centuries = (jd - 2451545.0) / 36525.0
    gmst_deg = (
        280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * centuries * centuries
        - centuries * centuries * centuries / 38710000.0
    )
    return math.radians(gmst_deg % 360.0)


def julian_date(t: datetime) -> float:
    utc = t.astimezone(timezone.utc)
    year = utc.year
    month = utc.month
    day = utc.day
    hour = utc.hour + utc.minute / 60.0 + (utc.second + utc.microsecond / 1e6) / 3600.0
    if month <= 2:
        year -= 1
        month += 12
    a = year // 100
    b = 2 - a + a // 4
    return (
        math.floor(365.25 * (year + 4716))
        + math.floor(30.6001 * (month + 1))
        + day
        + b
        - 1524.5
        + hour / 24.0
    )


def vector_sub(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vector_add(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vector_scale(v: Vector3, scale: float) -> Vector3:
    return (v[0] * scale, v[1] * scale, v[2] * scale)


def vector_norm(v: Vector3) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def vector_unit(v: Vector3) -> Vector3:
    norm = vector_norm(v)
    if norm == 0.0:
        return (0.0, 0.0, 0.0)
    return (v[0] / norm, v[1] / norm, v[2] / norm)


def cross(a: Vector3, b: Vector3) -> Vector3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def apply_impulse_maneuver(
    trajectory: list[TrajectoryPoint],
    maneuver_time: datetime,
    delta_v_km_s: Vector3,
) -> list[TrajectoryPoint]:
    """Apply one impulse, then continue with a two-body propagation."""
    if len(trajectory) < 2:
        return trajectory

    maneuver_index = min(
        range(len(trajectory)),
        key=lambda i: abs((trajectory[i].t - maneuver_time).total_seconds()),
    )
    maneuver_point = trajectory[maneuver_index]
    r = maneuver_point.r_km
    v = vector_add(maneuver_point.v_km_s, delta_v_km_s)

    points = list(trajectory[:maneuver_index])
    points.append(TrajectoryPoint(t=maneuver_point.t, r_km=r, v_km_s=v))
    previous_time = maneuver_point.t
    for base_point in trajectory[maneuver_index + 1:]:
        dt = (base_point.t - previous_time).total_seconds()
        r, v = rk4_two_body_step(r, v, dt)
        points.append(TrajectoryPoint(t=base_point.t, r_km=r, v_km_s=v))
        previous_time = base_point.t
    return points


def rk4_two_body_step(r: Vector3, v: Vector3, dt: float) -> tuple[Vector3, Vector3]:
    # ponytail: two-body post-impulse continuation; switch to high-fidelity OD when needed.
    def derivative(state_r: Vector3, state_v: Vector3) -> tuple[Vector3, Vector3]:
        radius = vector_norm(state_r)
        if radius == 0.0:
            acceleration = (0.0, 0.0, 0.0)
        else:
            factor = -MU_EARTH_KM3_S2 / (radius ** 3)
            acceleration = vector_scale(state_r, factor)
        return state_v, acceleration

    k1_r, k1_v = derivative(r, v)
    k2_r, k2_v = derivative(
        vector_add(r, vector_scale(k1_r, 0.5 * dt)),
        vector_add(v, vector_scale(k1_v, 0.5 * dt)),
    )
    k3_r, k3_v = derivative(
        vector_add(r, vector_scale(k2_r, 0.5 * dt)),
        vector_add(v, vector_scale(k2_v, 0.5 * dt)),
    )
    k4_r, k4_v = derivative(
        vector_add(r, vector_scale(k3_r, dt)),
        vector_add(v, vector_scale(k3_v, dt)),
    )

    r_next = vector_add(
        r,
        vector_scale(
            vector_add(
                vector_add(k1_r, vector_scale(k2_r, 2.0)),
                vector_add(vector_scale(k3_r, 2.0), k4_r),
            ),
            dt / 6.0,
        ),
    )
    v_next = vector_add(
        v,
        vector_scale(
            vector_add(
                vector_add(k1_v, vector_scale(k2_v, 2.0)),
                vector_add(vector_scale(k3_v, 2.0), k4_v),
            ),
            dt / 6.0,
        ),
    )
    return r_next, v_next
