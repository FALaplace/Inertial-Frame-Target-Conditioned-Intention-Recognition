"""Recovered historical CandidateMatcher scoring; no new calibration.

Exact method/helper bodies are extracted from the 2026-07-08 archive and its
successful patches. Its first 260 lines match the 2026-09-10 source snapshot;
2026-09-04 archived key lines also confirm the implemented gain/uncertainty.
Provenance: ../experiment_data/matching_score_legacy_20261019/history/.

Legacy semantics intentionally retained: pre/post factual maneuver gain,
neutral missing evidence, and dt per qualifying sample (including endpoints).
Main experiment: frozen 60-second trajectories; original 300-second histories
are a separate reproduction audit. This module does not select candidates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from orbit_core import EARTH_RADIUS_KM, TrajectoryPoint, cross, vector_norm, vector_sub

@dataclass(frozen=True)
class MatchScore:
    """Final score plus interpretable components for debugging."""

    total: float
    components: dict[str, float]
    metrics: dict[str, float]


class CandidateMatcher:
    """Score actor-counterpart relevance before intention classification."""

    def __init__(self) -> None:
        self.dca_scale_km = 1500.0
        self.tca_scale_s = 7200.0
        self.dwell_threshold_km = 1200.0
        self.dwell_scale_s = 1800.0

        self.altitude_scale_km = 500.0
        self.inclination_scale_deg = 12.0
        self.plane_angle_scale_deg = 20.0
        self.phase_angle_scale_deg = 45.0

        self.delta_v_scale_km_s = 1.0
        self.neutral_evidence_score = 0.5

        self.ground_distance_scale_km = 1200.0
        self.access_elevation_threshold_deg = 10.0
        self.access_duration_scale_s = 1200.0
        self.revisit_count_scale = 2.0

    def score_satellite_pair(
        self,
        actor_trajectory: list[TrajectoryPoint],
        target_trajectory: list[TrajectoryPoint],
        maneuver_events: list[dict[str, object]] | None = None,
        error_summary: dict[str, float | None] | None = None,
    ) -> float:
        return self.score_satellite_pair_details(
            actor_trajectory,
            target_trajectory,
            maneuver_events=maneuver_events,
            error_summary=error_summary,
        ).total

    def score_satellite_pair_details(
        self,
        actor_trajectory: list[TrajectoryPoint],
        target_trajectory: list[TrajectoryPoint],
        maneuver_events: list[dict[str, object]] | None = None,
        error_summary: dict[str, float | None] | None = None,
    ) -> MatchScore:
        proximity = self._satellite_proximity_score(actor_trajectory, target_trajectory)
        geometry = self._satellite_geometry_score(actor_trajectory, target_trajectory)
        reachability = self._satellite_reachability_score(actor_trajectory, target_trajectory)
        maneuver_gain = self._satellite_maneuver_gain_score(
            actor_trajectory,
            target_trajectory,
            maneuver_events,
        )
        uncertainty = self._uncertainty_score(error_summary)

        components = {
            "proximity": proximity.total,
            "geometry": geometry.total,
            "reachability": reachability.total,
            "maneuver_gain": maneuver_gain.total,
            "uncertainty": uncertainty.total,
        }
        total = (
            0.30 * components["proximity"]
            + 0.20 * components["geometry"]
            + 0.20 * components["reachability"]
            + 0.20 * components["maneuver_gain"]
            + 0.10 * components["uncertainty"]
        )
        metrics = {
            **prefix_keys("proximity", proximity.metrics),
            **prefix_keys("geometry", geometry.metrics),
            **prefix_keys("reachability", reachability.metrics),
            **prefix_keys("maneuver_gain", maneuver_gain.metrics),
            **prefix_keys("uncertainty", uncertainty.metrics),
        }
        return MatchScore(total=clamp01(total), components=components, metrics=metrics)

    def score_region_pair(
        self,
        actor_trajectory: list[TrajectoryPoint],
        region_trajectory: list[TrajectoryPoint],
        maneuver_events: list[dict[str, object]] | None = None,
        error_summary: dict[str, float | None] | None = None,
    ) -> float:
        return self.score_region_pair_details(
            actor_trajectory,
            region_trajectory,
            maneuver_events=maneuver_events,
            error_summary=error_summary,
        ).total

    def score_region_pair_details(
        self,
        actor_trajectory: list[TrajectoryPoint],
        region_trajectory: list[TrajectoryPoint],
        maneuver_events: list[dict[str, object]] | None = None,
        error_summary: dict[str, float | None] | None = None,
    ) -> MatchScore:
        ground_track = self._region_ground_track_score(actor_trajectory, region_trajectory)
        access = self._region_access_score(actor_trajectory, region_trajectory)
        revisit = self._region_revisit_score(actor_trajectory, region_trajectory)
        coverage_gain = self._region_coverage_gain_score(
            actor_trajectory,
            region_trajectory,
            maneuver_events,
        )
        uncertainty = self._uncertainty_score(error_summary)

        components = {
            "ground_track": ground_track.total,
            "access": access.total,
            "revisit": revisit.total,
            "coverage_gain": coverage_gain.total,
            "uncertainty": uncertainty.total,
        }
        total = (
            0.30 * components["ground_track"]
            + 0.25 * components["access"]
            + 0.20 * components["revisit"]
            + 0.15 * components["coverage_gain"]
            + 0.10 * components["uncertainty"]
        )
        metrics = {
            **prefix_keys("ground_track", ground_track.metrics),
            **prefix_keys("access", access.metrics),
            **prefix_keys("revisit", revisit.metrics),
            **prefix_keys("coverage_gain", coverage_gain.metrics),
            **prefix_keys("uncertainty", uncertainty.metrics),
        }
        return MatchScore(total=clamp01(total), components=components, metrics=metrics)

    def _satellite_proximity_score(
        self,
        actor_trajectory: list[TrajectoryPoint],
        target_trajectory: list[TrajectoryPoint],
    ) -> MatchScore:
        distances = relative_distances(actor_trajectory, target_trajectory)
        min_index = min(range(len(distances)), key=distances.__getitem__)
        min_distance = distances[min_index]
        time_to_ca = abs(
            (actor_trajectory[min_index].t - actor_trajectory[0].t).total_seconds()
        )
        dwell_seconds = dwell_time_seconds(
            actor_trajectory,
            distances,
            threshold_km=self.dwell_threshold_km,
        )

        dca_score = exp_score(min_distance, self.dca_scale_km)
        tca_score = exp_score(time_to_ca, self.tca_scale_s)
        dwell_score = one_minus_exp_score(dwell_seconds, self.dwell_scale_s)
        total = 0.50 * dca_score + 0.20 * tca_score + 0.30 * dwell_score
        return MatchScore(
            total=clamp01(total),
            components={},
            metrics={
                "d_min_km": min_distance,
                "time_to_ca_s": time_to_ca,
                "dwell_s": dwell_seconds,
            },
        )

    def _satellite_geometry_score(
        self,
        actor_trajectory: list[TrajectoryPoint],
        target_trajectory: list[TrajectoryPoint],
    ) -> MatchScore:
        altitude_gap = abs(mean_radius(actor_trajectory) - mean_radius(target_trajectory))
        inc_gap = abs(mean_inclination_deg(actor_trajectory) - mean_inclination_deg(target_trajectory))
        plane_angle = mean_plane_angle_deg(actor_trajectory, target_trajectory)
        phase_gap = mean_position_angle_deg(actor_trajectory, target_trajectory)

        altitude_score = exp_score(altitude_gap, self.altitude_scale_km)
        inclination_score = exp_score(inc_gap, self.inclination_scale_deg)
        plane_score = exp_score(plane_angle, self.plane_angle_scale_deg)
        phase_score = exp_score(phase_gap, self.phase_angle_scale_deg)
        total = (
            0.35 * altitude_score
            + 0.25 * inclination_score
            + 0.25 * plane_score
            + 0.15 * phase_score
        )
        return MatchScore(
            total=clamp01(total),
            components={},
            metrics={
                "altitude_gap_km": altitude_gap,
                "inclination_gap_deg": inc_gap,
                "plane_angle_deg": plane_angle,
                "phase_gap_deg": phase_gap,
            },
        )

    def _satellite_reachability_score(
        self,
        actor_trajectory: list[TrajectoryPoint],
        target_trajectory: list[TrajectoryPoint],
    ) -> MatchScore:
        actor_radius = mean_radius(actor_trajectory)
        target_radius = mean_radius(target_trajectory)
        plane_angle = mean_plane_angle_deg(actor_trajectory, target_trajectory)

        altitude_delta_v = hohmann_delta_v_km_s(actor_radius, target_radius)
        plane_delta_v = plane_change_delta_v_km_s(actor_trajectory, plane_angle)
        required_delta_v = altitude_delta_v + plane_delta_v
        total = exp_score(required_delta_v, self.delta_v_scale_km_s)
        return MatchScore(
            total=clamp01(total),
            components={},
            metrics={
                "altitude_delta_v_km_s": altitude_delta_v,
                "plane_delta_v_km_s": plane_delta_v,
                "required_delta_v_km_s": required_delta_v,
            },
        )

    def _satellite_maneuver_gain_score(
        self,
        actor_trajectory: list[TrajectoryPoint],
        target_trajectory: list[TrajectoryPoint],
        maneuver_events: list[dict[str, object]] | None,
    ) -> MatchScore:
        event_time = first_maneuver_time(maneuver_events)
        if event_time is None:
            return self._neutral_score("no_maneuver_event_input")

        distances = relative_distances(actor_trajectory, target_trajectory)
        before = [
            distance
            for point, distance in zip(actor_trajectory, distances)
            if point.t < event_time
        ]
        after = [
            distance
            for point, distance in zip(actor_trajectory, distances)
            if point.t >= event_time
        ]
        if not before or not after:
            return self._neutral_score("maneuver_outside_time_window")

        d_pre = min(before)
        d_post = min(after)
        approach_gain = max(0.0, d_pre - d_post)
        time_sigma = first_maneuver_time_sigma(maneuver_events)
        total = one_minus_exp_score(approach_gain, 1000.0) * exp_score(time_sigma, 1800.0)
        return MatchScore(
            total=clamp01(total),
            components={},
            metrics={
                "pre_min_distance_km": d_pre,
                "post_min_distance_km": d_post,
                "approach_gain_km": approach_gain,
                "time_sigma_s": time_sigma,
            },
        )

    def _neutral_score(self, reason: str) -> MatchScore:
        return MatchScore(
            total=self.neutral_evidence_score,
            components={},
            metrics={"placeholder": 1.0, reason: 1.0},
        )

    def _region_ground_track_score(
        self,
        actor_trajectory: list[TrajectoryPoint],
        region_trajectory: list[TrajectoryPoint],
    ) -> MatchScore:
        distances = surface_distances(actor_trajectory, region_trajectory)
        min_distance = min(distances)
        total = exp_score(min_distance, self.ground_distance_scale_km)
        return MatchScore(
            total=clamp01(total),
            components={},
            metrics={"min_surface_distance_km": min_distance},
        )

    def _region_access_score(
        self,
        actor_trajectory: list[TrajectoryPoint],
        region_trajectory: list[TrajectoryPoint],
    ) -> MatchScore:
        access_seconds, pass_count = access_statistics(
            actor_trajectory,
            region_trajectory,
            elevation_threshold_deg=self.access_elevation_threshold_deg,
        )
        total = one_minus_exp_score(access_seconds, self.access_duration_scale_s)
        return MatchScore(
            total=clamp01(total),
            components={},
            metrics={
                "access_duration_s": access_seconds,
                "pass_count": float(pass_count),
            },
        )

    def _region_revisit_score(
        self,
        actor_trajectory: list[TrajectoryPoint],
        region_trajectory: list[TrajectoryPoint],
    ) -> MatchScore:
        _, pass_count = access_statistics(
            actor_trajectory,
            region_trajectory,
            elevation_threshold_deg=self.access_elevation_threshold_deg,
        )
        total = one_minus_exp_score(float(pass_count), self.revisit_count_scale)
        return MatchScore(
            total=clamp01(total),
            components={},
            metrics={"pass_count": float(pass_count)},
        )

    def _region_coverage_gain_score(
        self,
        actor_trajectory: list[TrajectoryPoint],
        region_trajectory: list[TrajectoryPoint],
        maneuver_events: list[dict[str, object]] | None,
    ) -> MatchScore:
        event_time = first_maneuver_time(maneuver_events)
        if event_time is None:
            return self._neutral_score("no_maneuver_event_input")

        before_s, after_s = access_seconds_before_after(
            actor_trajectory,
            region_trajectory,
            event_time,
            elevation_threshold_deg=self.access_elevation_threshold_deg,
        )
        coverage_gain = max(0.0, after_s - before_s)
        time_sigma = first_maneuver_time_sigma(maneuver_events)
        total = one_minus_exp_score(coverage_gain, 1200.0) * exp_score(time_sigma, 1800.0)
        return MatchScore(
            total=clamp01(total),
            components={},
            metrics={
                "access_before_s": before_s,
                "access_after_s": after_s,
                "coverage_gain_s": coverage_gain,
                "time_sigma_s": time_sigma,
            },
        )

    def _uncertainty_score(
        self,
        error_summary: dict[str, float | None] | None,
    ) -> MatchScore:
        if not error_summary:
            return self._neutral_score("no_error_summary_input")

        position_sigma = float(error_summary.get("orbit_position_sigma") or 0.0)
        velocity_sigma = float(error_summary.get("orbit_velocity_sigma") or 0.0)
        maneuver_time_sigma = float(error_summary.get("maneuver_time_sigma") or 0.0)
        total = (
            0.55 * exp_score(position_sigma, 20.0)
            + 0.25 * exp_score(velocity_sigma, 0.02)
            + 0.20 * exp_score(maneuver_time_sigma, 1800.0)
        )
        return MatchScore(
            total=clamp01(total),
            components={},
            metrics={
                "orbit_position_sigma_km": position_sigma,
                "orbit_velocity_sigma_km_s": velocity_sigma,
                "maneuver_time_sigma_s": maneuver_time_sigma,
            },
        )


def first_maneuver_time(maneuver_events: list[dict[str, object]] | None) -> datetime | None:
    if not maneuver_events:
        return None
    raw = maneuver_events[0].get("t_maneuver")
    if not isinstance(raw, str):
        return None
    return datetime.fromisoformat(raw)


def first_maneuver_time_sigma(maneuver_events: list[dict[str, object]] | None) -> float:
    if not maneuver_events:
        return 0.0
    raw = maneuver_events[0].get("time_sigma")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def relative_distances(
    traj_a: list[TrajectoryPoint],
    traj_b: list[TrajectoryPoint],
) -> list[float]:
    return [
        vector_norm(vector_sub(a.r_km, b.r_km))
        for a, b in zip(traj_a, traj_b)
    ]


def dwell_time_seconds(
    trajectory: list[TrajectoryPoint],
    distances: list[float],
    threshold_km: float,
) -> float:
    if len(trajectory) < 2:
        return 0.0
    dt = abs((trajectory[1].t - trajectory[0].t).total_seconds())
    return sum(1 for distance in distances if distance <= threshold_km) * dt


def surface_distances(
    actor_trajectory: list[TrajectoryPoint],
    region_trajectory: list[TrajectoryPoint],
) -> list[float]:
    distances: list[float] = []
    for actor, region in zip(actor_trajectory, region_trajectory):
        angle = angle_between(actor.r_km, region.r_km)
        distances.append(EARTH_RADIUS_KM * angle)
    return distances


def access_statistics(
    actor_trajectory: list[TrajectoryPoint],
    region_trajectory: list[TrajectoryPoint],
    elevation_threshold_deg: float,
) -> tuple[float, int]:
    if len(actor_trajectory) < 2:
        return 0.0, 0
    dt = abs((actor_trajectory[1].t - actor_trajectory[0].t).total_seconds())
    access_flags = []
    for actor, region in zip(actor_trajectory, region_trajectory):
        elevation = elevation_angle_deg(actor.r_km, region.r_km)
        access_flags.append(elevation >= elevation_threshold_deg)

    access_seconds = sum(1 for flag in access_flags if flag) * dt
    pass_count = count_true_segments(access_flags)
    return access_seconds, pass_count


def access_seconds_before_after(
    actor_trajectory: list[TrajectoryPoint],
    region_trajectory: list[TrajectoryPoint],
    event_time: datetime,
    elevation_threshold_deg: float,
) -> tuple[float, float]:
    if len(actor_trajectory) < 2:
        return 0.0, 0.0
    dt = abs((actor_trajectory[1].t - actor_trajectory[0].t).total_seconds())
    before_s = 0.0
    after_s = 0.0
    for actor, region in zip(actor_trajectory, region_trajectory):
        if elevation_angle_deg(actor.r_km, region.r_km) < elevation_threshold_deg:
            continue
        if actor.t < event_time:
            before_s += dt
        else:
            after_s += dt
    return before_s, after_s


def elevation_angle_deg(actor_r: tuple[float, float, float], region_r: tuple[float, float, float]) -> float:
    line_of_sight = vector_sub(actor_r, region_r)
    los_norm = vector_norm(line_of_sight)
    region_norm = vector_norm(region_r)
    if los_norm == 0.0 or region_norm == 0.0:
        return -90.0
    up = (region_r[0] / region_norm, region_r[1] / region_norm, region_r[2] / region_norm)
    sin_elev = max(-1.0, min(1.0, dot(line_of_sight, up) / los_norm))
    return math.degrees(math.asin(sin_elev))


def count_true_segments(flags: list[bool]) -> int:
    count = 0
    previous = False
    for flag in flags:
        if flag and not previous:
            count += 1
        previous = flag
    return count


def mean_radius(trajectory: list[TrajectoryPoint]) -> float:
    return sum(vector_norm(point.r_km) for point in trajectory) / len(trajectory)


def mean_speed(trajectory: list[TrajectoryPoint]) -> float:
    return sum(vector_norm(point.v_km_s) for point in trajectory) / len(trajectory)


def mean_inclination_deg(trajectory: list[TrajectoryPoint]) -> float:
    inclinations: list[float] = []
    for point in trajectory:
        h = cross(point.r_km, point.v_km_s)
        h_norm = vector_norm(h)
        if h_norm == 0.0:
            continue
        inclinations.append(math.degrees(math.acos(max(-1.0, min(1.0, h[2] / h_norm)))))
    if not inclinations:
        return 180.0
    return sum(inclinations) / len(inclinations)


def mean_plane_angle_deg(
    traj_a: list[TrajectoryPoint],
    traj_b: list[TrajectoryPoint],
) -> float:
    angles: list[float] = []
    for a, b in zip(traj_a, traj_b):
        h_a = cross(a.r_km, a.v_km_s)
        h_b = cross(b.r_km, b.v_km_s)
        if vector_norm(h_a) == 0.0 or vector_norm(h_b) == 0.0:
            continue
        angles.append(math.degrees(angle_between(h_a, h_b)))
    if not angles:
        return 180.0
    return sum(angles) / len(angles)


def mean_position_angle_deg(
    traj_a: list[TrajectoryPoint],
    traj_b: list[TrajectoryPoint],
) -> float:
    angles = [
        math.degrees(angle_between(a.r_km, b.r_km))
        for a, b in zip(traj_a, traj_b)
    ]
    return sum(angles) / len(angles)


def hohmann_delta_v_km_s(radius_a_km: float, radius_b_km: float) -> float:
    if radius_a_km <= 0.0 or radius_b_km <= 0.0:
        return 99.0
    if abs(radius_a_km - radius_b_km) < 1e-9:
        return 0.0
    transfer_a = 0.5 * (radius_a_km + radius_b_km)
    v1 = math.sqrt(398600.4418 / radius_a_km)
    v2 = math.sqrt(398600.4418 / radius_b_km)
    v_transfer_1 = math.sqrt(398600.4418 * (2.0 / radius_a_km - 1.0 / transfer_a))
    v_transfer_2 = math.sqrt(398600.4418 * (2.0 / radius_b_km - 1.0 / transfer_a))
    return abs(v_transfer_1 - v1) + abs(v2 - v_transfer_2)


def plane_change_delta_v_km_s(
    actor_trajectory: list[TrajectoryPoint],
    plane_angle_deg: float,
) -> float:
    speed = mean_speed(actor_trajectory)
    angle_rad = math.radians(plane_angle_deg)
    return 2.0 * speed * math.sin(0.5 * angle_rad)


def angle_between(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    norm_a = vector_norm(a)
    norm_b = vector_norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return math.pi
    cos_angle = max(-1.0, min(1.0, dot(a, b) / (norm_a * norm_b)))
    return math.acos(cos_angle)


def dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def exp_score(value: float, scale: float) -> float:
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    return math.exp(-max(0.0, value) / scale)


def one_minus_exp_score(value: float, scale: float) -> float:
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    return 1.0 - math.exp(-max(0.0, value) / scale)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def prefix_keys(prefix: str, payload: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in payload.items()}

if __name__ == "__main__":
    matcher = CandidateMatcher()
    assert matcher._uncertainty_score(None).total == 0.5
    assert matcher._uncertainty_score({"orbit_position_sigma": 0.0}).total == 1.0
    uncertainty = matcher._uncertainty_score({
        "orbit_position_sigma": 20.0,
        "orbit_velocity_sigma": 0.02,
        "maneuver_time_sigma": 1800.0,
    }).total
    assert abs(uncertainty - math.exp(-1.0)) < 1e-15
    assert abs(math.degrees(angle_between((0.0, 0.0, 1.0), (0.0, 0.0, -1.0))) - 180.0) < 1e-12
    assert count_true_segments([False, True, True, False, True]) == 2
    print("Historical scoring self-check passed; stored-pair regression is reported separately.")
