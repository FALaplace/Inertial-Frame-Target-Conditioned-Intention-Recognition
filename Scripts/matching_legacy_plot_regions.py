"""地面区域两张图：星下点与仰角上下排列，评分单独绘制，仅使用 factual 轨迹。"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

OUTPUT_MODE = "show"  # "show": 只显示；"savefig": 保存 PDF/PNG，不弹窗。
OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'experiment_data/matching_score_legacy_20261019'
TRACK_FIGSIZE = (7.2, 7.2)
SCORE_FIGSIZE = (6.8, 4.2)

COLORS = ('#1B4F72', '#9C3B10', '#145A32', '#6C3483', '#7D6608')
MARKERS = ('^', 'v', 'D', 'P', 'X')
COMPONENT_COLORS = ('#332288', '#44AA99', '#DDCC77', '#CC6677', '#882255')
COMPONENT_HATCHES = ('', '///', '\\\\', 'xx', '..')
COMPONENTS = {
    'spacecraft': ('prox', 'geom', 'reach', 'man', 'unc'),
    'regions': ('ground', 'access', 'revisit', 'cov', 'unc'),
}
COMPONENT_LABELS = {
    'prox': 'Proximity', 'geom': 'Geometry', 'reach': 'Maneuver-cost proxy',
    'man': 'Pre/post approach', 'unc': 'Uncertainty', 'ground': 'Ground proximity',
    'access': 'Access', 'revisit': 'Revisit', 'cov': 'Pre/post access',
}


class MatchingRegionsPlot:
    def __init__(self, output_dir):
        self.scene = 'regions'
        self.output_dir = Path(output_dir)
        self.analysis = None
        self.arrays = {}
        self.summary = {}

    def read_data(self):
        self.analysis = json.loads((self.output_dir / 'analysis.json').read_text(encoding='utf-8'))
        with (self.output_dir / 'matching_screening_summary.csv').open(encoding='utf-8-sig', newline='') as stream:
            self.summary = {row['scene']: row for row in csv.DictReader(stream)}
        with np.load(self.output_dir / 'analysis_states.npz', allow_pickle=False) as data:
            self.arrays = {key: data[key] for key in data.files if key.startswith(self.scene + '_')}
        for scene, meta in [(self.scene, self.analysis['scenes'][self.scene])]:
            times = self.array(scene, 'times_s')
            actor = self.array(scene, 'actor_f')
            candidates = self.array(scene, 'candidates')
            ids = self.array(scene, 'ids').astype(str)
            if times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0):
                raise ValueError('Scene times must be strictly increasing: ' + scene)
            if actor.shape != (len(times), 6) or candidates.shape != (len(ids), len(times), 6):
                raise ValueError('Scene state dimensions do not align: ' + scene)
            if len(ids) != len(set(ids)) or len(ids) != meta['valid_candidate_count']:
                raise ValueError('Candidate identities/count do not align: ' + scene)
            if int(self.summary[scene]['candidates']) != len(ids):
                raise ValueError('Screening summary and trajectory pool do not align: ' + scene)
            if len(meta['display']) != len(COLORS):
                raise ValueError('Each scene needs five display candidates.')
            mixed_count, mixed_kept = int(meta['mixed_pool_count']), int(meta['mixed_retained'])
            cutoff = float(meta['mixed_score_cutoff'])
            if not len(ids) <= mixed_count or not 0 < mixed_kept <= mixed_count or not 0 <= cutoff <= 1:
                raise ValueError('Invalid mixed-pool count, retention or cutoff: ' + scene)
            if not 0 <= int(self.summary[scene]['retained']) <= min(len(ids), mixed_kept):
                raise ValueError('Type retention count is inconsistent with the mixed pool: ' + scene)
            for item in meta['display']:
                if ids[int(item['array_index'])] != str(item['candidate_id']):
                    raise ValueError('Display identity does not match its state array.')
                component = item['component']
                if not 1 <= int(component['score_rank']) <= len(ids):
                    raise ValueError('Type-specific score rank is outside the valid pool.')
                if not 1 <= int(component['mixed_rank']) <= mixed_count:
                    raise ValueError('Mixed score rank is outside the mixed pool.')
                if not isinstance(component['retained'], bool) or component['retained'] != (component['mixed_rank'] <= mixed_kept):
                    raise ValueError('Retention must follow the stored mixed TopK rank, including ties.')
                if scene == 'regions':
                    latitude, longitude = float(item['latitude_deg']), float(item['longitude_deg'])
                    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
                        raise ValueError('Region-center coordinates are invalid.')
            if not all(np.isfinite(self.array(scene, key)).all()
                       for key in ('times_s', 'actor_f', 'candidates', 'candidate_rtn', 'actor_f_ground')):
                raise ValueError('Plotting states must be finite: ' + scene)

    def array(self, scene, name):
        return self.arrays[scene + '_' + name]

    def items(self, scene):
        return self.analysis['scenes'][scene]['display']

    def minutes(self, scene):
        time = self.array(scene, 'times_s')
        return (time - time[0]) / 60.0

    @staticmethod
    def legend(ax, handles=None, ncol=2, loc='upper right'):
        ax.legend(handles=handles, loc=loc, ncol=ncol, fontsize=12,
                  frameon=True, facecolor='white', edgecolor='0.85', framealpha=0.92,
                  handlelength=1.5, handletextpad=0.4, columnspacing=0.7, borderpad=0.35)

    @staticmethod
    def equal_space(ax, points):
        xyz = np.concatenate(points, axis=0)
        lower, upper = xyz.min(axis=0), xyz.max(axis=0)
        center = (lower + upper) / 2.0
        half = max(float(np.max(upper - lower)) * 0.54, 1.0)
        for setter, value in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
            setter(value - half, value + half)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=23, azim=-57)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_major_locator(MaxNLocator(3))
            axis.pane.fill = False
        ax.tick_params(labelsize=12, pad=0)
        ax.grid(True, alpha=0.25)

    @staticmethod
    def orbit(ax, xyz, color, label, linestyle='-', marker=None, event_index=None):
        line, = ax.plot(*xyz.T, color=color, lw=1.05, ls=linestyle, label=label,
                        marker=marker, markevery=(25, 45), markersize=3)
        ax.scatter(*xyz[0], marker='o', s=15, facecolor='white', edgecolor=color, depthshade=False, zorder=5)
        ax.scatter(*xyz[-1], marker='s', s=13, color=color, depthshade=False, zorder=5)
        arrow_index = min(int(len(xyz) * 0.60), len(xyz) - 2)
        delta = xyz[min(arrow_index + 4, len(xyz) - 1)] - xyz[arrow_index]
        if np.linalg.norm(delta) > 1e-10:
            ax.quiver(*xyz[arrow_index], *delta, color=color, linewidth=0.9,
                      arrow_length_ratio=0.45, normalize=False)
        if event_index is not None:
            ax.scatter(*xyz[event_index], marker='*', s=42, color=color, depthshade=False, zorder=6)
        return line

    def spatial_panel(self, ax, scene, frame):
        meta = self.analysis['scenes'][scene]
        event = meta.get('event_offset_min')
        event_index = None if event is None else int(np.argmin(np.abs(self.minutes(scene) - event)))
        points, handles = [], []
        if frame == 'eci':
            radius = self.analysis['parameters']['shared']['earth_radius_km']
            longitude, latitude = np.meshgrid(np.linspace(0, 2 * np.pi, 32), np.linspace(-np.pi / 2, np.pi / 2, 17))
            ax.plot_surface(radius * np.cos(latitude) * np.cos(longitude),
                            radius * np.cos(latitude) * np.sin(longitude), radius * np.sin(latitude),
                            color='#b9c9d3', alpha=0.11, linewidth=0, shade=False)
            points.append(np.array([[-radius] * 3, [radius] * 3]))
            factual = self.array(scene, 'actor_f')[:, :3]
            handles.append(self.orbit(ax, factual, 'black', 'Actor', event_index=event_index))
            points.append(factual)
            candidates = self.array(scene, 'candidates')[..., :3]
            labels = ('X (km)', 'Y (km)', 'Z (km)')
        else:
            ax.scatter(0, 0, 0, color='black', s=25, depthshade=False)
            handles.append(Line2D([], [], marker='o', color='black', ls='none', label='Actor'))
            points.append(np.zeros((1, 3)))
            candidates = self.array(scene, 'candidate_rtn')
            labels = ('R (km)', 'T (km)', 'N (km)')
        for i, item in enumerate(self.items(scene)):
            xyz = candidates[int(item['array_index'])]
            handles.append(self.orbit(ax, xyz, COLORS[i], item['label'], marker=MARKERS[i]))
            points.append(xyz)
        self.equal_space(ax, points)
        ax.set(xlabel=labels[0], ylabel=labels[1], zlabel=labels[2])
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.labelpad = 1.5
        self.legend(ax, handles, ncol=3, loc='upper left')

    def time_panel(self, ax, scene, elevation=False):
        minutes = self.minutes(scene)
        candidates = self.array(scene, 'candidates')[..., :3]
        actor = self.array(scene, 'actor_f')[:, :3]
        marker_indices = np.linspace(0, len(minutes) - 1, min(10, len(minutes)), dtype=int)
        handles = []
        for i, item in enumerate(self.items(scene)):
            counterpart = candidates[int(item['array_index'])]
            relative = actor - counterpart
            distance = np.linalg.norm(relative, axis=1)
            if elevation:
                values = np.degrees(np.arcsin(np.clip(np.sum(relative * counterpart, axis=1)
                                     / (distance * np.linalg.norm(counterpart, axis=1)), -1, 1)))
            else:
                values = distance
            line, = ax.plot(minutes, values, color=COLORS[i], lw=1.0,
                            marker=MARKERS[i], markevery=marker_indices, ms=5, label=item['label'])
            handles.append(line)
        branch = self.analysis['parameters'][scene]
        threshold = branch.get('elevation_threshold_deg' if elevation else 'dwell_distance_threshold_km')
        if threshold is not None:
            threshold_label = '{:g} deg access'.format(threshold) if elevation else '{:g} km dwell'.format(threshold)
            handles.append(ax.axhline(threshold, color='0.25', ls='-.', lw=0.8, label=threshold_label))
        ax.set(xlim=(minutes[0], minutes[-1]), xlabel='Time (min)',
               ylabel='Elevation (deg)' if elevation else 'Distance (km)')
        if elevation:
            ax.set_ylim(-90, 150)  # Leave room above physical elevations for the internal legend.
            ax.set_yticks([-90, -45, 0, 45, 90])
        else:
            ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
        ax.set_xticks([0, 30, 60, 90, 120, 150, 180])
        ax.grid(color='0.9', lw=0.5)
        self.legend(ax, handles, ncol=3)

    def ground_panel(self, ax, scene):
        latlon = self.array(scene, 'actor_f_ground')
        longitude, latitude = latlon[:, 1].copy(), latlon[:, 0]
        crossing = np.r_[False, np.abs(np.diff(longitude)) > 180]
        longitude[crossing] = np.nan
        line, = ax.plot(longitude, latitude, color='black', lw=1.0, label='Actor')
        ax.plot(latlon[0, 1], latlon[0, 0], 'o', ms=3, mfc='white', mec='black')
        ax.plot(latlon[-1, 1], latlon[-1, 0], 's', ms=3, color='black')
        index = int(len(latlon) * 0.6)
        if abs(latlon[index + 1, 1] - latlon[index, 1]) < 180:
            ax.annotate('', xy=latlon[index + 1, ::-1], xytext=latlon[index, ::-1],
                        arrowprops={'arrowstyle': '->', 'color': 'black', 'lw': 0.8})
        event = self.analysis['scenes'][scene].get('event_offset_min')
        if event is not None:
            event_index = int(np.argmin(np.abs(self.minutes(scene) - event)))
            ax.plot(latlon[event_index, 1], latlon[event_index, 0], '*', color='black', ms=6)
        for i, item in enumerate(self.items(scene)):
            latitude, longitude = float(item['latitude_deg']), float(item['longitude_deg'])
            ax.plot(longitude, latitude, marker=MARKERS[i], color=COLORS[i], ls='none',
                    ms=5, label=item['label'])
            ax.annotate(item['label'], (longitude, latitude), xytext=(4, -10 if i == 3 else 4), textcoords='offset points',
                        fontsize=12, color=COLORS[i])
        ax.set(xlim=(-180, 180), ylim=(-90, 90), xlabel='Longitude (deg)', ylabel='Latitude (deg)')
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_yticks([-90, -45, 0, 45, 90])
        ax.grid(color='0.9', lw=0.5)
        self.legend(ax, [line], ncol=1)

    def score_panel(self, ax, scene):
        items, names = self.items(scene), COMPONENTS[scene]
        scores = np.array([item['component']['total_score'] for item in items], dtype=float)
        meta = self.analysis['scenes'][scene]
        cutoff, mixed_kept = float(meta['mixed_score_cutoff']), int(meta['mixed_retained'])
        if not np.all(np.isfinite(scores)) or not np.all((scores >= 0) & (scores <= 1)):
            raise ValueError('Matching scores must be finite and within [0, 1].')
        # A shared, zero-based 0--0.75 display range makes both scenes comparable.
        ax.set(xlim=(0, 0.75), ylim=(len(items) - 0.5, -3.3), xlabel='Matching Score')
        ax.set_xticks(np.arange(0, 0.71, 0.1))
        ax.set_yticks(range(len(items)))
        ax.set_yticklabels([item['label'] for item in items])
        left = np.zeros(len(items))
        for j, name in enumerate(names):
            values = np.array([item['component']['wS_' + name] for item in items], dtype=float)
            if not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError('Weighted contributions must be finite and nonnegative.')
            ax.barh(range(len(items)), values, left=left, height=0.44, color=COMPONENT_COLORS[j],
                    hatch=COMPONENT_HATCHES[j], edgecolor='0.3', lw=0.35, label=COMPONENT_LABELS[name])
            left += values
        if not np.allclose(left, scores, atol=1e-10, rtol=0):
            raise ValueError('Weighted contributions do not match total scores.')
        ax.axvline(cutoff, color='0.2', ls=':', lw=1)
        ax.text(cutoff, -0.65, f"cutoff={cutoff:.3f}", ha='center', va='center', color='grey',
                fontsize=10, bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 0.5})
        for index, item in enumerate(items):
            component = item['component']
            retained = component['retained']
            if retained != (component['mixed_rank'] <= mixed_kept):
                raise ValueError('Displayed retention disagrees with the mixed rank.')
            if retained and scores[index] < cutoff - 1e-10 or not retained and scores[index] > cutoff + 1e-10:
                raise ValueError('Displayed score is inconsistent with the actor mixed-pool cutoff.')
            status = 'Retained' if retained else 'Filtered'
            ax.text(scores[index] + 0.012, index, '{:.3f}  {}'.format(scores[index], status),
                    ha='left', va='center', fontsize=12,
                    bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 0.5})
        self.legend(ax, ncol=2)
        ax.set_axisbelow(True)
        ax.grid(axis='x', color='0.9', lw=0.5)

    def rank_panel(self, ax, scene):
        items = self.items(scene)
        rows = []
        for item in items:
            component = item['component']
            identity = item['candidate_id'] if scene == 'spacecraft' else item['candidate_name']
            metric = item['metric']
            physical = ([metric['legacy_proximity_d_min_km'],
                         metric['legacy_geometry_plane_angle_deg'],
                         metric['legacy_reachability_required_delta_v_km_s']]
                        if scene == 'spacecraft' else
                        [metric['legacy_ground_track_min_surface_distance_km'],
                         metric['legacy_access_access_duration_s'],
                         metric['legacy_access_pass_count']])
            rows.append(['{} / {}'.format(item['label'], identity),
                         '{:.1f}'.format(physical[0]),
                         '{:.1f}'.format(physical[1]) if scene == 'spacecraft' else '{:.0f}'.format(physical[1]),
                         '{:.2f}'.format(physical[2]) if scene == 'spacecraft' else '{:.0f}'.format(physical[2]),
                         '{} → {}'.format(int(metric['distance_rank']), int(component['score_rank'])),
                         '{} / {}'.format('Yes' if component['retained'] else 'No',
                                          'Yes' if component['distance_retained'] else 'No')])
        ax.axis('off')
        ax.set_title('Physical evidence, ranks and retention', fontsize=9, pad=7)
        headers = (['Object / NORAD ID', 'Min. distance\n(km)', 'Plane angle\n(deg)',
                    'Cost proxy\n(km/s)', 'Rank\nD → S', 'Retained\nS / D']
                   if scene == 'spacecraft' else
                   ['Region center', 'Min. surface\ndistance (km)', 'Access\n(s)',
                    'Passes', 'Rank\nD → S', 'Retained\nS / D'])
        table = ax.table(cellText=rows,
                         colLabels=headers,
                         loc='center', cellLoc='center', colWidths=[0.28, 0.16, 0.14, 0.14, 0.14, 0.14],
                         bbox=[0.0, 0.31, 1.0, 0.66])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        for (row, column), cell in table.get_celld().items():
            cell.set_edgecolor('0.8')
            cell.set_linewidth(0.4)
            if row == 0:
                cell.set_facecolor('0.94')
                cell.set_height(cell.get_height() * 1.6)
            elif column == 0:
                cell.get_text().set_color(COLORS[row - 1])
        summary, meta = self.summary[scene], self.analysis['scenes'][scene]
        count, retained = int(summary['candidates']), int(summary['retained'])
        ax.text(0.5, 0.19, 'S: matching score; D: distance. Ranks use the complete pool of this counterpart type.',
            transform=ax.transAxes, ha='center', fontsize=8)
        ax.text(0.5, 0.08, '{} / {} retained; reduction: {:.1%}; same-budget set overlap: {} / {}'.format(
            retained, count, 1 - retained / count, summary['same_budget_overlap'], retained),
            transform=ax.transAxes, ha='center', fontsize=8)
        ax.text(0.5, -0.03, 'Mixed pool: {} / {} retained. Quality exclusions: {} / {} (before ranking).'.format(
            int(meta['mixed_retained']), int(meta['mixed_pool_count']),
            summary['data_quality_excluded'], summary['source_candidates']),
            transform=ax.transAxes, ha='center', fontsize=8, color='0.4')

    def panel(self, ax, scene, name):
        if name in ('eci_3d', 'lvlh_3d'):
            self.spatial_panel(ax, scene, name.split('_')[0])
        elif name == 'distance':
            self.time_panel(ax, scene)
        elif name == 'elevation':
            self.time_panel(ax, scene, elevation=True)
        elif name == 'groundtrack':
            self.ground_panel(ax, scene)
        elif name == 'score':
            self.score_panel(ax, scene)
        elif name == 'rank':
            self.rank_panel(ax, scene)

    def save_figure(self, figure, name):
        for extension in ('pdf', 'png'):
            figure.savefig(self.output_dir / (name + '.' + extension), dpi=300,
                           bbox_inches=None,
                           pad_inches=0.04, facecolor='white')

    def run(self, output_mode=OUTPUT_MODE):
        if output_mode not in ('show', 'savefig'):
            raise ValueError("OUTPUT_MODE must be 'show' or 'savefig'.")
        self.read_data()
        scene = self.scene
        style = {'font.family': 'Times New Roman', 'font.size': 12,
                 'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
                 'axes.linewidth': 0.6, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                 'interactive': False}
        with plt.rc_context(style):
            track_figure = plt.figure(figsize=TRACK_FIGSIZE)
            grid = track_figure.add_gridspec(2, 1, left=0.12, right=0.97,
                                            bottom=0.09, top=0.96, hspace=0.36)
            for index, name in enumerate(('groundtrack', 'elevation')):
                ax = track_figure.add_subplot(grid[index, 0])
                self.panel(ax, scene, name)

            score_figure = plt.figure(figsize=SCORE_FIGSIZE)
            score_figure.subplots_adjust(left=0.10, right=0.98, bottom=0.18, top=0.95)
            self.score_panel(score_figure.add_subplot(111), scene)
            figures = {'ground_elevation': track_figure, 'score': score_figure}
            if output_mode == 'savefig':
                self.save_figure(track_figure, 'results_matching_' + scene)
                self.save_figure(score_figure, 'results_matching_' + scene + '_score')
        if output_mode == 'show':
            plt.show()
        else:
            for figure in figures.values():
                plt.close(figure)
        return figures


if __name__ == '__main__':
    MatchingRegionsPlot(OUTPUT_DIR).run("show")
