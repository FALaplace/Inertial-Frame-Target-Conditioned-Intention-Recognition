"""航天器两张横排图：ECI 五组配对，以及 LVLH、距离、相对速度和评分。"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

OUTPUT_MODE = "show"  # "show": 只显示；"savefig": 保存 PDF/PNG，不弹窗。
OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'experiment_data/matching_score_legacy_20261019'
ECI_FIGSIZE = (18, 4.2)
ANALYSIS_FIGSIZE = (18, 4.2)

COLORS = ('#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00')
ANALYSIS_COLORS = ('#1B4F72', '#9C3B10', '#145A32', '#6C3483', '#7D6608')
MARKERS = ('^', 'v', 'D', 'P', 'X')
COMPONENT_COLORS = ('#332288', '#44AA99', '#DDCC77', '#CC6677', '#882255')
COMPONENT_HATCHES = ('', '///', '\\\\', 'xx', '..')
COMPONENTS = {'spacecraft': ('prox', 'geom', 'reach', 'man', 'unc'),
              'regions': ('ground', 'access', 'revisit', 'cov', 'unc')}
COMPONENT_LABELS = {
    'prox': 'Proximity', 'geom': 'Geometry', 'reach': 'Maneuver-cost proxy',
    'man': 'Pre/post approach', 'unc': 'Uncertainty', 'ground': 'Ground proximity',
    'access': 'Access', 'revisit': 'Revisit', 'cov': 'Pre/post access',
}


class MatchingSpacecraftPlot:
    def __init__(self, output_dir):
        self.scene = 'spacecraft'
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
                if not isinstance(component['retained'], bool) or component['retained'] != (
                        component['mixed_rank'] <= mixed_kept):
                    raise ValueError('Retention must follow the stored mixed TopK rank, including ties.')
                if scene == 'regions':
                    latitude, longitude = float(item['latitude_deg']), float(item['longitude_deg'])
                    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
                        raise ValueError('Region-center coordinates are invalid.')
            if not all(np.isfinite(self.array(scene, key)).all()
                       for key in ('times_s', 'actor_f', 'candidates', 'candidate_rtn')):
                raise ValueError('Plotting states must be finite: ' + scene)

    def array(self, scene, name):
        return self.arrays[scene + '_' + name]

    def items(self, scene):
        return self.analysis['scenes'][scene]['display']

    def minutes(self, scene):
        time = self.array(scene, 'times_s')
        return (time - time[0]) / 60.0

    @staticmethod
    def legend(ax, handles=None, ncol=2, loc='upper right', fontsize=12):
        ax.legend(handles=handles, loc=loc, ncol=ncol, fontsize=fontsize,
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

    def spatial_panel(self, ax, scene, frame, item_index=None, reference_points=None):
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
            title, labels = 'ECI J2000', ('X (km)', 'Y (km)', 'Z (km)')
        else:
            ax.scatter(0, 0, 0, color='black', s=25, depthshade=False)
            handles.append(Line2D([], [], marker='o', color='black', ls='none', label='Actor F'))
            points.append(np.zeros((1, 3)))
            candidates = self.array(scene, 'candidate_rtn')
            title, labels = 'Actor-centered LVLH', ('R (km)', 'T (km)', 'N (km)')
        for i, item in enumerate(self.items(scene)):
            if item_index is not None and i != item_index:
                continue
            xyz = candidates[int(item['array_index'])]
            color = COLORS[0] if frame == 'eci' else ANALYSIS_COLORS[i]
            label = 'Target {}'.format(i + 1) if frame == 'eci' else item['label']
            handles.append(self.orbit(ax, xyz, color, label, marker=MARKERS[i]))
            points.append(xyz)
        self.equal_space(ax, points if reference_points is None else reference_points)
        ax.set(xlabel=labels[0], ylabel=labels[1], zlabel=labels[2])
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.labelpad = 1.5
        self.legend(ax, handles, ncol=1 if item_index is not None else 3, loc='upper left')

    def time_panel(self, ax, scene, quantity='distance'):
        if quantity not in ('distance', 'relative_speed'):
            raise ValueError("quantity must be 'distance' or 'relative_speed'.")
        minutes = self.minutes(scene)
        state_slice = slice(0, 3) if quantity == 'distance' else slice(3, 6)
        candidates = self.array(scene, 'candidates')[..., state_slice]
        actor = self.array(scene, 'actor_f')[:, state_slice]
        marker_indices = np.linspace(0, len(minutes) - 1, min(10, len(minutes)), dtype=int)
        handles = []
        for i, item in enumerate(self.items(scene)):
            counterpart = candidates[int(item['array_index'])]
            # Relative speed is ||v_actor - v_counterpart|| in J2000
            values = np.linalg.norm(actor - counterpart, axis=1)
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError('Distance and relative speed must be finite and nonnegative.')
            line, = ax.plot(minutes, values, color=ANALYSIS_COLORS[i], lw=1.0,
                            marker=MARKERS[i], markevery=marker_indices, ms=5, label=item['label'])
            handles.append(line)
        ax.set(xlim=(minutes[0], minutes[-1]), xlabel='Time (min)',
               ylabel='Distance (km)' if quantity == 'distance' else 'Relative speed (km/s)')
        ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
        ax.set_xticks([0, 30, 60, 90, 120, 150, 180])
        ax.grid(color='0.9', lw=0.5)
        self.legend(ax, handles, ncol=3)

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
        ax.text(cutoff, -0.65, f"cutoff={cutoff:.3f}", ha='center', va='center',c="grey",
                fontsize=10, bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 0.5})
        for index, item in enumerate(items):
            component = item['component']
            retained = component['retained']
            status = 'Retained' if retained else 'Filtered'
            ax.text(scores[index] + 0.012, index, '{:.3f}  {}'.format(scores[index], status),
                    ha='left', va='center', fontsize=12,
                    bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 0.5})
        self.legend(ax, ncol=2)
        ax.set_axisbelow(True)
        ax.grid(axis='x', color='0.9', lw=0.5)

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
                 'axes.linewidth': 0.6, 'pdf.fonttype': 42, 'ps.fonttype': 42}
        with plt.rc_context(style):
            # eci_figure = plt.figure(figsize=(18, 4.2))
            # grid = eci_figure.add_gridspec(1, 5, left=0.015, right=0.97, bottom=0.1, top=0.96, wspace=0.1)
            # radius = self.analysis['parameters']['shared']['earth_radius_km']
            # # All five pairs use identical limits, equal axis scales and camera angles.
            # reference = [np.array([[-radius] * 3, [radius] * 3]), self.array(scene, 'actor_f')[:, :3]]
            # reference.extend(self.array(scene, 'candidates')[int(item['array_index']), :, :3]
            #                  for item in self.items(scene))
            # for index in range(5):
            #     ax = eci_figure.add_subplot(grid[0, index], projection='3d')
            #     self.spatial_panel(ax, scene, 'eci', item_index=index, reference_points=reference)

            analysis_figure = plt.figure(figsize=(18, 4.2))
            grid = analysis_figure.add_gridspec(1, 6, width_ratios=(1.0, 0.08, 1.0, 0.05, 1.0, 1.28),
                                                left=0.025, right=0.99, bottom=0.18, top=0.9, wspace=0.12)
            names = ('lvlh', 'distance', 'relative_speed', 'score')
            for index, name in enumerate(names):
                column = (0, 2, 4, 5)[index]
                ax = analysis_figure.add_subplot(grid[0, column], projection='3d' if name == 'lvlh' else None)
                if name == 'lvlh':
                    self.spatial_panel(ax, scene, 'lvlh')
                elif name in ('distance', 'relative_speed'):
                    self.time_panel(ax, scene, quantity=name)
                else:
                    self.score_panel(ax, scene)
            # figures = {'eci_pairs': eci_figure, 'analysis': analysis_figure}
            # if output_mode == 'savefig':
            #     self.save_figure(eci_figure, 'results_matching_spacecraft_eci_pairs')
            #     self.save_figure(analysis_figure, 'results_matching_spacecraft')
        if output_mode == 'show':
            plt.show()
        # else:
        #     for figure in figures.values():
        #         plt.close(figure)
        # return figures


if __name__ == '__main__':
    MatchingSpacecraftPlot(OUTPUT_DIR).run("show")
