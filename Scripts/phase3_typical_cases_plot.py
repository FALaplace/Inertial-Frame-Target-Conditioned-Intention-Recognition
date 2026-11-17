"""
C4/C2 包含 ECI 与 LVLH 轨迹；C1/C7 的第二张子图为星下点轨迹。
两种模式均保留原有物理时序和概率热图。
在 main 中修改 mode 切换；C1/C7 的时序显示仰角和地面目标可见性。
PyCharm 解释器：D:/ProgramFiles/Anaconda3/python.exe
直接运行即可通过 plt.show() 显示，默认不保存图片。
读取已生成的 v3 本机预览数据，使用 seed 42；不重新选样或运行模型。
从冻结数据集读取本个例的原始位置和速度，不加载整个数据集到内存。
调整整体布局看 plot()，三维轨迹看 trajectory_axes()，星下点看 ground_track_axis()，
物理曲线看 physical_axes()，
热图看 probability_axis()。
"""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


IDEA = Path(__file__).resolve().parents[1]
MODELS = (
    ('analytic_random_forest', 'Analytic RF'),
    ('factual_only', 'Factual TCN'),
    ('bilstm_attention', 'BiLSTM + Attention'),
    ('inception_time', 'InceptionTime'),
    ('lightweight_transformer', 'Transformer'),
    ('full', 'Full'),
)
CLASSES = tuple('C' + str(i) for i in range(8))
CASE_IDS = {'c4_c2': 17985, 'c7_c1': 12976}
FEASIBLE = {'satellite': (0, 1, 2, 3, 4, 5), 'ground_region': (0, 1, 5, 6, 7)}


class Phase3TypicalCasesPlot:
    def __init__(self, project_root=IDEA, seed=42):
        self.root = Path(project_root)
        self.seed = seed
        self.probability_path = self.root / 'experiment_data/baseline_test_analysis_20260917_v3/case_probabilities.csv'
        self.physical_path = self.root / 'experiment_data/baseline_test_analysis_20260917/preparation/physical_cases.json'
        self.dataset_path = self.root / 'dataset/x_samples_frozen_18000_v1.json'
        self.physical = {}
        self.probabilities = {}
        self.trajectories = {}

    def read_data(self, cases=('c4_c2',)):
        """按样本标识、模型和 seed 读取既有数据，构造每个个例的 7×8 矩阵。"""
        if len(cases) != 1 or cases[0] not in CASE_IDS:
            raise ValueError('Choose one case: c4_c2 or c7_c1.')
        physical = json.loads(self.physical_path.read_text(encoding='utf-8'))
        with self.probability_path.open(encoding='utf-8-sig', newline='') as stream:
            rows = [row for row in csv.DictReader(stream) if int(row['seed']) == self.seed]

        for case in cases:
            index = CASE_IDS[case]
            data = physical[str(index)]
            typ = data['counterpart_type']
            expected_type = 'satellite' if case == 'c4_c2' else 'ground_region'
            if typ != expected_type:
                raise ValueError(f'Unexpected counterpart type for {case}: {typ}')
            selected = [row for row in rows if row['case'] == case and int(row['sample_index']) == index]
            by_model = {row['model']: row for row in selected}
            if len(selected) != len(MODELS) or set(by_model) != {model for model, _ in MODELS}:
                raise ValueError(f'Expected one prediction per model: {case}, seed {self.seed}')
            reference = None
            probabilities = []
            for model, _ in MODELS:
                row = by_model[model]
                for field in ('sample_id', 'actor_norad_cat_id', 'counterpart_type'):
                    if row[field] != str(data[field]):
                        raise ValueError(f'Physical/prediction identity mismatch: {case}/{field}')
                y = np.array([float(row['y_' + c]) for c in CLASSES])
                p = np.array([float(row['p_' + c]) for c in CLASSES])
                if reference is None:
                    reference = y
                elif not np.array_equal(reference, y):
                    raise ValueError(f'Reference labels differ between models: {case}')
                probabilities.append(p)
            values = np.vstack([reference] + probabilities)
            if (not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1)
                    or not np.allclose(values.sum(axis=1), 1, atol=1e-6, rtol=0)):
                raise ValueError(f'Invalid probability matrix: {case}')
            if not np.allclose(reference, data['y'], atol=1e-6, rtol=0):
                raise ValueError(f'Physical/prediction reference labels differ: {case}')
            banned = sorted(set(range(8)) - set(FEASIBLE[typ]))
            if np.any(values[:, banned] != 0):
                raise ValueError(f'Nonzero probability in infeasible classes: {case}')

            time = np.asarray(data['time_offset_s'])
            if len(time) < 2 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
                raise ValueError(f'Invalid physical time grid: {case}')
            if data['maneuvers'] and not data['counterfactual_valid']:
                raise ValueError(f'Counterfactual trajectory is missing: {case}')
            fields = ('distance_km', 'range_rate_km_s') if typ == 'satellite' else (
                'elevation_deg', 'access_flag', 'actor_latitude_deg', 'actor_longitude_deg')
            for branch in ('factual', 'counterfactual'):
                for field in fields:
                    series = np.asarray(data[branch][field])
                    if series.shape != time.shape or not np.isfinite(series).all():
                        raise ValueError(f'Invalid physical series: {case}/{branch}/{field}')
            self.physical[case] = data
            self.probabilities[case] = values
            self.trajectories[case] = self.read_trajectory(index, data)

    def read_trajectory(self, index, data):
        """冻结 JSON 数组为每行一个样本；只解析指定行，并核对原始样本哈希。"""
        sample = None
        current = 0
        with self.dataset_path.open(encoding='utf-8') as stream:
            for line in stream:
                record = line.strip().rstrip(',')
                if record in ('', '[', ']'):
                    continue
                if current == index:
                    sample = json.loads(record)
                    break
                current += 1
        if sample is None:
            raise ValueError(f'Sample {index} not found in {self.dataset_path}')
        digest = hashlib.sha256(json.dumps(sample, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
        if digest != data['metadata']['raw_sample_canonical_sha256']:
            raise ValueError('Raw trajectory does not match the physical/prediction case.')

        result = {}
        for role in ('actor', 'counterpart'):
            records = sample[role + '_trajectory']
            if [point['t'] for point in records] != data['timestamps_utc']:
                raise ValueError(f'{role} trajectory timestamps do not match the physical curves.')
            for field in ('r', 'v'):
                values = np.asarray([point[field] for point in records], dtype=np.float64)
                if values.shape != (len(data['time_offset_s']), 3) or not np.isfinite(values).all():
                    raise ValueError(f'Invalid {role} trajectory: {field}')
                result[role + '_' + field] = values

        result['actor_lvlh'] = self.target_centered_lvlh(
            result['actor_r'], result['counterpart_r'], result['counterpart_v'])
        relative_r = result['actor_r'] - result['counterpart_r']
        relative_v = result['actor_v'] - result['counterpart_v']
        distance = np.linalg.norm(relative_r, axis=1)
        rate = np.einsum('ij,ij->i', relative_r, relative_v) / distance
        for field, values in (('distance_km', distance), ('range_rate_km_s', rate)):
            if not np.allclose(values, data['factual'][field], atol=1e-6, rtol=1e-6):
                raise ValueError(f'Raw trajectory and cached {field} disagree.')
        return result

    @staticmethod
    def target_centered_lvlh(actor_r, counterpart_r, counterpart_v):
        """与 human-annotation-tool/orbit_math.py 的 target_centered_rtn 约定一致。

        每个时刻以 counterpart 为原点：R 为径向向外，N 为 r×v 方向，T=N×R。
        地面目标使用随地球自转的惯性速度，此时 R/T/N 对应地心径向向上/东/北。
        这里只转换相对位置；range rate 仍使用原始相对位置与速度计算。
        """
        radius = np.linalg.norm(counterpart_r, axis=1, keepdims=True)
        normal = np.cross(counterpart_r, counterpart_v)
        angular_momentum = np.linalg.norm(normal, axis=1, keepdims=True)
        if np.any(radius < 1e-12) or np.any(angular_momentum < 1e-12):
            raise ValueError('Cannot construct LVLH axes from degenerate counterpart states.')
        radial = counterpart_r / radius
        normal = normal / angular_momentum
        transverse = np.cross(normal, radial)
        transverse /= np.linalg.norm(transverse, axis=1, keepdims=True)
        basis = np.stack([radial, transverse, normal], axis=1)
        return np.einsum('nij,nj->ni', basis, actor_r - counterpart_r)

    @staticmethod
    def format_3d_axis(ax, points, labels):
        """按真实 km 比例设置三轴，避免把轨迹沿某个方向拉伸。"""
        low, high = points.min(axis=0), points.max(axis=0)
        # 三轴使用相同的长度范围；LVLH 的窄径向跨度也不会挤压刻度文字。
        center = (low + high) / 2
        radius = max(float(np.max(high - low)), 1.0) * .58
        limits = np.column_stack([center - radius, center + radius])
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])
        ax.set_zlim(limits[2])
        ax.set_box_aspect((1, 1, 1))
        ax.set_proj_type('ortho')
        for axis, label in zip((ax.xaxis, ax.yaxis, ax.zaxis), labels):
            axis.set_major_locator(MaxNLocator(nbins=3))
            axis.set_rotate_label(False)
            axis.set_label_text(label, fontsize=10)
            axis.labelpad = 5
            axis.pane.fill = False
            axis.pane.set_edgecolor('.85')
        ax.tick_params(labelsize=9, pad=1)

    def trajectory_axes(self, fig, eci_spec, lvlh_spec, case):
        """左图保持事实 ECI；地面个例的右图显示星下点，航天器个例保持 LVLH。"""
        data = self.trajectories[case]
        actor, counterpart, lvlh = data['actor_r'], data['counterpart_r'], data['actor_lvlh']
        eci = fig.add_subplot(eci_spec, projection='3d')
        eci.plot(*actor.T, color='#0072B2', lw=1.25, label='Actor')
        eci.plot(*counterpart.T, color='#666666', ls='--', lw=1.15, label='Counterpart')
        self.format_3d_axis(eci, np.vstack([actor, counterpart]), ('X (km)', 'Y (km)', 'Z (km)'))
        eci.view_init(elev=40, azim=-15)
        eci.set_title('ECI trajectories', fontsize=11, pad=24)
        eci.legend(loc='upper right', frameon=False, fontsize=9)

        if self.physical[case]['counterpart_type'] == 'ground_region':
            for event in self.physical[case]['maneuvers']:
                index = np.argmin(np.abs(np.asarray(self.physical[case]['time_offset_s']) - event['time_offset_s']))
                eci.scatter(*actor[index], marker='D', s=20, color='#333333', depthshade=False)
            return eci, self.ground_track_axis(fig, lvlh_spec, case)

        relative = fig.add_subplot(lvlh_spec, projection='3d')
        relative.plot(*lvlh.T, color='#0072B2', lw=1.25, label='Actor')
        relative.scatter(0, 0, 0, color='#666666', marker='*', s=65, depthshade=False,
                         label='Counterpart (origin)')
        self.format_3d_axis(relative, np.vstack([lvlh, np.zeros((1, 3))]), ('R (km)', 'T (km)', 'N (km)'))
        relative.view_init(elev=22, azim=-50)
        relative.set_title('Counterpart-centered LVLH (RTN)', fontsize=11, pad=24)
        for event in self.physical[case]['maneuvers']:
            index = np.argmin(np.abs(np.asarray(self.physical[case]['time_offset_s']) - event['time_offset_s']))
            for ax, points in ((eci, actor), (relative, lvlh)):
                ax.scatter(*points[index], marker='D', s=20, color='#333333', depthshade=False)
        relative.legend(loc='upper right', frameon=False, fontsize=9)
        return eci, relative

    @staticmethod
    def ground_track_coordinates(latitude, longitude, center_longitude):
        """仅展开绘图经度，在 Region 对面的接缝插入断点；不丢弃原有轨迹点。"""
        longitude = center_longitude + (np.asarray(longitude) - center_longitude + 180) % 360 - 180
        latitude = np.asarray(latitude)
        breaks = np.flatnonzero(np.abs(np.diff(longitude)) > 180) + 1
        return np.insert(longitude, breaks, np.nan), np.insert(latitude, breaks, np.nan)

    def ground_track_axis(self, fig, spec, case):
        data = self.physical[case]
        region = np.column_stack((data['counterpart_latitude_deg'], data['counterpart_longitude_deg']))
        if region.shape != (len(data['time_offset_s']), 2) or not np.isfinite(region).all():
            raise ValueError('Invalid cached region-center coordinates.')
        region_latitude, region_longitude = region[0]
        longitude_drift = (region[:, 1] - region_longitude + 180) % 360 - 180
        if (not np.allclose(region[:, 0], region_latitude, atol=1e-6, rtol=0)
                or np.any(np.abs(longitude_drift) > 1e-6)):
            raise ValueError('The region center must be fixed in Earth coordinates.')

        ax = fig.add_subplot(spec)
        visible_latitudes = [np.array([region_latitude])]
        for branch, label, linestyle, color in (('factual', 'Factual', '-', '#0072B2'),
                                               ('counterfactual', 'Counterfactual', '--', '#D55E00')):
            latitude = np.asarray(data[branch]['actor_latitude_deg'])
            longitude = np.asarray(data[branch]['actor_longitude_deg'])
            if np.any(np.abs(latitude) > 90) or np.any(np.abs(longitude) > 180):
                raise ValueError(f'Geographic coordinates outside degree bounds: {branch}')
            x, y = self.ground_track_coordinates(latitude, longitude, region_longitude)
            ax.plot(x, y, color=color, ls=linestyle, lw=1.15, label=label)
            inside = np.abs(x - region_longitude) <= 45
            visible_latitudes.append(y[inside])
            ax.plot(x[0], y[0], 'o', ms=3, mfc='white', mec=color)
            ax.plot(x[-1], y[-1], 's', ms=3, color=color)
            if branch == 'factual':
                for event in data['maneuvers']:
                    index = np.argmin(np.abs(np.asarray(data['time_offset_s']) - event['time_offset_s']))
                    event_longitude = region_longitude + (longitude[index] - region_longitude + 180) % 360 - 180
                    ax.plot(event_longitude, latitude[index], 'D', ms=4, color='#333333')
        ax.plot(region_longitude, region_latitude, '*', ms=9, color='#666666',
                mec='white', mew=.5, label='Region center', zorder=5)
        latitude_extent = np.concatenate(visible_latitudes)
        margin = max(2.0, float(np.ptp(latitude_extent)) * .08)
        ax.set(xlim=(region_longitude - 45, region_longitude + 45),
               ylim=(max(-90, latitude_extent.min() - margin), min(90, latitude_extent.max() + margin)),
               xlabel='Longitude (deg)', ylabel='Latitude (deg)')
        # 方形绘图区，不强制经纬度跨度相等，也不改变其他面板的布局。
        ax.set_box_aspect(1)
        # 第二子图整体右移画布宽度的 2.5%，避开左侧 ECI 的坐标轴文字。
        ax.set_position(ax.get_position().translated(.035, 0))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.tick_params(labelsize=9)
        ax.set_title('Sub-satellite ground track', fontsize=11, pad=12)
        ax.grid(color='.9', lw=.5)
        ax.legend(loc='upper right', frameon=True, fontsize=9, handlelength=1.6)
        return ax

    def physical_axes(self, fig, spec, case):
        data = self.physical[case]
        grid = spec.subgridspec(2, 1, hspace=.12)
        axes = [fig.add_subplot(grid[0]), fig.add_subplot(grid[1])]
        if data['counterpart_type'] == 'satellite':
            variables = [('distance_km', 'Distance (km)'), ('range_rate_km_s', 'Range rate (km/s)')]
        else:
            variables = [('elevation_deg', 'Elevation (deg)'), ('access_flag', 'Access (0/1)')]
        x = np.asarray(data['time_offset_s']) / 60
        for row, (ax, (field, ylabel)) in enumerate(zip(axes, variables)):
            for branch, label, linestyle, color in (('factual', 'Factual', '-', '#0072B2'),
                                                   ('counterfactual', 'Counterfactual', '--', '#D55E00')):
                ax.plot(x, data[branch][field], linestyle, color=color, label=label, lw=1.15,
                        drawstyle='steps-post' if field == 'access_flag' else 'default')
            for event in data['maneuvers']:
                ax.axvline(event['time_offset_s'] / 60, color='.35', ls=':', lw=.9)
            if field == 'elevation_deg':
                threshold = data['metadata']['access_elevation_threshold_deg']
                ax.axhline(threshold, color='.6', ls=':', lw=.7)
                ax.text(.02, threshold + 5, f'Access: \u2265 {threshold:g}\u00b0',
                        transform=ax.get_yaxis_transform(), fontsize=8, ha='left', va='bottom')
            if field == 'access_flag':
                ax.set_yticks([0, 1])
                ax.set_ylim(-.12, 1.12)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_xlim(x[0], x[-1])
            ax.set_xticks(np.arange(0, 181, 30))
            ax.grid(color='.91', linewidth=.5)
            ax.tick_params(labelsize=9)
            if row == 0:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel('Time (min)', fontsize=10)
        return axes

    def probability_axis(self, fig, spec, case):
        """每格为概率；第一行是参考标签，其余六行为模型预测。灰色为不适用类。"""
        values = self.probabilities[case]
        feasible = FEASIBLE[self.physical[case]['counterpart_type']]
        mask = np.ones_like(values, dtype=bool)
        mask[:, feasible] = False
        cmap = plt.get_cmap('cividis').copy()
        cmap.set_bad('#dddddd')
        ax = fig.add_subplot(spec)
        mesh = ax.pcolormesh(np.ma.array(values, mask=mask), cmap=cmap, vmin=0, vmax=1,
                            edgecolors='white', linewidth=.4, rasterized=False)
        ax.set(xlim=(0, 8), ylim=(7, 0))
        ax.set_xticks(np.arange(8) + .5, CLASSES)
        ax.set_yticks(np.arange(7) + .5, ['Reference'] + [label for _, label in MODELS])
        ax.tick_params(length=0, labelsize=10, pad=4)
        ax.set_xlabel('Intention class', fontsize=10)
        for row in range(7):
            for column in range(8):
                if column in feasible:
                    value = values[row, column]
                    ax.text(column + .5, row + .5, f'{value:.2f}', fontsize=9, ha='center', va='center',
                            color='white' if value < .48 else 'black')
                else:
                    ax.text(column + .5, row + .5, '\u2014', fontsize=9, ha='center', va='center', color='.4')
        return ax, mesh

    def plot(self, cases=('c4_c2',)):
        """两种模式共用横向四部分；orbit_grid.wspace 控制前两张轨迹图的间距。"""
        if len(cases) != 1 or cases[0] not in CASE_IDS:
            raise ValueError('Choose one case: c4_c2 or c7_c1.')
        plt.rcParams.update({
            'font.family': 'Times New Roman', 'font.size': 10, 'axes.labelsize': 11,
            'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9,
            'axes.linewidth': .65, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        })
        case = cases[0]
        fig = plt.figure(figsize=(14.6, 3.7))
        outer = fig.add_gridspec(1, 3, left=.018, right=.937, bottom=.13, top=.93,
                                wspace=.36, width_ratios=[2.0, .87, 1.13])
        # 轨迹图共用左侧区域，独立收紧间距；右侧留出模型名称需要的空间。
        orbit_grid = outer[0, 0].subgridspec(1, 2, wspace=.20)
        self.trajectory_axes(fig, orbit_grid[0, 0], orbit_grid[0, 1], case)
        physical = self.physical_axes(fig, outer[0, 1], case)
        heatmap, mesh = self.probability_axis(fig, outer[0, 2], case)
        handles, labels = physical[0].get_legend_handles_labels()
        handles.append(plt.Line2D([], [], color='.35', ls=':', lw=.9))
        labels.append('Maneuver')
        physical[0].legend(handles, labels, loc='upper right', frameon=False, handlelength=1.6, fontsize=9)
        if case == 'c4_c2':
            physical[0].set_ylim(top=physical[0].get_ylim()[1] * 1.35)
        else:
            physical[0].set_ylim(-90, 90)
        position = heatmap.get_position()
        bar = fig.colorbar(mesh, cax=fig.add_axes([position.x1 + .012, position.y0, .010, position.height]),
                          orientation='vertical', ticks=[0, .25, .5, .75, 1])
        bar.solids.set_rasterized(False)
        bar.set_label(f'Probability (seed {self.seed})', fontsize=9, labelpad=3)
        bar.ax.tick_params(labelsize=8)
        # fig.text(position.x0 + position.width / 2, .008, 'Grey cells: infeasible classes', ha='center', fontsize=8.5)
        return fig

    def run(self, show=True, cases=('c4_c2',)):
        self.read_data(cases)
        fig = self.plot(cases)
        if show:
            plt.show()
        return fig


if __name__ == '__main__':
    # plt.switch_backend('TkAgg')
    mode = 'c7_c1'  # 'c4_c2'：绘制 C4/C2；'c7_c1'：绘制 C1/C7（沿用数据中的名称）。
    plotter = Phase3TypicalCasesPlot(seed=42)
    figure = plotter.run(show=True, cases=(mode,))
