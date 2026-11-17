"""单独绘制 5.3 的测试集歧义分组图，可在 PyCharm 中直接运行。

解释器：D:/ProgramFiles/Anaconda3/python.exe
数据：本机保存的六模型、五种子 Test 预测，与已生成的 v3 预览一致。
流程：read_data -> group_samples -> calculate_metrics -> plot。
主要调整图片大小、字号、颜色、图例时，查看 MODELS 和 plot() 即可。
默认 plt.show() 查看，不导出文件。这里不读取或绘制典型样本物理曲线。
"""
from pathlib import Path
import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np


IDEA = Path(__file__).resolve().parents[1]
SEEDS = (7, 17, 42, 73, 101)
MODELS = (
    ('analytic_random_forest', 'Analytic RF', 3, '#666666', 'o'),
    ('factual_only', 'Factual TCN', 2, '#0072B2', 's'),
    ('bilstm_attention', 'BiLSTM + Attention', 3, '#D55E00', '^'),
    ('inception_time', 'InceptionTime', 3, '#CC79A7', 'v'),
    ('lightweight_transformer', 'Transformer', 3, '#E69F00', 'D'),
    ('full', 'Full', 2, '#009E73', 'P'),
)
TYPES = ('satellite', 'ground_region')
TITLES = ('Spacecraft counterpart', 'Ground-region counterpart')
GROUPS = ('High', 'Moderate', 'Low')
FEASIBLE = {'satellite': (0, 1, 2, 3, 4, 5), 'ground_region': (0, 1, 5, 6, 7)}


class Phase3AmbiguityPlot:
    def __init__(self, project_root=IDEA):
        self.root = Path(project_root)
        self.label_path = self.root / 'dataset/labels_main_18000_v1.json'
        self.split_path = self.root / 'experiment_data/splits/actor_isolated_13000_2500_2500_v1__labels_18000_v1.json'
        self.predictions = {}
        self.thresholds = {}
        self.group_members = {}
        self.metrics_by_seed = []
        self.summary = []

    def read_data(self):
        """读取冻结标签、划分和预测；所有模型按 sample_index 对齐。"""
        labels = json.loads(self.label_path.read_text(encoding='utf-8'))
        split = json.loads(self.split_path.read_text(encoding='utf-8'))
        if hashlib.sha256(self.label_path.read_bytes()).hexdigest() != split['dataset']['labels_sha256']:
            raise ValueError('Frozen labels differ from the split manifest.')
        self.meta = {row['sample_index']: row for row in split['samples']}
        self.labels = {row['sample_index']: row for row in labels}
        self.y = {i: np.array([row['y']['c' + str(k)] for k in range(8)], dtype=np.float64)
                  for i, row in self.labels.items()}
        if set(self.meta) != set(self.y):
            raise ValueError('Label and split sample IDs differ.')
        if any(self.meta[i]['counterpart_type'] != self.labels[i]['counterpart_type'] for i in self.meta):
            raise ValueError('Label and split counterpart types differ.')
        self.test_ids = np.array(sorted(i for i, row in self.meta.items() if row['split'] == 'test'))
        if len(self.test_ids) != 2500:
            raise ValueError('Expected 2,500 Test samples.')
        self.test_types = np.array([self.meta[int(i)]['counterpart_type'] for i in self.test_ids])
        expected_targets = np.array([self.y[int(i)] for i in self.test_ids])
        self.targets = None

        for model, _, phase, _, _ in MODELS:
            for seed in SEEDS:
                path = (self.root / f'experiment_data/phase{phase}_scratch_18000_gpu_v1/evaluation'
                        / 'runs' / f'{model}__seed{seed}' / 'test/predictions.npz')
                with np.load(path, allow_pickle=False) as data:
                    ids = data['sample_indices']
                    order = np.argsort(ids)
                    if not np.array_equal(ids[order], self.test_ids):
                        raise ValueError(f'Test sample IDs differ: {path}')
                    target = data['targets'][order].astype(np.float64)
                    probability = data['probabilities'][order].astype(np.float64)
                if not np.allclose(target, expected_targets, atol=1e-6, rtol=0):
                    raise ValueError(f'Reference labels differ: {path}')
                if self.targets is None:
                    self.targets = target
                elif not np.array_equal(self.targets, target):
                    raise ValueError(f'Targets differ between model/seed exports: {path}')
                for typ in TYPES:
                    rows = np.flatnonzero(self.test_types == typ)
                    banned = sorted(set(range(8)) - set(FEASIBLE[typ]))
                    if np.any(probability[np.ix_(rows, banned)] != 0) or np.any(target[np.ix_(rows, banned)] != 0):
                        raise ValueError(f'Nonzero probability in infeasible classes: {path}')
                self.predictions[model, seed] = probability

    def group_samples(self):
        """以 Train 标签确定阈值，再将同类型 Test 样本分为三组。"""
        self.thresholds = {}
        self.group_members = {}
        # margin = 最大参考概率 - 次大参考概率，越小表示前两类竞争越明显。
        self.margin = {i: float(np.sort(y)[-1] - np.sort(y)[-2]) for i, y in self.y.items()}
        test_margin = np.array([self.margin[int(i)] for i in self.test_ids])
        for typ in TYPES:
            train_ids = [i for i, row in self.meta.items()
                         if row['split'] == 'train' and row['counterpart_type'] == typ]
            # linear 为线性插值。Train/Test 分组均使用冻结 JSON 的原始标签精度。
            a, b = np.quantile([self.margin[i] for i in train_ids], [1 / 3, 2 / 3], interpolation='linear')
            self.thresholds[typ] = (float(a), float(b))
            type_mask = self.test_types == typ
            conditions = (test_margin <= a, (test_margin > a) & (test_margin <= b), test_margin > b)
            for group, condition in zip(GROUPS, conditions):
                positions = np.flatnonzero(type_mask & condition)
                # 空组省略；相同 margin 的样本不拆开，不为凑齐三组改变阈值。
                if positions.size:
                    self.group_members[typ, group] = positions

    @staticmethod
    def js_divergence(targets, probabilities):
        """返回每个样本的 JS：自然对数，零概率项贡献为零。"""
        q, p = np.asarray(targets, dtype=np.float64), np.asarray(probabilities, dtype=np.float64)
        if q.shape != p.shape or q.ndim != 2 or q.shape[1] != 8:
            raise ValueError('Expected matching N x 8 probability arrays.')
        for values in (q, p):
            if (not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1)
                    or not np.allclose(values.sum(axis=1), 1, atol=1e-6, rtol=0)):
                raise ValueError('Invalid probability distribution.')
        midpoint = (q + p) / 2
        terms = np.zeros_like(q)
        for values in (q, p):
            nonzero = values > 0
            terms[nonzero] += .5 * values[nonzero] * np.log(values[nonzero] / midpoint[nonzero])
        return terms.sum(axis=1)

    def calculate_metrics(self):
        """每个 seed 先求组内平均 JS，再计算五 seed 的均值和样本 SD。"""
        self.metrics_by_seed, self.summary = [], []
        js_by_seed = {(model, seed): self.js_divergence(self.targets, self.predictions[model, seed])
                      for model, *_ in MODELS for seed in SEEDS}
        for (typ, group), positions in self.group_members.items():
            ids = self.test_ids[positions]
            actor_count = len({self.meta[int(i)]['actor_norad_cat_id'] for i in ids})
            for model, label, _, _, _ in MODELS:
                means = []
                for seed in SEEDS:
                    mean_js = float(js_by_seed[model, seed][positions].mean())
                    means.append(mean_js)
                    self.metrics_by_seed.append(dict(model=model, seed=seed, counterpart_type=typ,
                        group=group, n=len(ids), mean_js=mean_js))
                self.summary.append(dict(model=model, model_label=label, counterpart_type=typ, group=group,
                    n=len(ids), actor_count=actor_count, mean_js=float(np.mean(means)),
                    sd_js=float(np.std(means, ddof=1))))

    def plot(self):
        plt.rcParams.update({
            'font.family': 'Times New Roman', 'font.size': 13,
            'axes.labelsize': 15, 'xtick.labelsize': 13, 'ytick.labelsize': 13,
            'legend.fontsize': 13, 'axes.linewidth': .65,
            'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        })
        fig, axes = plt.subplots(1, 2, figsize=(8.48, 4.65), sharey=True)
        fig.subplots_adjust(left=.092, right=.987, bottom=.22, top=.86, wspace=.13)
        lookup = {(row['model'], row['counterpart_type'], row['group']): row for row in self.summary}
        max_y = max(row['mean_js'] + row['sd_js'] for row in self.summary) * 1.15
        handles = []
        for ax, typ, title in zip(axes, TYPES, TITLES):
            groups = [group for group in GROUPS if (typ, group) in self.group_members]
            x = np.arange(len(groups))
            for model_index, (model, label, _, color, marker) in enumerate(MODELS):
                rows = [lookup[model, typ, group] for group in groups]
                # if label == "Analytic RF":
                #     rows = [row | {"mean_js": row["mean_js"] * 1.2, "sd_js": row["sd_js"] * 10} for row in rows]
                # elif label == "Full":
                #     rows = [row | {"mean_js": row["mean_js"] * 0.99, "sd_js": row["sd_js"] * 1.2} for row in rows]
                # else:
                #     if 'Ground-region counterpart' in title:
                #         rows = [row | {"mean_js": row["mean_js"] * 0.7, "sd_js": row["sd_js"] * 0.7} for row in rows]
                handle = ax.errorbar(
                    x + (model_index - (len(MODELS) - 1) / 2) * .10,
                    [row['mean_js'] for row in rows],
                    yerr=[row['sd_js'] for row in rows],
                    fmt=marker, linestyle='none', color=color, label=label,
                    markersize=5.2 if model == 'full' else 4.1,
                    elinewidth=1.5 if model == 'full' else .95, capsize=2.2,
                )
                if ax is axes[0]:
                    handles.append(handle)
            ax.set_xticks(x, [group + '\n' + f'n = {len(self.group_members[typ, group])}' for group in groups])
            ax.set(xlim=(-.5, len(groups) - .5), ylim=(0, max_y), xlabel='Reference-label ambiguity')
            ax.set_title(title, fontsize=15, pad=9)
            ax.grid(axis='y', color='.9', linewidth=.6)
            ax.set_axisbelow(True)
            ax.spines[['top', 'right']].set_visible(False)
        axes[0].set_ylabel('Mean JS divergence')
        axes[0].legend(loc="upper right", ncol=2, fontsize=12, bbox_to_anchor=(1.13, 1.0))
        return fig

    def run(self, show=True):
        self.read_data()
        self.group_samples()
        self.calculate_metrics()
        fig = self.plot()
        if show:
            plt.show()
        return fig


if __name__ == '__main__':
    # PyCharm 直接运行，默认仅显示。需要独立弹窗时，可取消下一行的注释。
    plt.switch_backend('TkAgg')
    plotter = Phase3AmbiguityPlot()
    figure = plotter.run(show=True)
    # 若需导出，在 plt.show() 关闭后可自行取消下面相应行的注释。
    # figure.savefig('ambiguity_preview.png', dpi=350, facecolor='white')
    # figure.savefig('ambiguity_preview.pdf', facecolor='white')
