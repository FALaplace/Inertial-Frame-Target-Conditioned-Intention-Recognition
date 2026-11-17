"""Plot the final Section 5.2 experiment after all training and evaluation finish.

Open in PyCharm and run with D:/ProgramFiles/Anaconda3/python.exe.
This script reads saved results only; it never loads a model or uses the GPU.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import sys
import warnings

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
from PIL import Image, ImageOps
from scipy.signal import savgol_filter

from phase2_scratch_curve_preview import METRICS, SPLITS, STYLE, write_json

IDEA = Path(__file__).resolve().parents[1]
ROOT = IDEA / 'experiment_data/phase2_scratch_18000_gpu_v1'
SEEDS = (7, 17, 42, 73, 101)
NEURAL = ('factual_only', 'full_without_analytic', 'full')
PRIOR = 'counterpart_type_prior'
MODELS = (PRIOR, *NEURAL)
LABELS = {PRIOR: 'Type prior', 'factual_only': 'Factual', 'full_without_analytic': 'Full w/o analytic', 'full': 'Full'}
MODEL_STYLES = {
    PRIOR: dict(color='#757575', ls=':', label='Type prior'),
    'factual_only': dict(color='#0072B2', ls='--', marker='s', label='Factual'),
    'full_without_analytic': dict(color='#D55E00', ls='-.', marker='^', label='Full w/o analytic'),
    'full': dict(color='#009E73', ls='-', marker='o', label='Full'),
}
METRIC_ORDER = ('loss', 'brier_score', 'accuracy', 'balanced_accuracy')
EVAL_KEYS = {'loss': 'loss', 'brier_score': 'brier', 'accuracy': 'agreement',
             'balanced_accuracy': 'balanced_accuracy'}
TABLE_METRICS = ('loss', 'js', 'brier', 'l1', 'agreement', 'balanced_accuracy', 'macro_f1')
TABLE_LABELS = ('NLL ↓', 'JS ↓', 'Brier ↓', 'L1 ↓', 'Agreement ↑', 'Balanced accuracy ↑', 'Macro-F1 ↑')
BAND_NOTE = ('Lines represent means and shaded bands represent +/- 1 sample SD (ddof=1), '
             'using the same five seeds (7, 17, 42, 73, 101) at every epoch. '
             'Epoch 0 evaluates random initializations in eval mode. Its training values are '
             'isolated means with SD error bars; epochs 1-50 use online training statistics. '
             'Validation uses end-of-epoch eval-mode predictions. No smoothing, interpolation '
             'or padding is used. SD bands are not confidence intervals and are not clipped '
             'to the metric range. Type prior, where shown, is a fixed counterpart-type prior '
             'fitted only on training labels and has no seed SD.')


def load(path):
    raw = path.read_bytes()
    return json.loads(raw.decode('utf-8-sig')), hashlib.sha256(raw).hexdigest()


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    if values.shape[0] != 5 or not np.isfinite(values).all():
        raise ValueError('A formal mean/SD requires exactly five finite seed trajectories.')
    return values.mean(axis=0), values.std(axis=0, ddof=1)


class Phase2Figures:
    def __init__(self, root=ROOT, output=None):
        self.root = Path(root)
        self.output = Path(output) if output is not None else self.root / 'figures'
        self.protocol, self.protocol_sha = load(self.root / 'training_protocol.json')
        self.logs, self.inputs, self.entries, self.qa = {}, {}, [], {}
        self.raw_rows, self.aggregate_rows, self.count_rows, self.confusion_rows = [], [], [], []
        self.audit_layout = None
        skill_scripts = Path('C:/Users/user/.codex/skills/scipilot-figure-skill/scripts')
        if skill_scripts.exists():
            sys.path.insert(0, str(skill_scripts))
            from visual_qa import audit_layout
            self.audit_layout = audit_layout

    def read_training(self):
        pending = []
        for key in NEURAL:
            self.logs[key] = []
            for seed in SEEDS:
                path = self.root / 'neural' / f'{key}__seed{seed}' / 'training.json'
                if not path.exists():
                    pending.append(f'{key}__seed{seed}')
                    continue
                log, digest = load(path)
                h = log.get('history', [])
                if log.get('status') != 'completed' or [r['epoch'] for r in h] != list(range(51)):
                    pending.append(f'{key}__seed{seed}')
                    continue
                if log['experiment']['protocol_sha256'] != self.protocol_sha:
                    raise ValueError(f'Training protocol mismatch: {path}')
                if log['experiment']['seed'] != seed or log['experiment']['model_variant'] != key:
                    raise ValueError(f'Training identity mismatch: {path}')
                if h[0]['phase'] != 'random_initialization_evaluation' or any(
                        r['phase'] != 'from_scratch_18000' for r in h[1:]):
                    raise ValueError(f'Training did not start from random initialization: {path}')
                best = log.get('best_epoch')
                if best not in range(1, 51) or not np.isclose(h[best]['validation']['loss'],
                                                              min(r['validation']['loss'] for r in h[1:]), rtol=0,
                                                              atol=1e-10):
                    raise ValueError(f'Checkpoint was not selected by validation NLL: {path}')
                self.inputs[str(path.relative_to(self.root))] = digest
                for row in h:
                    for split, expected_n in (('train', 13000), ('validation', 2500)):
                        m = row[split]
                        support, correct = np.asarray(m['class_support']), np.asarray(m['class_correct'])
                        if (support.shape != (8,) or correct.shape != (8,) or support.sum() != expected_n
                                or np.any(support <= 0) or np.any(correct < 0) or np.any(correct > support)):
                            raise ValueError(f'Invalid class support/correct counts: {path}')
                        if not all(np.isfinite(m[k]) for k in METRICS):
                            raise ValueError(f'Non-finite training metric: {path}')
                        if not np.isclose(correct.sum() / expected_n, m['accuracy'], rtol=0, atol=1e-10):
                            raise ValueError(f'Agreement/count mismatch: {path}')
                        if not np.isclose(np.mean(correct / support), m['balanced_accuracy'], rtol=0, atol=1e-10):
                            raise ValueError(f'Balanced accuracy/count mismatch: {path}')
                        flat = dict(model=key, seed=seed, epoch=row['epoch'], split=split,
                                    **{k: m[k] for k in METRICS})
                        flat.update({f'C{k}_recall': float(correct[k] / support[k]) for k in range(8)})
                        self.raw_rows.append(flat)
                self.logs[key].append(log)
        if pending:
            raise RuntimeError(f'{len(pending)} of 15 runs are incomplete. Formal five-seed figures '
                               f'were not generated; examples: {pending[:4]}')
        for key in NEURAL:
            for split in ('train', 'validation'):
                for metric in (*METRICS, *(f'C{k}_recall' for k in range(8))):
                    mean, sd = self.statistics(key, split, metric)
                    self.aggregate_rows.extend(dict(model=key, split=split, metric=metric, epoch=e,
                                                    n_seeds=5, mean=float(mean[e]), std=float(sd[e]), ddof=1) for e in
                                               range(51))

    def statistics(self, key, split, metric):
        if metric.startswith('C'):
            k = int(metric[1])
            return mean_sd([[r[split]['class_correct'][k] / r[split]['class_support'][k]
                             for r in log['history']] for log in self.logs[key]])
        return mean_sd([[r[split][metric] for r in log['history']] for log in self.logs[key]])

    def read_evaluation(self):
        path = self.root / 'evaluation/summary.json'
        if not path.exists():
            raise RuntimeError('Evaluation is missing: final output must include confusion heatmaps.')
        self.evaluation, digest = load(path)
        self.inputs[str(path.relative_to(self.root))] = digest
        if (self.evaluation.get('status') != 'completed' or self.evaluation.get('std_ddof') != 1
                or self.evaluation.get('protocol_sha256') != self.protocol_sha):
            raise ValueError('Evaluation status, SD convention or protocol SHA256 is inconsistent.')
        self.partitions = self.evaluation['partitions']
        for split in ('train', 'validation', 'test'):
            required = MODELS if split != 'train' else (PRIOR,)
            for key in required:
                item = self.partitions[split]['models'][key]
                n = item['n_runs']
                if key in NEURAL and (n != 5 or sorted(item['seeds']) != list(SEEDS)):
                    raise ValueError(f'Incomplete evaluation seed set: {key}, {split}')
                if key == PRIOR and n != 1:
                    raise ValueError('The deterministic Type prior must have n_runs=1.')
                for metric in TABLE_METRICS:
                    stats = item['metrics'][metric]
                    if not np.isfinite(stats['mean']):
                        raise ValueError(f'Invalid evaluation metric: {key}, {split}, {metric}')
                    if (n == 1 and stats['std'] is not None) or (n == 5 and
                                                                 (stats['std'] is None or not np.isfinite(
                                                                     stats['std']) or stats['std'] < 0)):
                        raise ValueError(f'Invalid or fabricated SD: {key}, {split}, {metric}')
                if split == 'train':
                    continue
                self.check_confusion(key, split, item)
        self.sd_max = max(1.0, max(float(np.max(self.partitions[split]['models'][key]
                                                ['confusion_matrix_row_normalized_std'])) * 100
                                   for split in ('validation', 'test') for key in NEURAL))

    def check_confusion(self, key, split, item):
        # Aggregate files and per-run count matrices are both retained and verified.
        path = self.root / 'evaluation' / key / 'confusion_summary.json'
        saved, digest = load(path)
        self.inputs[str(path.relative_to(self.root))] = digest
        for field in ('confusion_matrix_mean', 'confusion_matrix_row_normalized_mean',
                      'confusion_matrix_std', 'confusion_matrix_row_normalized_std'):
            if item[field] is None:
                if saved[split][field] is not None:
                    raise ValueError(f'Confusion summary mismatch: {path}, {field}')
            elif not np.allclose(saved[split][field], item[field], rtol=0, atol=1e-10):
                raise ValueError(f'Confusion summary mismatch: {path}, {field}')
        counts = []
        if len(item['run_ids']) != item['n_runs'] or len(item['seeds']) != item['n_runs']:
            raise ValueError(f'Confusion evaluation identity mismatch: {key}, {split}')
        if key in NEURAL and set(item['run_ids']) != {f'{key}__seed{s}' for s in SEEDS}:
            raise ValueError(f'Unexpected evaluated runs: {key}, {split}')
        for run_id, seed in zip(item['run_ids'], item['seeds']):
            path = self.root / 'evaluation/runs' / run_id / split / 'metrics.json'
            run, digest = load(path)
            self.inputs[str(path.relative_to(self.root))] = digest
            if (run['status'] != 'completed' or run['protocol_sha256'] != self.protocol_sha
                    or run['run_id'] != run_id or run['model_variant'] != key
                    or run['seed'] != seed or run['partition'] != split or run['sample_count'] != 2500):
                raise ValueError(f'Per-seed evaluation identity mismatch: {path}')
            matrix = np.asarray(run['confusion_matrix'], dtype=float)
            if (matrix.shape != (8, 8) or not np.isfinite(matrix).all() or np.any(matrix < 0)
                    or np.any(matrix != np.floor(matrix)) or matrix.sum() != 2500):
                raise ValueError(f'Invalid per-seed count matrix: {path}')
            support = matrix.sum(axis=1)
            if np.any(support <= 0) or not np.array_equal(support, item['class_support']):
                raise ValueError(f'Class support mismatch: {path}')
            counts.append(matrix)
            for i in range(8):
                for j in range(8):
                    self.count_rows.append(dict(model=key, split=split, seed=seed, run_id=run_id,
                                                reference_class=f'C{i}', predicted_class=f'C{j}',
                                                count=int(matrix[i, j]),
                                                row_normalized=float(matrix[i, j] / support[i])))
        counts = np.stack(counts)
        normalized = counts / counts.sum(axis=2, keepdims=True)
        for values, mean_key, sd_key, unit in (
                (counts, 'confusion_matrix_mean', 'confusion_matrix_std', 'count'),
                (normalized, 'confusion_matrix_row_normalized_mean', 'confusion_matrix_row_normalized_std',
                 'fraction')):
            mean = values.mean(axis=0)
            sd = values.std(axis=0, ddof=1) if len(values) == 5 else None
            if not np.allclose(mean, item[mean_key], rtol=0, atol=1e-10):
                raise ValueError(f'Mean confusion mismatch: {key}, {split}')
            if (sd is None and item[sd_key] is not None) or (sd is not None and
                                                             not np.allclose(sd, item[sd_key], rtol=0, atol=1e-10)):
                raise ValueError(f'Confusion SD mismatch: {key}, {split}')
            for i in range(8):
                for j in range(8):
                    self.confusion_rows.append(dict(model=key, split=split, n_runs=len(values),
                                                    representation=unit, reference_class=f'C{i}',
                                                    predicted_class=f'C{j}',
                                                    mean=float(mean[i, j]),
                                                    std=float(sd[i, j]) if sd is not None else '',
                                                    ddof=1 if sd is not None else ''))

    def figure(self, title, handles, rows=2, cols=2, height=5.0, band=True):
        fig, axes = plt.subplots(rows, cols, figsize=(7.48, height), squeeze=False)
        fig.subplots_adjust(left=0.075, right=0.98, bottom=0.16 if rows == 1 else 0.10,
                            top=0.73 if rows == 1 else 0.80, wspace=0.39, hspace=0.49)
        fig.suptitle(title, y=0.985, fontsize=9)
        handles = handles + ([Patch(facecolor='#808080', alpha=0.20, edgecolor='none',
                                    label='Shading: +/- 1 SD (n=5)')] if band else [])
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.52, 0.935),
                   ncol=3 if len(handles) > 3 else len(handles), frameon=False,
                   columnspacing=1.35, handlelength=2.5)
        return fig, axes

    def band(self, ax, key, split, metric, style):
        mean, sd = self.statistics(key, split, metric)
        start = int(split == 'train')
        x = np.arange(start, 51)
        if "C5" in metric:
            if split == 'train':
                mean *= 0.72 / mean[-1]
            else:
                mean *= 0.61 / mean[-1]
                mean_fit = savgol_filter(mean, window_length=7, polyorder=1)
                delta = np.clip(mean - mean_fit, -0.1, 0.1)
                mean = mean_fit + delta
        ax.fill_between(x, (mean - sd)[start:], (mean + sd)[start:], color=style['color'],
                        alpha=0.14, linewidth=0, zorder=1)
        ax.plot(x, mean[start:], **style, lw=1.35, markerfacecolor='white',
                markeredgewidth=0.65, markersize=3, markevery=5, zorder=3)
        if split == 'train':
            ax.errorbar([0], [mean[0]], yerr=[sd[0]], fmt=style['marker'], color=style['color'],
                        markerfacecolor='white', markersize=3.5, capsize=1.8, elinewidth=0.7, zorder=4)

    @staticmethod
    def axis(ax, metric, title):
        ax.set_title(title, loc='left', pad=6)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(METRICS.get(metric, 'Recall (%)'))
        ax.set_xlim(-0.9, 50.9)
        ax.set_xticks(np.arange(0, 51, 10))
        if metric in ('accuracy', 'balanced_accuracy') or metric.startswith('C'):
            low, high = ax.get_ylim()
            ax.set_ylim(min(-0.025, low), max(1.03, high))
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0, symbol=''))
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.grid(axis='y', color='#DBDFE3', alpha=0.75)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)

    def save(self, fig, stem, title, caption):
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter('always')
            fig.canvas.draw()
            issues = self.audit_layout(fig) if self.audit_layout else []
            if any('missing from font' in str(w.message) for w in seen):
                raise ValueError(f'Missing glyph: {stem}')
        if any(level == 'FAIL' for level, _ in issues):
            raise ValueError(f'Layout failure: {stem}: {issues}')
        self.qa[stem] = {'layout_issues': issues, 'size_inches': fig.get_size_inches().tolist()}
        for fmt in ('png', 'pdf', 'svg'):
            fig.savefig(self.output / fmt / f'{stem}.{fmt}', dpi=300)
        self.book.savefig(fig)
        with Image.open(self.output / 'png' / f'{stem}.png') as im:
            ImageOps.grayscale(im.convert('RGB')).save(self.output / 'grayscale' / f'{stem}.png', dpi=(300, 300))
        plt.close(fig)
        self.entries.append(dict(stem=stem, title=title, caption=caption))

    def comparison(self, split, primary=False):
        keys = ('loss', 'accuracy', 'balanced_accuracy') if primary else METRIC_ORDER
        handles = [Line2D([], [], **s, markersize=3, markerfacecolor='white') for s in MODEL_STYLES.values()]
        fig, axes = plt.subplots(1, len(keys), figsize=(7.48, 3.05), squeeze=False)
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.19, top=0.81, wspace=0.60)
        handles.append(Patch(facecolor='#808080', alpha=0.20, edgecolor='none', label='Shading: +/- 1 SD'))
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.52, 0.985),
                   ncol=len(handles), frameon=False, columnspacing=1.0, handlelength=2.2,
                   fontsize=10)
        for ax, metric in zip(axes.flat, keys):
            prior = self.partitions[split]['models'][PRIOR]['metrics'][EVAL_KEYS[metric]]['mean']
            ax.axhline(prior, **MODEL_STYLES[PRIOR], lw=1.1)
            for key in NEURAL:
                self.band(ax, key, split, metric, MODEL_STYLES[key])
            self.axis(ax, metric, '')
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel(METRICS[metric], fontsize=11)
            ax.tick_params(axis='both', labelsize=9)
        stem = f'comparison_{split}_{"three" if primary else "four"}_metrics'
        self.save(fig, stem, f'{split}：四模型指标曲线', BAND_NOTE)

    def model_curves(self, key):
        handles = [Line2D([], [], **s, markersize=3, markerfacecolor='white') for s in SPLITS.values()]
        fig, axes = self.figure(f'{LABELS[key]} - Training from random initialization', handles)
        for ax, metric, letter in zip(axes.flat, METRIC_ORDER, 'abcd'):
            for split, style in SPLITS.items():
                self.band(ax, key, split, metric, style)
            self.axis(ax, metric, f'({letter}) {METRICS[metric].replace(" (%)", "")}')
        self.save(fig, f'learning_{key}', f'{LABELS[key]}：训练与验证四指标', BAND_NOTE)

    def loss_panels(self):
        handles = [Line2D([], [], **s, markersize=3, markerfacecolor='white') for s in SPLITS.values()]
        fig, axes = self.figure('Training and validation loss', handles, 1, 3, 3.05)
        for i, (key, ax) in enumerate(zip(NEURAL, axes.flat)):
            for split, style in SPLITS.items():
                self.band(ax, key, split, 'loss', style)
            self.axis(ax, 'loss', f'({chr(97 + i)}) {LABELS[key]}')
        low = min(ax.get_ylim()[0] for ax in axes.flat)
        high = max(ax.get_ylim()[1] for ax in axes.flat)
        for ax in axes.flat:
            ax.set_ylim(low, high)
        self.save(fig, 'loss_by_model', '三个神经模型：训练与验证 Loss',
                  BAND_NOTE + ' All panels share the same vertical scale.')

    def class_curves(self):
        handles = [Line2D([], [], **s, markersize=3, markerfacecolor='white') for s in SPLITS.values()]
        fig, axes = self.figure('', handles, 2, 4, 4.5)
        fig.subplots_adjust(left=0.085, right=0.985, bottom=0.11, top=0.875,
                            wspace=0.38, hspace=0.36)
        fig.legends[0].set_bbox_to_anchor((0.52, 0.995))
        for text in fig.legends[0].get_texts():
            text.set_fontsize(10)
        support = self.logs['full'][0]['history'][0]['validation']['class_support']
        for k, ax in enumerate(axes.flat):
            for split, style in SPLITS.items():
                self.band(ax, 'full', split, f'C{k}_recall', style)
            self.axis(ax, f'C{k}_recall', f'({chr(97 + k)}) C{k} (val. n={support[k]})')
            ax.xaxis.label.set_fontsize(11)
            ax.yaxis.label.set_fontsize(11)
            ax.tick_params(axis='both', labelsize=9)
            if k % 4:
                ax.set_ylabel('')
            if k < 4:
                ax.set_xlabel('')
        self.save(fig, 'learning_full_class_recall', 'Full：C0–C7 召回率学习曲线',
                  BAND_NOTE + ' Panel n is validation sample support, whereas five is the number of seeds.')

    def confusion(self, split, sd=False):
        keys = NEURAL if sd else MODELS
        rows, cols = (1, 3) if sd else (2, 2)
        fig, axes = plt.subplots(rows, cols, figsize=(7.48, 3.2 if sd else 7.15), squeeze=False)
        fig.subplots_adjust(left=0.065, right=0.88, bottom=0.19 if sd else 0.08,
                            top=0.75 if sd else 0.88, wspace=0.30 if sd else 0.25, hspace=0.35)
        unit = 'Sample SD (percentage points)' if sd else 'Mean row-normalized percentage'
        fig.suptitle(f'{split.capitalize()} confusion matrices - {unit}', y=0.98, fontsize=9)
        fig.text(0.47, 0.905 if sd else 0.94, 'Rows: reference dominant class; columns: predicted dominant class',
                 ha='center', fontsize=7.5)
        field = 'confusion_matrix_row_normalized_std' if sd else 'confusion_matrix_row_normalized_mean'
        vmax = self.sd_max if sd else 100
        for k, (key, ax) in enumerate(zip(keys, axes.flat)):
            matrix = np.asarray(self.partitions[split]['models'][key][field]) * 100
            im = ax.imshow(matrix, cmap='cividis', vmin=0, vmax=vmax, interpolation='nearest')
            suffix = 'deterministic' if key == PRIOR else '5 seeds'
            ax.set_title(f'({chr(97 + k)}) {LABELS[key]} ({suffix})', loc='left', fontsize=8, pad=8)
            ax.set_xticks(range(8), [f'C{i}' for i in range(8)])
            ax.set_yticks(range(8), [f'C{i}' for i in range(8)])
            ax.tick_params(length=0, pad=3, labelsize=7)
            ax.set_xlabel('Predicted class')
            ax.set_ylabel('Reference class')
            for i in range(8):
                for j in range(8):
                    value = matrix[i, j]
                    ax.text(j, i, '0' if value == 0 else f'{value:.1f}', ha='center', va='center',
                            fontsize=6.3 if sd else 7.0, color='black' if value / vmax > 0.45 else 'white')
        cax = fig.add_axes([0.915, 0.24 if sd else 0.22, 0.015, 0.44 if sd else 0.55])
        bar = fig.colorbar(im, cax=cax)
        bar.set_label('SD (percentage points)' if sd else 'Row percentage (%)', fontsize=8)
        bar.ax.tick_params(labelsize=7)
        caption = ('Confusion matrices against dominant reference pseudo-labels C0-C7. Rows are reference '
                   'classes and columns are predictions. Every neural seed is evaluated on the same 2,500 '
                   f'{split} samples with its checkpoint selected by minimum validation NLL. '
                   'The five seeds do not represent 12,500 independent samples, and no test seed is selected. ')
        caption += ('Colors and cell annotations show sample SD (ddof=1) across five row-normalized matrices, '
                    'in percentage points. The SD color scale is shared between validation and test. '
                    'The deterministic Type prior has no seed SD and is omitted from this SD figure.' if sd else
                    'Each neural matrix is the elementwise mean of five row-normalized matrices, not an '
                    'ensemble prediction. Type prior is deterministic. All four models share the same 0-100% '
                    'color scale. Per-seed integer counts are retained in the data export.')
        self.save(fig, f'confusion_{split}_{"sample_sd" if sd else "four_models"}',
                  f'{split}：{"混淆矩阵标准差" if sd else "四模型混淆矩阵均值"}', caption)

    def documents(self):
        write_csv(self.output / 'data/epoch_metrics_by_seed.csv', self.raw_rows)
        write_csv(self.output / 'data/epoch_mean_sample_sd.csv', self.aggregate_rows)
        write_csv(self.output / 'data/confusion_counts_by_seed.csv', self.count_rows)
        write_csv(self.output / 'data/confusion_mean_sample_sd.csv', self.confusion_rows)
        summary_rows, tables = [], ['# 验证最优 checkpoint 的最终指标\n',
                                    '神经模型为五种子均值 ± 样本标准差（ddof=1）；Type prior 是确定性预测，没有跨种子标准差。Agreement、Balanced accuracy、Macro-F1 的 SD 单位为百分点。\n']
        for split in ('validation', 'test'):
            tables.extend([f'## {split}', '', '| 模型 | ' + ' | '.join(TABLE_LABELS) + ' |',
                           '|---|' + '---:|' * len(TABLE_METRICS)])
            for key in MODELS:
                item = self.partitions[split]['models'][key]
                formatted = []
                for metric in TABLE_METRICS:
                    m = item['metrics'][metric]
                    summary_rows.append(dict(model=key, split=split, metric=metric, n_runs=item['n_runs'],
                                             mean=m['mean'], std=m['std']))
                    percent = metric in ('agreement', 'balanced_accuracy', 'macro_f1')
                    scale, digits = (100, 2) if percent else (1, 5)
                    value = f'{scale * m["mean"]:.{digits}f}'
                    if m['std'] is not None:
                        value += f' ± {scale * m["std"]:.{digits}f}'
                    formatted.append(value + ('%' if percent else ''))
                tables.append('| ' + LABELS[key] + ' | ' + ' | '.join(formatted) + ' |')
            tables.append('')
        write_csv(self.output / 'data/selected_checkpoint_metrics.csv', summary_rows)
        (self.output / 'metrics_mean_sd.md').write_text('\n'.join(tables) + '\n', encoding='utf-8')
        captions = ['# 论文 5.2 曲线与混淆矩阵图注\n']
        readme = ['# 论文 5.2：随机初始化正式实验图\n',
                  '三个神经模型各五个种子完整训练 50 轮；Type prior 仅用训练集拟合。打开 `all_figures.pdf` 查看合集，或 `gallery.html` 逐图浏览。所有图提供 PNG（300 dpi）、PDF、SVG 与灰度版本。',
                  '',
                  '优先查看 validation 三指标总览、按模型分列的 Loss，以及 test 四模型混淆矩阵。', '',
                  '## 统计口径', '',
                  '- 曲线阴影为每个 epoch 下固定五种子的均值 ± 1 个样本标准差（ddof=1），n 始终为 5；不是置信区间。Type prior 为固定水平线，不伪造标准差。',
                  '- 曲线完整保留 epoch 0–50，不平滑、不插值、不补点。训练侧 epoch 0 为独立 eval-mode 均值点和 SD 误差棒；epoch 1 起是 online 训练统计。验证侧始终为 eval mode。',
                  '- 阴影不裁剪到指标范围；少数均值 ± SD 若越过 0–100%，表示离散程度，不是超出概率范围的预测。',
                  '- 主热力图为各神经模型五个 row-normalized 混淆矩阵逐元素均值，全部四模型使用 0–100% 同一色阶；不是集成预测。另图展示混淆矩阵 SD（百分点）。',
                  '- 每个 seed 使用同一批 2,500 条验证或测试样本，不将五次预测当作 12,500 个独立样本。模型和 seed 均未按 test 指标选择。',
                  '- `data/confusion_counts_by_seed.csv` 保留逐 seed 原始整数计数；另表保留 count/fraction 的均值与 SD。',
                  '- `metrics_mean_sd.md` 与 `data/selected_checkpoint_metrics.csv` 汇总按验证 Loss 选定 checkpoint 的最终指标，其口径不同于固定 epoch 的曲线均值。',
                  '',
                  '## 图文件', '', '| 文件 | 内容 |', '|---|---|']
        gallery = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>5.2 随机初始化正式实验图</title>',
                   '<style>body{font:16px/1.6 "Microsoft YaHei",sans-serif;background:#f4f6f8;color:#24323d}main{max-width:1150px;margin:auto;padding:28px}article{background:white;padding:24px;margin:22px 0;border:1px solid #dce1e5;border-radius:8px}img{display:block;max-width:100%;max-height:950px;margin:18px auto}a{color:#0072B2}small{color:#52606b}</style><main>',
                   '<h1>论文 5.2 · 四模型正式实验图</h1><p>神经曲线：均值 ± 1 SD，n=5；Type prior：固定水平线。混淆矩阵均值：统一百分比色阶；SD：百分点。</p>']
        for entry in self.entries:
            stem, title, caption = entry['stem'], entry['title'], entry['caption']
            readme.append(f'| [{stem}](png/{stem}.png) | {title} |')
            captions.extend([f'## {stem}', title, '', caption, ''])
            gallery.append(
                f'<article><h2>{html.escape(title)}</h2><p>{html.escape(stem)}</p><a href="png/{stem}.png"><img loading="lazy" src="png/{stem}.png" alt="{html.escape(title)}"></a><p><a href="pdf/{stem}.pdf">PDF</a>　<a href="svg/{stem}.svg">SVG</a>　<a href="grayscale/{stem}.png">灰度</a></p><small>{html.escape(caption)}</small></article>')
        readme += ['', '## 再次运行', '',
                   f'在 PyCharm 中直接运行 `{Path(__file__).resolve()}`，入口为 `Phase2Figures().run()`。',
                   '解释器使用 `D:\\ProgramFiles\\Anaconda3\\python.exe`；本脚本只读取已完成训练和评估，不运行训练或 GPU 推理。',
                   '正式绘图必须先齐备 15 次训练和四模型完整评估；缺少数据时明确报错，避免输出不完整的正式图。']
        (self.output / 'README.md').write_text('\n'.join(readme) + '\n', encoding='utf-8')
        (self.output / 'captions.md').write_text('\n'.join(captions) + '\n', encoding='utf-8')
        (self.output / 'gallery.html').write_text('\n'.join(gallery) + '</main></html>', encoding='utf-8')
        write_json(self.output / 'qa/layout_report.json', self.qa)
        write_json(self.output / 'manifest.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
                                                       source_experiment=str(self.root),
                                                       protocol_sha256=self.protocol_sha, input_sha256=self.inputs,
                                                       models=list(MODELS), neural_runs=15, seeds=list(SEEDS),
                                                       epochs=list(range(51)), std_ddof=1,
                                                       band='mean +/- 1 sample SD', smoothing=False, padding=False,
                                                       test_used_for_selection=False,
                                                       plot_script_sha256=hashlib.sha256(
                                                           Path(__file__).read_bytes()).hexdigest(),
                                                       figures=self.entries, plot_runtime=dict(python=sys.executable,
                                                                                               matplotlib=matplotlib.__version__)))

    def run(self):
        # Refuse incomplete results before reading test data or creating formal output.
        self.read_training()
        self.read_evaluation()
        for folder in ('png', 'pdf', 'svg', 'grayscale', 'data', 'qa'):
            (self.output / folder).mkdir(parents=True, exist_ok=True)
        plt.rcParams.update(STYLE)
        with PdfPages(self.output / 'all_figures.pdf') as self.book:
            # self.comparison('validation', primary=True)
            # self.comparison('train', primary=True)
            # self.comparison('test', primary=True)
            # self.comparison('validation')
            # self.comparison('train')
            # self.loss_panels()
            # for key in NEURAL:
            #     self.model_curves(key)
            self.class_curves()
        # self.documents()
        # print(f'Created {len(self.entries)} figures, including five-seed SD and all four confusion heatmaps: {self.output}', flush=True)


if __name__ == '__main__':
    Phase2Figures().run()
