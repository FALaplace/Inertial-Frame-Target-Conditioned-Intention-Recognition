"""Phase 3: five-seed learning curves, baseline/ablation tables and heatmaps.

Run Phase3Figures().run() in PyCharm with D:/ProgramFiles/Anaconda3/python.exe.
preview_completed_groups() reads only finished training logs, never test results.
The existing Phase 2 scripts and their published figures are not modified.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import html
from pathlib import Path

from project_paths import resolve_project_path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np

from phase2_gpu_figures import (
    Phase2Figures, IDEA, SEEDS, PRIOR, METRICS, METRIC_ORDER, SPLITS, STYLE,
    BAND_NOTE, TABLE_METRICS, TABLE_LABELS, load, mean_sd, write_csv, write_json,
)

ROOT = IDEA / 'experiment_data/phase3_scratch_18000_gpu_v1'
REFERENCES = ('factual_only', 'full_without_analytic', 'full')
RF = 'analytic_random_forest'
LABELS = {
    PRIOR: 'Type prior', RF: 'Analytic RF', 'factual_only': 'Factual TCN',
    'full_without_analytic': 'Full w/o analytic', 'full': 'Full',
    'bilstm_attention': 'BiLSTM + attention', 'inception_time': 'InceptionTime',
    'lightweight_transformer': 'Lightweight Transformer',
    'a0_factual_only_no_mask': 'A0 Factual (no mask)',
    'a1_direct_counterfactual': 'A1 Direct counterfactual', 'a2_event': 'A2 Event comparison',
    'a3_event_analytic': 'A3 + analytic',
    'a4_event_analytic_static_no_mask': 'A4 + static (no mask)',
    'a6_separate_encoders': 'Separate encoders',
    'a6_separate_encoders_parameter_matched': 'Separate, param.-matched',
    'without_maneuver_sequence': 'W/o maneuver sequence',
    'without_orbit_error_summary': 'W/o orbit-error summary',
}
# Each question is kept to at most three models; Full is the recurring reference.
GROUPS = (
    ('baseline_recurrent', 'Baselines: recurrent and convolutional',
     ('factual_only', 'bilstm_attention', 'full')),
    ('baseline_backbones', 'Baselines: alternative sequence backbones',
     ('inception_time', 'lightweight_transformer', 'full')),
    ('baseline_features', 'Baselines: analytic features and fusion',
     (RF, 'full_without_analytic', 'full')),
    ('ablation_fusion', 'Ablation: counterfactual fusion',
     ('a0_factual_only_no_mask', 'a1_direct_counterfactual', 'a2_event')),
    ('ablation_features', 'Ablation: analytic and static features',
     ('a2_event', 'a3_event_analytic', 'a4_event_analytic_static_no_mask')),
    ('ablation_mask', 'Ablation: counterpart-class mask',
     ('a4_event_analytic_static_no_mask', 'full')),
    ('ablation_encoders', 'Ablation: encoder sharing',
     ('a6_separate_encoders', 'a6_separate_encoders_parameter_matched', 'full')),
    ('ablation_inputs', 'Ablation: input channels',
     ('without_maneuver_sequence', 'without_orbit_error_summary', 'full')),
)
LONG_METRICS = {'loss': 'soft_nll', 'brier_score': 'brier_score_sum_over_classes',
                'accuracy': 'dominant_label_agreement', 'balanced_accuracy': 'balanced_accuracy'}
PALETTE = ('#0072B2', '#D55E00', '#CC79A7')
STYLE_INDEX = {key: index for index, keys in enumerate((
    ('factual_only', 'inception_time', 'a0_factual_only_no_mask', 'a3_event_analytic',
     'a6_separate_encoders', 'without_maneuver_sequence'),
    ('bilstm_attention', 'lightweight_transformer', 'full_without_analytic',
     'a1_direct_counterfactual', 'a4_event_analytic_static_no_mask',
     'a6_separate_encoders_parameter_matched', 'without_orbit_error_summary'),
    ('a2_event',),
)) for key in keys}


class Phase3Figures(Phase2Figures):
    def __init__(self, root=ROOT, output=None):
        super().__init__(root, output)
        self.specs = self.protocol['neural_runs']
        grouped = defaultdict(list)
        for spec in self.specs:
            grouped[spec['model_variant']].append(spec)
        self.specs_by_model = dict(grouped)
        expected_new = set(LABELS) - {PRIOR, RF, *REFERENCES}
        if (len(self.specs) != 60 or set(grouped) != expected_new
                or any(sorted(s['seed'] for s in runs) != list(SEEDS) for runs in grouped.values())
                or len({s['run_id'] for s in self.specs}) != 60):
            raise ValueError('Phase 3 requires exactly 12 declared neural models, each with five seeds.')
        if any(s['run_id'] != f"{s['model_variant']}__seed{s['seed']}" for s in self.specs):
            raise ValueError('Unexpected Phase 3 run identity.')
        self.neural = (*grouped, *REFERENCES)
        self.models = (PRIOR, RF, *self.neural)
        self.reference = self.protocol['reference_phase2']
        self.reference_root = resolve_project_path(self.reference['path']).resolve()
        self.reference_hashes = {str(resolve_project_path(name).resolve()): sha
                                 for name, sha in self.reference['artifacts_sha256'].items()}
        _, reference_sha = load(self.reference_root / 'training_protocol.json')
        if reference_sha != self.reference['protocol_sha256']:
            raise ValueError('Referenced Phase 2 protocol has changed.')
        self.rf_logs, self.rf_rows, self.rf_aggregate = [], [], []
        self.preview = False
        # Stable model styles across groups, with Full always green and solid.
        other = [key for key in self.neural if key != 'full']
        self.styles = {key: dict(color=PALETTE[STYLE_INDEX[key]],
                       ls=('--', '-.', ':')[STYLE_INDEX[key]], marker=('s', '^', 'D')[STYLE_INDEX[key]],
                       label=LABELS[key]) for key in other}
        self.styles['full'] = dict(color='#009E73', ls='-', marker='o', label='Full')

    def tracked(self, path, reference=False):
        value, digest = load(path)
        path_key = str(path.resolve())
        self.inputs[path_key] = digest
        if reference:
            if self.reference_hashes.get(path_key) != digest:
                raise ValueError(f'Reference artifact is not frozen or changed: {path}')
        return value

    def read_training(self, allow_incomplete=False):
        pending = []
        for key in self.neural:
            is_reference = key in REFERENCES
            source = self.reference_root if is_reference else self.root
            logs = []
            for seed in SEEDS:
                path = source / 'neural' / f'{key}__seed{seed}' / 'training.json'
                if not path.exists():
                    pending.append(f'{key}__seed{seed}')
                    continue
                log = self.tracked(path, is_reference)
                history = log.get('history', [])
                if log.get('status') != 'completed' or [r['epoch'] for r in history] != list(range(51)):
                    pending.append(f'{key}__seed{seed}')
                    continue
                identity = log['experiment']
                expected_sha = self.reference['protocol_sha256'] if is_reference else self.protocol_sha
                if (identity['protocol_sha256'] != expected_sha or identity['seed'] != seed
                        or identity['model_variant'] != key):
                    raise ValueError(f'Training identity/protocol mismatch: {path}')
                if history[0]['phase'] != 'random_initialization_evaluation' or any(
                        row['phase'] != 'from_scratch_18000' for row in history[1:]):
                    raise ValueError(f'Not a random-initialization training history: {path}')
                best = log.get('best_epoch')
                if best not in range(1, 51) or not np.isclose(history[best]['validation']['loss'],
                        min(r['validation']['loss'] for r in history[1:]), rtol=0, atol=1e-10):
                    raise ValueError(f'Checkpoint not selected by validation NLL: {path}')
                for row in history:
                    for split, n in (('train', 13000), ('validation', 2500)):
                        m = row[split]
                        support, correct = np.asarray(m['class_support']), np.asarray(m['class_correct'])
                        if (support.shape != (8,) or correct.shape != (8,)
                                or support.sum() != n or np.any(support <= 0)
                                or np.any(correct < 0) or np.any(correct > support)
                                or not all(np.isfinite(m[k]) for k in METRICS)
                                or not np.isclose(correct.sum()/n, m['accuracy'], atol=1e-10, rtol=0)
                                or not np.isclose(np.mean(correct/support), m['balanced_accuracy'], atol=1e-10, rtol=0)):
                            raise ValueError(f'Invalid training metrics/counts: {path}, epoch {row["epoch"]}')
                logs.append(log)
            if len(logs) == 5:
                self.logs[key] = logs
        if pending and not allow_incomplete:
            raise RuntimeError(f'{len(pending)} runs remain incomplete; no formal figures were created.')
        for key, logs in self.logs.items():
            for log in logs:
                for row in log['history']:
                    for split in SPLITS:
                        m = row[split]
                        self.raw_rows.append(dict(model=key, seed=log['experiment']['seed'],
                            epoch=row['epoch'], split=split, source='phase2_reference' if key in REFERENCES else 'phase3',
                            **{metric: m[metric] for metric in METRICS}))
            for split in SPLITS:
                for metric in METRICS:
                    mean, sd = self.statistics(key, split, metric)
                    self.aggregate_rows.extend(dict(model=key, split=split, metric=metric, epoch=e,
                        n_seeds=5, mean=float(mean[e]), std=float(sd[e]), ddof=1) for e in range(51))
        self.pending = pending

    def read_evaluation(self):
        self.evaluation = self.tracked(self.root / 'evaluation/summary.json')
        if (self.evaluation.get('status') != 'completed' or self.evaluation.get('std_ddof') != 1
                or self.evaluation.get('protocol_sha256') != self.protocol_sha):
            raise ValueError('Evaluation must be completed under the Phase 3 protocol with sample SD.')
        self.partitions = self.evaluation['partitions']
        for split in ('validation', 'test'):
            if set(self.partitions[split]['models']) != set(self.models):
                raise ValueError(f'Expected all 17 models in {split} evaluation.')
            for key in self.models:
                item = self.partitions[split]['models'][key]
                n = 1 if key == PRIOR else 5
                if item['n_runs'] != n or (n == 5 and sorted(item['seeds']) != list(SEEDS)):
                    raise ValueError(f'Incomplete evaluation seeds: {key}, {split}')
                for metric in TABLE_METRICS:
                    stats = item['metrics'][metric]
                    if (not np.isfinite(stats['mean']) or (n == 1 and stats['std'] is not None)
                            or (n == 5 and (stats['std'] is None or not np.isfinite(stats['std']) or stats['std'] < 0))):
                        raise ValueError(f'Invalid evaluation mean/SD: {key}, {split}, {metric}')
                self.check_confusion(key, split, item)

    def check_confusion(self, key, split, item):
        saved = self.tracked(self.root / 'evaluation' / key / 'confusion_summary.json')[split]
        fields = ('confusion_matrix_mean', 'confusion_matrix_std',
                  'confusion_matrix_row_normalized_mean', 'confusion_matrix_row_normalized_std')
        for field in fields:
            if ((item[field] is None and saved[field] is not None) or (item[field] is not None
                    and not np.allclose(item[field], saved[field], rtol=0, atol=1e-10))):
                raise ValueError(f'Confusion summary mismatch: {key}/{split}/{field}')
        if len(item['run_ids']) != item['n_runs'] or len(set(item['run_ids'])) != item['n_runs']:
            raise ValueError(f'Duplicate/missing confusion runs: {key}, {split}')
        counts = []
        is_reference = item.get('source') == 'reference_phase2'
        if is_reference and item.get('source_protocol_sha256') != self.reference['protocol_sha256']:
            raise ValueError(f'Unexpected reference evaluation protocol: {key}, {split}')
        source_root = self.reference_root if is_reference else self.root
        expected_sha = self.reference['protocol_sha256'] if is_reference else self.protocol_sha
        for run_id, seed in zip(item['run_ids'], item['seeds']):
            run = self.tracked(source_root / 'evaluation/runs' / run_id / split / 'metrics.json', is_reference)
            if (run['status'] != 'completed' or run['protocol_sha256'] != expected_sha
                    or run['run_id'] != run_id or run['model_variant'] != key
                    or run['seed'] != seed or run['partition'] != split or run['sample_count'] != 2500):
                raise ValueError(f'Invalid evaluation identity: {run_id}, {split}')
            matrix = np.asarray(run['confusion_matrix'], dtype=float)
            if (matrix.shape != (8, 8) or not np.isfinite(matrix).all() or np.any(matrix < 0)
                    or np.any(matrix != np.floor(matrix)) or matrix.sum() != 2500
                    or not np.array_equal(matrix.sum(axis=1), item['class_support'])
                    or np.any(matrix.sum(axis=1) <= 0)):
                raise ValueError(f'Invalid raw confusion counts: {run_id}, {split}')
            counts.append(matrix)
            self.count_rows.extend(dict(model=key, split=split, seed=seed, run_id=run_id,
                reference_class=f'C{i}', predicted_class=f'C{j}', count=int(matrix[i,j]))
                for i in range(8) for j in range(8))
        counts = np.stack(counts)
        for values, representation, mean_field, sd_field in (
                (counts, 'count', fields[0], fields[1]),
                (counts/counts.sum(axis=2, keepdims=True), 'fraction', fields[2], fields[3])):
            mean = values.mean(axis=0)
            sd = values.std(axis=0, ddof=1) if len(values) == 5 else None
            if not np.allclose(mean, item[mean_field], rtol=0, atol=1e-10) or (
                    sd is None and item[sd_field] is not None) or (sd is not None
                    and not np.allclose(sd, item[sd_field], rtol=0, atol=1e-10)):
                raise ValueError(f'Confusion aggregation mismatch: {key}, {split}')
            self.confusion_rows.extend(dict(model=key, split=split, n_runs=len(values),
                representation=representation, reference_class=f'C{i}', predicted_class=f'C{j}',
                mean=float(mean[i,j]), std=float(sd[i,j]) if sd is not None else '',
                ddof=1 if sd is not None else '') for i in range(8) for j in range(8))

    def read_rf(self):
        classical = self.root / 'classical'
        manifest = self.tracked(classical / 'manifest.json')
        if manifest.get('status') != 'complete' or sorted(manifest['rf_seeds']) != list(SEEDS):
            raise ValueError('RF five-seed training is incomplete.')
        for seed in SEEDS:
            path = classical / 'B1_analytic_random_forest' / f'seed_{seed}' / 'history.json'
            log = self.tracked(path)
            if (hashlib.sha256(path.read_bytes()).hexdigest() != manifest['artifact_sha256'][path.relative_to(classical).as_posix()]
                    or log['seed'] != seed or log['step_axis'] != 'tree_count'
                    or [r['tree_count'] for r in log['history']] != [50, 100, 150, 200, 250, 300]):
                raise ValueError(f'Invalid RF tree history: {path}')
            self.rf_logs.append(log)
            for row in log['history']:
                for split in SPLITS:
                    self.rf_rows.append(dict(model=RF, seed=seed, tree_count=row['tree_count'], split=split,
                        **{metric: row[split][long_key] for metric, long_key in LONG_METRICS.items()}))
        for split in SPLITS:
            for metric, long_key in LONG_METRICS.items():
                mean, sd = mean_sd([[r[split][long_key] for r in log['history']] for log in self.rf_logs])
                self.rf_aggregate.extend(dict(model=RF, split=split, metric=metric, tree_count=tree,
                    n_seeds=5, mean=float(m), std=float(s), ddof=1)
                    for tree, m, s in zip((50, 100, 150, 200, 250, 300), mean, sd))

    def comparison_group(self, group, split):
        slug, title, models = group
        keys = [key for key in models if key != RF]
        handles = [Line2D([], [], **self.styles[key], markersize=3, markerfacecolor='white') for key in keys]
        heading = ('Preview - ' if self.preview else '') + title + f' - {split.capitalize()}'
        fig, axes = self.figure(heading, handles)
        for ax, metric, letter in zip(axes.flat, METRIC_ORDER, 'abcd'):
            for key in keys:
                self.band(ax, key, split, metric, self.styles[key])
            self.axis(ax, metric, f'({letter}) {METRICS[metric].replace(" (%)", "")}')
        self.save(fig, f'{slug}_{split}_four_metrics', heading, BAND_NOTE)

    def group_loss(self, group):
        slug, title, models = group
        keys = [key for key in models if key != RF]
        handles = [Line2D([], [], **s, markersize=3, markerfacecolor='white') for s in SPLITS.values()]
        heading = ('Preview - ' if self.preview else '') + title + ' - Loss'
        fig, axes = self.figure(heading, handles, 1, len(keys), 3.15)
        for i, (ax, key) in enumerate(zip(axes.flat, keys)):
            for split, style in SPLITS.items():
                self.band(ax, key, split, 'loss', style)
            self.axis(ax, 'loss', f'({chr(97+i)}) {LABELS[key]}')
            ax.set_title(f'({chr(97+i)}) {LABELS[key]}', loc='left', fontsize=7.3, pad=6)
        low = min(ax.get_ylim()[0] for ax in axes.flat)
        high = max(ax.get_ylim()[1] for ax in axes.flat)
        for ax in axes.flat:
            ax.set_ylim(low, high)
        self.save(fig, f'{slug}_training_validation_loss', heading, BAND_NOTE + ' Panels share the same NLL scale.')

    def confusion_group(self, group, split):
        slug, title, keys = group
        fig, axes = plt.subplots(1, len(keys), figsize=(7.48, 3.2), squeeze=False)
        fig.subplots_adjust(left=0.065, right=0.88, bottom=0.19, top=0.75, wspace=0.30)
        heading = f'{title} - {split.capitalize()}'
        fig.suptitle(heading, y=0.985, fontsize=9)
        fig.text(0.47, 0.905, 'Rows: reference class; columns: predicted class', ha='center', fontsize=7)
        for i, (key, ax) in enumerate(zip(keys, axes.flat)):
            matrix = np.asarray(self.partitions[split]['models'][key]['confusion_matrix_row_normalized_mean']) * 100
            # pcolormesh preserves a fully vector matrix in PDF/SVG.
            mesh = ax.pcolormesh(np.arange(9)-0.5, np.arange(9)-0.5, matrix,
                                cmap='cividis', vmin=0, vmax=100, shading='flat')
            ax.set_ylim(7.5, -0.5)
            ax.set_aspect('equal')
            ax.set_title(f'({chr(97+i)}) {LABELS[key]}', loc='left', fontsize=7.2, pad=8)
            ax.set_xticks(range(8), [f'C{k}' for k in range(8)])
            ax.set_yticks(range(8), [f'C{k}' for k in range(8)])
            ax.tick_params(length=0, pad=3, labelsize=7)
            ax.set_xlabel('Predicted class')
            ax.set_ylabel('Reference class')
            for r in range(8):
                for c in range(8):
                    value = matrix[r,c]
                    ax.text(c, r, '0' if value == 0 else f'{value:.1f}', ha='center', va='center',
                            fontsize=6.3, color='black' if value > 45 else 'white')
        bar = fig.colorbar(mesh, cax=fig.add_axes([0.915, 0.24, 0.015, 0.44]))
        bar.set_label('Row percentage (%)', fontsize=8)
        bar.ax.tick_params(labelsize=7)
        caption = (f'同一批 2,500 条 {split} 样本上的 C0-C7 混淆矩阵。行是真实伪标签主类，列为预测主类。'
                   '每个模型为五个种子行归一化矩阵的逐元素均值，所有面板统一 0-100% 色阶。'
                   '五个种子重复评价同一批样本，不是 12,500 条独立样本，也不是模型集成。'
                   '神经模型按最小验证 NLL 选择 checkpoint，RF 固定 300 棵树；不按测试结果挑种子。'
                   'Type prior 不出现在热力图中；单种子整数计数和逐元素样本 SD 保留在 CSV。')
        self.save(fig, f'{slug}_confusion_{split}', heading, caption)

    def rf_curves(self):
        styles = {split: {**style, 'label': 'Training (in-sample)' if split == 'train' else 'Validation'}
                  for split, style in SPLITS.items()}
        handles = [Line2D([], [], **s, markersize=3, markerfacecolor='white') for s in styles.values()]
        fig, axes = self.figure('Analytic Random Forest - Tree-count learning curves', handles)
        x = np.array([50, 100, 150, 200, 250, 300])
        for ax, metric, letter in zip(axes.flat, METRIC_ORDER, 'abcd'):
            for split, style in styles.items():
                mean, sd = mean_sd([[r[split][LONG_METRICS[metric]] for r in log['history']] for log in self.rf_logs])
                ax.fill_between(x, mean-sd, mean+sd, color=style['color'], alpha=0.14, linewidth=0)
                ax.plot(x, mean, **style, lw=1.35, markersize=3, markerfacecolor='white')
            self.axis(ax, metric, f'({letter}) {METRICS[metric].replace(" (%)", "")}')
            ax.set_xlabel('Number of trees')
            ax.set_xlim(45, 305)
            ax.set_xticks(x)
        self.save(fig, 'baseline_rf_tree_count_curves', '随机森林：树数量与训练/验证四指标',
                  '横轴是实际树数量 50/100/150/200/250/300，不是 epoch。曲线为五种子均值，阴影为±1 样本 SD（ddof=1，n=5）。'
                  '训练指标是在训练样本上的拟合内指标；模型固定使用 300 棵树，不按验证或测试曲线选择树数量。')

    def documents(self):
        for name, rows in (('epoch_metrics_by_seed', self.raw_rows), ('epoch_mean_sample_sd', self.aggregate_rows),
                           ('confusion_counts_by_seed', self.count_rows), ('confusion_mean_sample_sd', self.confusion_rows),
                           ('rf_tree_metrics_by_seed', self.rf_rows), ('rf_tree_mean_sample_sd', self.rf_aggregate)):
            if rows:
                write_csv(self.output / 'data' / f'{name}.csv', rows)
        if not self.preview:
            table_rows, table = [], ['# Phase 3 最终指标（均值 ± 样本标准差）', '',
                '神经网络与 RF 均为五种子；Type prior 为确定性单次结果，SD 记为不适用。', '']
            for split in ('validation', 'test'):
                table += [f'## {split}', '', '| Model | ' + ' | '.join(TABLE_LABELS) + ' |',
                          '|---|' + '---:|' * len(TABLE_METRICS)]
                for key in self.models:
                    item = self.partitions[split]['models'][key]
                    cells = []
                    for metric in TABLE_METRICS:
                        stats = item['metrics'][metric]
                        cells.append(f"{stats['mean']:.4f}" + (f" ± {stats['std']:.4f}" if stats['std'] is not None else ' (SD N/A)'))
                        table_rows.append(dict(model=key, split=split, metric=metric, n_runs=item['n_runs'],
                            mean=stats['mean'], std=stats['std'], ddof=1 if item['n_runs'] == 5 else ''))
                    table.append('| ' + LABELS[key] + ' | ' + ' | '.join(cells) + ' |')
                table.append('')
            write_csv(self.output / 'data/final_metrics_mean_sd.csv', table_rows)
            (self.output / 'metrics_mean_sd.md').write_text('\n'.join(table) + '\n', encoding='utf-8')
        readme = ['# Phase 3 绘图' + ('（Preview，仅已完成组）' if self.preview else ''), '',
            '每组最多三个模型。神经模型每个种子均由随机初始化训练 50 轮；三个 Phase 2 对照直接引用冻结结果。',
            '曲线与阴影分别为五种子均值和 ±1 样本 SD（ddof=1），每个 epoch 均为 n=5；不是置信区间。',
            'epoch 0 训练集是独立初始化评估点；epoch 1-50 为在线训练统计。验证集均为 eval 模式。',
            '不平滑、不插值、不补点。均值±SD 不裁剪，因此其范围可能超出 0-100%，这不代表概率超界。',
            'RF 横轴是树数量，单独出图，不混用 epoch。Type prior 仅出现在最终表格中，无虚构 SD。',
            '热力图为五种子行归一化矩阵均值，统一 0-100%；每个种子使用同一批 2,500 条样本。',
            '最终指标对应按验证 NLL 选定的 checkpoint，因此不一定等于第 50 轮曲线端点。',
            'A1→A2 是直接融合与事件条件比较的替换对照，不表述为单纯增加一个模块。',
            '自动布局检查在 qa/layout_report.json；实际成图目视复核需另行记录，脚本不会自行宣称目视通过。', '',
            '## 图目录', '', '| 图文件 | 内容 |', '|---|---|']
        captions, gallery = [], ['<!doctype html><meta charset="utf-8"><title>Phase 3 figures</title>',
            '<style>body{max-width:1100px;margin:28px auto;font:16px sans-serif;background:#f4f5f7}article{background:white;padding:20px;margin:24px 0}img{width:100%}</style><main>']
        for entry in self.entries:
            stem, title, caption = entry['stem'], entry['title'], entry['caption']
            readme.append(f'| [{stem}](png/{stem}.png) | {title} |')
            captions.extend([f'## {stem}', title, '', caption, ''])
            gallery.append(f'<article><h2>{html.escape(title)}</h2><img loading="lazy" src="png/{stem}.png" alt="{html.escape(title)}"><p><a href="pdf/{stem}.pdf">PDF</a> · <a href="svg/{stem}.svg">SVG</a> · <a href="grayscale/{stem}.png">灰度</a></p><small>{html.escape(caption)}</small></article>')
        for name, lines in (('README.md', readme), ('captions.md', captions), ('gallery.html', gallery + ['</main>'])):
            (self.output / name).write_text('\n'.join(lines) + '\n', encoding='utf-8')
        write_json(self.output / 'qa/layout_report.json', self.qa)
        write_json(self.output / 'manifest.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
            status='preview' if self.preview else 'completed', source_experiment=str(self.root),
            protocol_sha256=self.protocol_sha, input_sha256=self.inputs, seeds=list(SEEDS), std_ddof=1,
            epochs=list(range(51)), new_neural_runs=60, reference_neural_runs=15,
            complete_models=list(self.logs), pending_runs=self.pending, figures=self.entries,
            smoothing=False, padding=False, test_used_for_selection=False,
            plot_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            reused_script_sha256={name: hashlib.sha256((IDEA/'scripts'/name).read_bytes()).hexdigest()
                                  for name in ('phase2_gpu_figures.py', 'phase2_scratch_curve_preview.py')},
            plot_runtime=dict(python=sys.executable, matplotlib=matplotlib.__version__),
            visual_review_status='pending_independent_inspection'))

    def render(self, groups):
        for folder in ('png', 'pdf', 'svg', 'grayscale', 'data', 'qa'):
            (self.output / folder).mkdir(parents=True, exist_ok=True)
        plt.rcParams.update(STYLE)
        with PdfPages(self.output / 'all_figures.pdf') as self.book:
            for group in groups:
                for split in SPLITS:
                    self.comparison_group(group, split)
                self.group_loss(group)
                if not self.preview:
                    for split in ('validation', 'test'):
                        self.confusion_group(group, split)
            if not self.preview:
                self.rf_curves()
        self.documents()
        return self.output

    def run(self):
        # No output directory or test read until all 75 real neural histories pass.
        self.read_training()
        self.read_rf()
        self.read_evaluation()
        result = self.render(GROUPS)
        print(f'Created {len(self.entries)} real Phase 3 figures: {result}', flush=True)
        return result

    def preview_completed_groups(self):
        """Optional monitoring: only complete five-seed groups, train/validation only."""
        self.preview = True
        self.output = self.root / 'figures_preview'
        self.read_training(allow_incomplete=True)
        groups = [g for g in GROUPS if all(k in self.logs for k in g[2] if k != RF)]
        if not groups:
            raise RuntimeError('No complete five-seed comparison group is available for preview.')
        return self.render(groups)


def self_check():
    """Small CPU-only check: sample SD must not silently become population SD."""
    for _, _, keys in GROUPS:
        indices = [STYLE_INDEX.get(key, -1) for key in keys if key != RF]
        assert len(indices) == len(set(indices)), 'Model styles must differ within a comparison.'
    mean, sd = mean_sd([[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]])
    np.testing.assert_allclose(mean, [3, 6])
    np.testing.assert_allclose(sd, np.sqrt([2.5, 10]))
    try:
        mean_sd([[1, 2]])
    except ValueError:
        pass
    else:
        raise AssertionError('A single seed must not generate a formal SD band.')


if __name__ == '__main__':
    self_check()
    Phase3Figures().run()
