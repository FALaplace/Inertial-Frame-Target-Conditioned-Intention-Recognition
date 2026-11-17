"""Plot the real Full / seed 7 run on the frozen 18,000-sample dataset.

Run directly in PyCharm with D:/ProgramFiles/Anaconda3/python.exe.
The input may still be running; incomplete figures are labelled Preview.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
from PIL import Image, ImageOps


IDEA = Path(__file__).resolve().parents[1]
EXPERIMENT = IDEA / 'experiment_data/scratch_18000_full_seed7_v1'
SOURCE = EXPERIMENT / 'neural/full__seed7/training.json'
OUTPUT = EXPERIMENT / 'figures/curve_preview'
METRICS = {'loss': 'Soft-label NLL', 'accuracy': 'Agreement (%)',
           'balanced_accuracy': 'Balanced accuracy (%)', 'brier_score': 'Brier score'}
SPLITS = {'train': dict(color='#0072B2', ls='--', marker='s', label='Training (online)'),
          'validation': dict(color='#D55E00', ls='-', marker='o', label='Validation')}
STYLE = {'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
         'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9,
         'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 8,
         'axes.linewidth': 0.65, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
         'xtick.direction': 'out', 'ytick.direction': 'out', 'grid.linewidth': 0.45,
         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
         'savefig.facecolor': 'white', 'figure.facecolor': 'white', 'axes.facecolor': 'white'}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                    encoding='utf-8')


class CurvePreview:
    def __init__(self):
        # Read one immutable snapshot even while the trainer updates the file.
        raw = SOURCE.read_bytes()
        self.input_sha = hashlib.sha256(raw).hexdigest()
        self.log = json.loads(raw.decode('utf-8-sig'))
        self.history = self.log['history']
        self.rows, self.entries, self.qa = [], [], {}
        self.validate()
        self.epoch = self.history[-1]['epoch']
        self.complete = self.log['status'] == 'completed' and self.epoch == 50
        self.suffix = 'complete_50epochs' if self.complete else f'preview_epoch{self.epoch:02d}'
        self.heading = ('Full / seed 7 - Complete: 50 epochs' if self.complete else
                        f'Preview - Full / seed 7 - Epoch {self.epoch} of 50')
        self.audit_layout = None
        skill_scripts = Path('C:/Users/user/.codex/skills/scipilot-figure-skill/scripts')
        if skill_scripts.exists():
            sys.path.insert(0, str(skill_scripts))
            from visual_qa import audit_layout
            self.audit_layout = audit_layout

    def validate(self):
        if self.log['status'] not in ('running', 'completed'):
            raise ValueError('Only a running or completed training log can be plotted.')
        if not self.history or [r['epoch'] for r in self.history] != list(range(len(self.history))):
            raise ValueError('Expected consecutive real epochs starting at epoch 0.')
        if self.history[0]['phase'] != 'random_initialization_evaluation':
            raise ValueError('Epoch 0 must evaluate randomly initialized weights.')
        if any(r['phase'] != 'from_scratch_18000' for r in self.history[1:]):
            raise ValueError('A non-scratch training phase was found; refusing to mix runs.')
        if self.history[-1]['epoch'] > 50:
            raise ValueError('Expected at most 50 training epochs.')
        for row in self.history:
            for split, expected_n in (('train', 13000), ('validation', 2500)):
                m = row[split]
                support, correct = np.array(m['class_support']), np.array(m['class_correct'])
                if support.shape != (8,) or correct.shape != (8,) or support.sum() != expected_n:
                    raise ValueError(f'Unexpected class counts: epoch {row["epoch"]}, {split}.')
                if np.any(support <= 0) or np.any(correct < 0) or np.any(correct > support):
                    raise ValueError('Invalid per-class counts.')
                if not all(np.isfinite(m[key]) for key in METRICS):
                    raise ValueError('A metric is missing or non-finite.')
                if not np.isclose(correct.sum() / support.sum(), m['accuracy'], atol=1e-10):
                    raise ValueError('Agreement does not match class counts.')
                if not np.isclose(np.mean(correct / support), m['balanced_accuracy'], atol=1e-10):
                    raise ValueError('Balanced accuracy does not match class counts.')
                flat = dict(seed=7, epoch=row['epoch'], phase=row['phase'], split=split,
                            epoch_seconds=row.get('epoch_seconds', ''),
                            learning_rate=row.get('learning_rate', ''),
                            **{key: m[key] for key in METRICS})
                for k in range(8):
                    flat[f'C{k}_support'] = int(support[k])
                    flat[f'C{k}_recall'] = float(correct[k] / support[k])
                self.rows.append(flat)

    def figure(self, rows, cols, height):
        fig, axes = plt.subplots(rows, cols, figsize=(7.48, height), squeeze=False)
        fig.subplots_adjust(left=0.078, right=0.98, bottom=0.17 if rows == 1 else 0.11,
                            top=0.72 if rows == 1 else 0.83, wspace=0.40, hspace=0.54)
        fig.suptitle(self.heading, y=0.985, fontsize=9)
        handles = [Line2D([], [], **s, markersize=3.2, markerfacecolor='white')
                   for s in SPLITS.values()]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.53, 0.91 if rows == 1 else 0.945),
                   ncol=2, frameon=False)
        return fig, axes

    def curve(self, ax, split, metric=None, class_index=None):
        x = np.array([row['epoch'] for row in self.history])
        y = np.array([row[split][metric] if metric else
                      row[split]['class_correct'][class_index] / row[split]['class_support'][class_index]
                      for row in self.history])
        style = SPLITS[split]
        kw = dict(**style, lw=1.35, markersize=3, markerfacecolor='white',
                  markeredgewidth=0.7, markevery=max(1, self.epoch // 10),
                  zorder=6 if split == 'validation' else 4)
        start = 1 if split == 'train' else 0
        ax.plot(x[start:], y[start:], **kw)
        if split == 'train':
            # Epoch 0 is eval mode; epochs 1..50 are online training with dropout.
            ax.plot(0, y[0], color=style['color'], marker=style['marker'], ls='none',
                    markerfacecolor='white', markersize=4.3, zorder=5)

    def axis(self, ax, metric, title):
        ax.set_title(title, loc='left', pad=7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(METRICS.get(metric, 'Recall (%)'))
        ax.set_xlim(-0.45, max(self.epoch, 1) + 0.45)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        if metric in ('accuracy', 'balanced_accuracy', 'recall'):
            ax.set_ylim(-0.025, 1.03)
            ax.set_yticks(np.linspace(0, 1, 6))
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0, symbol=''))
        else:
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.margins(y=0.12)
        ax.grid(axis='y', color='#DBDFE3', alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)

    def save(self, fig, basename, title, caption):
        stem = f'{basename}_{self.suffix}'
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter('always')
            fig.canvas.draw()
            issues = self.audit_layout(fig) if self.audit_layout else []
            if any('missing from font' in str(w.message) for w in seen):
                raise ValueError(f'Missing glyph in {stem}.')
        if any(level == 'FAIL' for level, _ in issues):
            raise ValueError(f'Layout failure: {stem}: {issues}')
        self.qa[stem] = {'layout_issues': issues, 'size_inches': fig.get_size_inches().tolist()}
        for fmt in ('png', 'pdf', 'svg'):
            fig.savefig(OUTPUT / fmt / f'{stem}.{fmt}', dpi=300)
        self.book.savefig(fig)
        with Image.open(OUTPUT / 'png' / f'{stem}.png') as im:
            ImageOps.grayscale(im.convert('RGB')).save(OUTPUT / 'grayscale' / f'{stem}.png', dpi=(300, 300))
        plt.close(fig)
        self.entries.append({'stem': stem, 'title': title, 'caption': caption})

    def metric_panels(self, keys, rows, cols, height, basename, title):
        fig, axes = self.figure(rows, cols, height)
        for k, (ax, metric) in enumerate(zip(axes.flat, keys)):
            for split in SPLITS:
                self.curve(ax, split, metric)
            self.axis(ax, metric, f'({chr(97+k)}) {METRICS[metric].replace(" (%)", "")}')
        self.save(fig, basename, title,
                  'Training and validation trajectories of Full, initialized randomly and trained on the '
                  'frozen 18,000-sample dataset (seed 7). Dashed blue lines show online training metrics; '
                  'solid orange lines show end-of-epoch validation metrics. Epoch 0 evaluates the random '
                  'initialization in eval mode; its training point is isolated. No smoothing or padding is used.')

    def class_recall(self):
        fig, axes = self.figure(2, 4, 4.85)
        fig.subplots_adjust(left=0.065, wspace=0.30)
        support = self.history[0]['validation']['class_support']
        for k, ax in enumerate(axes.flat):
            for split in SPLITS:
                self.curve(ax, split, class_index=k)
            self.axis(ax, 'recall', f'({chr(97+k)}) C{k} (val. n={support[k]})')
            if k % 4:
                ax.set_ylabel('')
            if k < 4:
                ax.set_xlabel('')
        self.save(fig, '04_class_recall', 'C0–C7 训练与验证召回率',
                  'Per-class dominant-label recall of Full from random initialization (seed 7). '
                  'Panels show C0-C7; n denotes validation support. Dashed blue and solid orange lines '
                  'represent online training and validation, respectively. These are recall curves, '
                  'not per-class F1 or test-set confusion matrices.')

    def documents(self):
        with (OUTPUT / 'data' / f'epoch_metrics_{self.suffix}.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        status = '完整 50 轮已结束' if self.complete else f'训练中预览：已有 epoch 0–{self.epoch}，目标为 50 轮'
        captions = ['# 随机初始化训练曲线图注\n', status, '']
        readme = ['# Full / seed 7 随机初始化训练曲线\n', status, '',
                  f'合集：`all_curves_{self.suffix}.pdf`。每张图提供 PNG（300 dpi）、PDF、SVG 及灰度版。', '',
                  '| 编号 | 内容 | PNG 文件 |', '|---|---|---|']
        for i, entry in enumerate(self.entries, 1):
            readme.append(f'| {i:02d} | {entry["title"]} | [打开](png/{entry["stem"]}.png) |')
            captions.extend([f'## {i:02d} {entry["title"]}', entry['caption'], ''])
        readme.extend(['', '## 数据口径', '',
                      '- 输入只来自本次随机初始化训练；没有拼接旧模型的续训记录。',
                      '- 训练集 13,000 条，验证集 2,500 条；没有使用测试集绘制学习曲线。',
                      '- epoch 0 为随机权重的 eval-mode 评估；训练侧该点单独显示，后续为在线训练统计。',
                      '- 每个点均对应真实记录；不平滑、不插值、不将未训练轮次补点。',
                      '- 本次仅一个种子，因此不绘制种子标准差。图的终点不一定是验证 Loss 最佳 checkpoint。',
                      '- 完整图须同时满足日志 status=completed 且已有连续 epoch 0–50；否则一律标记 Preview。',
                      '- `manifest.json` 保存绘图时日志的 SHA256，供核对具体数据快照。', '',
                      '## 在 PyCharm 中重新生成', '',
                      f'直接运行 `{Path(__file__).resolve()}`，入口为 `CurvePreview().run()`。',
                      '绘图解释器使用 `D:\\ProgramFiles\\Anaconda3\\python.exe`。',
                      '文件开头的 STYLE/SPLITS 可调整字体、配色和线型；本脚本不运行训练。'])
        (OUTPUT / 'README.md').write_text('\n'.join(readme) + '\n', encoding='utf-8')
        (OUTPUT / 'captions.md').write_text('\n'.join(captions) + '\n', encoding='utf-8')
        write_json(OUTPUT / 'qa' / f'layout_{self.suffix}.json', self.qa)
        write_json(OUTPUT / 'manifest.json', {
            'created_at': datetime.now(timezone.utc).isoformat(), 'source': str(SOURCE),
            'input_log_sha256': self.input_sha, 'experiment': self.log.get('experiment', {}),
            'status': self.log['status'], 'completed_epochs': self.epoch, 'planned_epochs': 50,
            'is_complete_50_epochs': self.complete, 'best_epoch': self.log.get('best_epoch'),
            'seed_count': 1, 'smoothing': False, 'padding': False, 'test_evaluation_performed': False,
            'plot_runtime': {'python': sys.executable, 'matplotlib': matplotlib.__version__},
            'figures': self.entries})

    def run(self):
        for name in ('png', 'pdf', 'svg', 'grayscale', 'data', 'qa'):
            (OUTPUT / name).mkdir(parents=True, exist_ok=True)
        plt.rcParams.update(STYLE)
        with PdfPages(OUTPUT / f'all_curves_{self.suffix}.pdf') as self.book:
            self.metric_panels(('loss', 'accuracy', 'balanced_accuracy'), 1, 3, 2.95,
                               '01_training_overview', 'Loss、Agreement、Balanced accuracy 总览')
            self.metric_panels(('loss', 'brier_score', 'accuracy', 'balanced_accuracy'), 2, 2, 5.0,
                               '02_four_metrics', '四项训练与验证指标')
            self.metric_panels(('loss',), 1, 1, 3.5, '03_loss', '训练与验证 Loss')
            self.class_recall()
        self.documents()
        print(f'Generated {self.suffix}: {OUTPUT}', flush=True)


if __name__ == '__main__':
    CurvePreview().run()
