"""全池筛选与稳定性图：直接在 PyCharm 运行；只读取已有实验结果。"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

OUTPUT_MODE = "show"  # "show": 只显示；"savefig": 保存 PDF/PNG，不弹窗。
OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'experiment_data/matching_score_legacy_20261019'
COLORS = ('#0072B2', '#D55E00')


class MatchingScreeningStabilityPlot:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.analysis = None

    def read_data(self):
        self.analysis = json.loads((self.output_dir / 'analysis.json').read_text(encoding='utf-8'))

    @staticmethod
    def legend(ax, handles=None, ncol=2, loc='upper right'):
        ax.legend(handles=handles, loc=loc, ncol=ncol, fontsize=8,
                  frameon=True, facecolor='white', edgecolor='0.85', framealpha=0.92,
                  handlelength=1.5, handletextpad=0.4, columnspacing=0.7, borderpad=0.35)

    def screening_stability_figure(self):
        def read_csv(name):
            with (self.output_dir / name).open(encoding='utf-8-sig', newline='') as stream:
                return list(csv.DictReader(stream))

        candidates = read_csv('all_candidate_results.csv')
        summaries = read_csv('all_screening_summary.csv')
        stability = read_csv('stability_summary.csv')
        display = read_csv('stability_display.csv')
        figure, axes = plt.subplots(2, 2, figsize=(7.4, 6.7))
        figure.subplots_adjust(left=0.09, right=0.98, bottom=0.09, top=0.94,
                               hspace=0.42, wspace=0.33)
        actors = [self.analysis['scenes'][scene]['actor_id'] for scene in ('spacecraft', 'regions')]
        for ax, counterpart_type in zip(axes[0], ('spacecraft', 'regions')):
            count = None
            for index, actor in enumerate(actors):
                rows = [row for row in candidates
                        if row['actor_id'] == str(actor) and row['scene_type'] == counterpart_type]
                summary = next(row for row in summaries
                               if row['actor_id'] == str(actor) and row['scene_type'] == counterpart_type)
                count = len(rows)
                if count != int(summary['candidates']):
                    raise ValueError('Full-pool rank plot and summary counts differ.')
                for key in ('distance_rank', 'score_rank'):
                    if sorted(int(row[key]) for row in rows) != list(range(1, count + 1)):
                        raise ValueError('Full-pool ranks must be a permutation of 1..N.')
                kept = [row for row in rows if row['retained'] == 'True']
                overlap = sum(row['distance_retained'] == 'True' for row in kept)
                if len(kept) != int(summary['retained']) or overlap != int(summary['same_budget_overlap']):
                    raise ValueError('Full-pool retention/overlap does not match the summary.')
                ax.scatter([int(row['distance_rank']) for row in rows],
                           [int(row['score_rank']) for row in rows],
                           s=9 if counterpart_type == 'spacecraft' else 18,
                           color=COLORS[index], marker=('o', '^')[index], alpha=0.6,
                           linewidths=0, label='Actor ' + str(actor))
            ax.plot([1, count], [1, count], color='0.5', ls='--', lw=0.8)
            ax.set(xlim=(0, count + 1), ylim=(0, count + 1),
                   xlabel='Distance rank', ylabel='Matching-score rank')
            ax.set_title(('Spacecraft' if counterpart_type == 'spacecraft' else 'Ground regions')
                         + ' ({} per actor)'.format(count), fontsize=9)
            ax.grid(color='0.9', lw=0.5)
            self.legend(ax, ncol=1, loc='upper left')

        ax = axes[1, 0]
        labels, shared, only_score = [], [], []
        for row in summaries:
            retained, overlap = int(row['retained']), int(row['same_budget_overlap'])
            labels.append('{} / {}'.format(row['actor_id'],
                          'Sat.' if row['scene_type'] == 'spacecraft' else 'Region'))
            shared.append(overlap)
            only_score.append(retained - overlap)
        ax.barh(range(4), shared, color='0.65', label='Shared with distance')
        ax.barh(range(4), only_score, left=shared, color=COLORS[0], hatch='///',
                edgecolor='0.25', linewidth=0.35, label='Score only')
        for index, (overlap, extra) in enumerate(zip(shared, only_score)):
            ax.text(overlap + extra + 0.6, index, '{}/{}'.format(overlap, overlap + extra),
                    va='center', fontsize=8)
        ax.set(yticks=range(4), yticklabels=labels, xlim=(0, 42), ylim=(3.6, -1.8),
               xlabel='Candidates retained by matching score')
        ax.set_xticks([0, 8, 16, 24, 32, 40])
        ax.set_title('Overlap with the equal-budget distance set', fontsize=9)
        self.legend(ax, ncol=1, loc='upper right')
        ax.grid(axis='x', color='0.9', lw=0.5)
        ax.set_axisbelow(True)

        ax = axes[1, 1]
        ax.axis('off')
        ax.set_title('Geometry-scale and budget checks', fontsize=9)
        variants = ('baseline', 'geometry_0.8', 'geometry_1.2', 'budget_32', 'budget_48')
        variant_labels = ('Baseline', 'Geometry ×0.8', 'Geometry ×1.2', 'Top32', 'Top48')
        table_rows = []
        for variant, label in zip(variants, variant_labels):
            rows = [row for row in stability if row['variant'] == variant]
            if len(rows) != 4:
                raise ValueError('Each stability setting needs two actors and two types.')
            counts = {}
            for counterpart_type in ('spacecraft', 'regions'):
                values = {int(row['retained']) for row in rows if row['scene_type'] == counterpart_type}
                if len(values) != 1:
                    raise ValueError('The compact stability table requires equal counts for both actors.')
                counts[counterpart_type] = values.pop()
            rho = min(float(row['spearman_to_baseline']) for row in rows)
            table_rows.append([label, '{}/{}'.format(counts['spacecraft'], counts['regions']),
                               '{:.4f}'.format(rho)])
        table = ax.table(cellText=table_rows, colLabels=['Setting', 'Kept\nSat./Region', 'Min. rank\ncorrelation'],
                         cellLoc='center', colWidths=[0.42, 0.29, 0.29], bbox=[0, 0.24, 1, 0.73])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        for (row, _), cell in table.get_celld().items():
            cell.set_edgecolor('0.8')
            cell.set_linewidth(0.4)
            if row == 0:
                cell.set_facecolor('0.94')
        baseline = {(row['scene'], row['candidate_id']): row['retained']
                    for row in display if row['variant'] == 'baseline'}
        if len(baseline) != 10 or len(display) != 50:
            raise ValueError('Stability display must contain ten objects across five settings.')
        unchanged = all(row['retained'] == baseline[(row['scene'], row['candidate_id'])] for row in display)
        ax.text(0.5, 0.12, 'Kept counts apply to each actor; rank correlation\nis the minimum across both actors and types.',
                ha='center', va='center', transform=ax.transAxes, fontsize=8)
        ax.text(0.5, -0.015, 'Displayed-object retention: ' + ('unchanged (10/10)' if unchanged else 'changes observed'),
                ha='center', transform=ax.transAxes, fontsize=8)
        for index, ax in enumerate(axes.flat):
            ax.text(-0.13, 1.06, '(' + chr(ord('a') + index) + ')',
                    transform=ax.transAxes, fontsize=9, fontweight='bold')
        return figure

    def save_figure(self, figure, name):
        single_spatial = len(figure.axes) == 1 and hasattr(figure.axes[0], 'zaxis')
        for extension in ('pdf', 'png'):
            figure.savefig(self.output_dir / (name + '.' + extension), dpi=300,
                           bbox_inches=None if single_spatial else 'tight',
                           pad_inches=0.04, facecolor='white')

    def run(self, output_mode=OUTPUT_MODE):
        if output_mode not in ('show', 'savefig'):
            raise ValueError("OUTPUT_MODE must be 'show' or 'savefig'.")
        self.read_data()
        style = {'font.family': 'Times New Roman', 'font.size': 8,
                 'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                 'axes.linewidth': 0.6, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                 'interactive': False}
        with plt.rc_context(style):
            figure = self.screening_stability_figure()
            if output_mode == 'savefig':
                self.save_figure(figure, 'results_matching_screening_stability')
        if output_mode == 'show':
            plt.show()
        else:
            plt.close(figure)
        return figure


if __name__ == '__main__':
    MatchingScreeningStabilityPlot(OUTPUT_DIR).run(OUTPUT_MODE)
