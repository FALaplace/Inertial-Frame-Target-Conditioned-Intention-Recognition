"""Section 5.3: reference ambiguity and physical case analysis.

Open in PyCharm with D:/ProgramFiles/Anaconda3/python.exe.
PhysicalCaseData uses the training environment separately to prepare physical_cases.json.
All settings for an IDE run are at the bottom; no training or model selection occurs here.
"""
from pathlib import Path

from project_paths import resolve_project_path
import csv
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
TYPE_TITLES = ('Spacecraft counterpart', 'Ground-region counterpart')
GROUPS = ('High', 'Moderate', 'Low')
CLASSES = tuple('C' + str(i) for i in range(8))
FEASIBLE = {'satellite': (0, 1, 2, 3, 4, 5), 'ground_region': (0, 1, 5, 6, 7)}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError('Empty table: ' + str(path))
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def js_divergence(targets, probabilities):
    """The evaluator's natural-log JS, with 0 log(0/m) = 0; no clipping/masking."""
    q, p = np.asarray(targets, dtype=np.float64), np.asarray(probabilities, dtype=np.float64)
    if q.shape != p.shape or q.ndim != 2 or q.shape[1] != 8:
        raise ValueError('Expected matching N x 8 distributions.')
    for array in (q, p):
        if (not np.isfinite(array).all() or np.any(array < 0) or np.any(array > 1)
                or not np.allclose(array.sum(axis=1), 1, atol=1e-6, rtol=0)):
            raise ValueError('Invalid input probability distribution.')
    midpoint = (q + p) / 2
    result = np.zeros_like(q)
    for array in (q, p):
        mask = array > 0
        result[mask] += .5 * array[mask] * np.log(array[mask] / midpoint[mask])
    return result.sum(axis=1)


class Phase3TestAnalysis:
    def __init__(self, project_root=IDEA, output=None, prediction_paths=None, source_manifest=None):
        self.root = Path(project_root)
        self.base_output = self.root / 'experiment_data/baseline_test_analysis_20260917'
        self.output = Path(output) if output else self.base_output
        self.physical_path = self.base_output / 'preparation/physical_cases.json'
        # Optional {(model_key, seed): Path(...)} overrides for another host's exports.
        self.prediction_paths = prediction_paths or {}
        self.source_manifest_path = Path(source_manifest) if source_manifest else None
        self.inputs = {}
        self.predictions = {}
        self.js = {}
        self.seed = 42

    def record_input(self, path):
        path = Path(path)
        self.inputs[str(path)] = {'sha256': sha256(path), 'size_bytes': path.stat().st_size}

    def read_data(self):
        frozen = self.root / 'dataset'
        label_path = frozen / 'labels_main_18000_v1.json'
        split_path = self.root / 'experiment_data/splits/actor_isolated_13000_2500_2500_v1__labels_18000_v1.json'
        provenance_path = frozen / 'sample_provenance_frozen.json'
        self.labels = {x['sample_index']: x for x in read_json(label_path)}
        self.split_manifest = read_json(split_path)
        self.meta = {x['sample_index']: x for x in self.split_manifest['samples']}
        self.provenance = {x['sample_index']: x for x in read_json(provenance_path)}
        self.y = {i: np.array([x['y']['c' + str(k)] for k in range(8)]) for i, x in self.labels.items()}
        for path in (label_path, split_path, provenance_path, frozen / 'freeze_manifest.json'):
            self.record_input(path)
        self.dataset_path = resolve_project_path(self.split_manifest['dataset']['path'], self.root)
        self.record_input(self.dataset_path)
        if self.inputs[str(self.dataset_path)]['sha256'] != self.split_manifest['dataset']['sha256']:
            raise ValueError('Frozen dataset hash differs from the split.')
        if self.inputs[str(label_path)]['sha256'] != self.split_manifest['dataset']['labels_sha256']:
            raise ValueError('Frozen labels differ from the split.')
        self.indices = np.array(sorted(i for i, x in self.meta.items() if x['split'] == 'test'))
        if len(self.indices) != 2500:
            raise ValueError('Expected 2,500 Test samples.')
        self.positions = {int(i): n for n, i in enumerate(self.indices)}
        actor_sets = [{x['actor_norad_cat_id'] for x in self.meta.values() if x['split'] == split}
                      for split in ('train', 'validation', 'test')]
        if any(actor_sets[i] & actor_sets[j] for i in range(3) for j in range(i)):
            raise ValueError('Actors cross dataset partitions.')
        for i in self.meta:
            for key in ('counterpart_type', 'actor_norad_cat_id'):
                if self.meta[i][key] != self.provenance[i][key]:
                    raise ValueError('Sample identity differs at ' + str(i))
            key = 'counterpart_id' if 'counterpart_id' in self.provenance[i] else 'counterpart'
            if self.meta[i][key] != self.provenance[i][key]:
                raise ValueError('Counterpart identity differs at ' + str(i))
            if self.labels[i]['sample_id'] != self.provenance[i]['sample_id']:
                raise ValueError('Stable sample ID differs at ' + str(i))
        self.checkpoints = {}
        for phase in (2, 3):
            evaluation = self.root / f'experiment_data/phase{phase}_scratch_18000_gpu_v1/evaluation'
            paths = [evaluation / 'selected_checkpoints.json']
            if phase == 3:
                paths.append(evaluation / 'selected_random_forests.json')
            for path in paths:
                doc = read_json(path)
                if doc.get('test_used_for_selection', False):
                    raise ValueError('Checkpoint selection used Test.')
                self.record_input(path)
                for x in doc.get('selected_checkpoints', doc.get('selected_forests', [])):
                    self.checkpoints[x['run_id']] = x
        self.prediction_sources = []
        self.exported_targets = None
        for key, label, phase, _, _ in MODELS:
            for seed in SEEDS:
                run_id = f'{key}__seed{seed}'
                default = self.root / f'experiment_data/phase{phase}_scratch_18000_gpu_v1/evaluation/runs/{run_id}/test/predictions.npz'
                path = Path(self.prediction_paths.get((key, seed), default))
                with np.load(path, allow_pickle=False) as data:
                    ids = np.asarray(data['sample_indices'], dtype=int)
                    if len(set(ids)) != len(ids) or set(ids) != set(self.indices):
                        raise ValueError('Different sample set: ' + str(path))
                    order = np.argsort(ids)
                    target = data['targets'][order].astype(np.float64)
                    p = data['probabilities'][order].astype(np.float64)
                    expected = np.array([self.y[int(i)] for i in self.indices])
                    if not np.allclose(target, expected, atol=1e-6, rtol=0):
                        raise ValueError('Reference labels differ: ' + str(path))
                    if self.exported_targets is None:
                        self.exported_targets = target.copy()
                    elif not np.array_equal(target, self.exported_targets):
                        raise ValueError('Targets differ between model/seed exports: ' + str(path))
                    if 'counterpart_types' in data:
                        expected_types = [self.meta[int(i)]['counterpart_type'] for i in self.indices]
                        if not np.array_equal(data['counterpart_types'][order], expected_types):
                            raise ValueError('Counterpart types differ: ' + str(path))
                for typ in TYPES:
                    rows = [n for n, i in enumerate(self.indices) if self.meta[int(i)]['counterpart_type'] == typ]
                    banned = sorted(set(range(8)) - set(FEASIBLE[typ]))
                    if np.any(p[np.ix_(rows, banned)] != 0) or np.any(target[np.ix_(rows, banned)] != 0):
                        raise ValueError('Unexpected probability in infeasible class: ' + str(path))
                self.predictions[key, seed] = p
                # Preserve the evaluator's exported target precision.
                self.js[key, seed] = js_divergence(target, p)
                self.record_input(path)
                self.prediction_sources.append({'model': key, 'seed': seed, 'path': str(path),
                    'sha256': self.inputs[str(path)]['sha256'],
                    'checkpoint': None if (key, seed) in self.prediction_paths else self.checkpoints.get(run_id),
                    'checkpoint_association': 'external export association pending' if (key, seed) in self.prediction_paths
                        else 'local selected-checkpoint record'})

    def verify_current_sources(self):
        """Require a binding export manifest; a display-status string is not evidence."""
        if self.source_manifest_path is None:
            raise ValueError('Current manuscript figures need the matching external export manifest.')
        source = read_json(self.source_manifest_path)
        self.record_input(self.source_manifest_path)
        if source.get('dataset_sha256') != self.split_manifest['dataset']['sha256']:
            raise ValueError('External dataset identity differs.')
        if source.get('labels_sha256') != self.split_manifest['dataset']['labels_sha256']:
            raise ValueError('External labels identity differs.')
        split_path = self.root / 'experiment_data/splits/actor_isolated_13000_2500_2500_v1__labels_18000_v1.json'
        if source.get('split_sha256') != self.inputs[str(split_path)]['sha256']:
            raise ValueError('External split identity differs.')
        records = {(x['model'], x['seed']): x for x in source['predictions']}
        for item in self.prediction_sources:
            record = records.get((item['model'], item['seed']), {})
            if record.get('prediction_sha256') != item['sha256']:
                raise ValueError('External prediction binding differs: ' + item['path'])
            checkpoint = record.get('checkpoint', {})
            if not checkpoint.get('checkpoint_sha256') or record.get('test_used_for_selection') is not False:
                raise ValueError('Missing external checkpoint provenance or selection policy.')
            item['checkpoint'] = checkpoint
            item['checkpoint_association'] = 'matching supplied export manifest; remote weight hash recorded'
        if not source.get('experiment_id'):
            raise ValueError('External export must identify its experiment.')
        self.external_experiment_id = source['experiment_id']

    def make_groups(self):
        self.margin = {i: float(np.sort(y)[-1] - np.sort(y)[-2]) for i, y in self.y.items()}
        self.thresholds, self.group_rows, self.group_members = {}, [], {}
        for typ in TYPES:
            train = [i for i in self.meta if self.meta[i]['split'] == 'train' and self.meta[i]['counterpart_type'] == typ]
            a, b = np.quantile([self.margin[i] for i in train], [1 / 3, 2 / 3], interpolation='linear')
            self.thresholds[typ] = {'a': float(a), 'b': float(b), 'train_count': len(train),
                                    'quantile_method': 'linear', 'merged_groups': []}
            for i in self.indices:
                i = int(i)
                if self.meta[i]['counterpart_type'] != typ:
                    continue
                group = GROUPS[0 if self.margin[i] <= a else 1 if self.margin[i] <= b else 2]
                row = self.identity(i)
                row.update(margin=self.margin[i], group=group, reference_class=CLASSES[self.y[i].argmax()],
                           maneuver_count=self.meta[i]['maneuver_count'], has_maneuver=self.meta[i]['maneuver_count'] > 0,
                           threshold_a=float(a), threshold_b=float(b))
                self.group_rows.append(row)
            for group in GROUPS:
                members = [x['sample_index'] for x in self.group_rows if x['counterpart_type'] == typ and x['group'] == group]
                if members:
                    self.group_members[typ, group] = members
                else:
                    self.thresholds[typ]['merged_groups'].append(group + ': empty group omitted; ties kept together')
        self.metric_rows, self.summary_rows, self.composition = [], [], []
        for (typ, group), ids in self.group_members.items():
            positions = [self.positions[i] for i in ids]
            actors = len({self.meta[i]['actor_norad_cat_id'] for i in ids})
            maneuver_fraction = float(np.mean([self.meta[i]['maneuver_count'] > 0 for i in ids]))
            for class_index, class_name in enumerate(CLASSES):
                count = sum(self.y[i].argmax() == class_index for i in ids)
                self.composition.append(dict(counterpart_type=typ, group=group, reference_class=class_name,
                    class_count=count, class_fraction=count / len(ids), n=len(ids), actor_count=actors,
                    maneuver_fraction=maneuver_fraction))
            for key, label, _, _, _ in MODELS:
                means = [float(self.js[key, seed][positions].mean()) for seed in SEEDS]
                for seed, mean in zip(SEEDS, means):
                    self.metric_rows.append(dict(model=key, model_label=label, seed=seed, counterpart_type=typ,
                        group=group, n=len(ids), actor_count=actors, mean_js=mean))
                self.summary_rows.append(dict(model=key, model_label=label, counterpart_type=typ, group=group,
                    n=len(ids), actor_count=actors, mean_js=float(np.mean(means)), sd_js=float(np.std(means, ddof=1)),
                    seed_count=len(SEEDS), threshold_a=self.thresholds[typ]['a'], threshold_b=self.thresholds[typ]['b'],
                    maneuver_fraction=maneuver_fraction))

    def identity(self, i):
        m = self.meta[i]
        return dict(sample_index=i, sample_id=self.labels[i]['sample_id'], actor=m['actor'],
                    actor_norad_cat_id=m['actor_norad_cat_id'], counterpart=m['counterpart'],
                    counterpart_type=m['counterpart_type'])

    def select_cases(self):
        self.candidate_rows, self.case_rules, self.selected = [], {}, {}
        for typ, pair, case_key in (('satellite', {2, 4}, 'c4_c2'), ('ground_region', {1, 7}, 'c7_c1')):
            pool = [int(i) for i in self.indices if self.meta[int(i)]['counterpart_type'] == typ
                    and set(np.argsort(self.y[int(i)])[-2:]) == pair]
            rule = 'reference top-two classes equal target pair'
            if not pool:
                pool = [int(i) for i in self.indices if self.meta[int(i)]['counterpart_type'] == typ
                        and int(self.y[int(i)].argmax()) in pair and all(self.y[int(i)][k] > 0 for k in pair)]
                rule = 'no strict candidates; dominant class in pair and other class has nonzero support'
            if not pool:
                raise ValueError('No reference-label candidates: ' + typ)
            median = float(np.median([self.margin[i] for i in pool]))
            candidates = sorted(pool, key=lambda i: (abs(self.margin[i] - median), self.labels[i]['sample_id']))[:10]
            available = [i for i in candidates if str(i) in self.physical]
            if not available:
                raise ValueError('Physical cache missing candidates: ' + str(candidates))
            selected = available[0]
            self.selected[case_key] = selected
            self.case_rules[case_key] = dict(counterpart_type=typ, classes=sorted(pair), pool_count=len(pool),
                rule=rule, median_margin=median, ranked_candidates=candidates, selected_index=selected,
                selection='nearest candidate-pool median reference margin, stable sample ID tie-break, complete physical data')
            for rank, i in enumerate(candidates, 1):
                row = self.identity(i)
                row.update(case=case_key, rank=rank, margin=self.margin[i], pool_median_margin=median,
                    maneuver_count=self.meta[i]['maneuver_count'], physical_complete=str(i) in self.physical,
                    selected=i == selected, reason=self.case_rules[case_key]['selection'] if i == selected else 'candidate')
                row.update({'reference_' + c: float(self.y[i][k]) for k, c in enumerate(CLASSES)})
                for key, _, _, _, _ in MODELS:
                    pos = self.positions[i]
                    row[key + '_js_seed42'] = float(self.js[key, 42][pos])
                    row[key + '_predicted_class_seed42'] = CLASSES[self.predictions[key, 42][pos].argmax()]
                self.candidate_rows.append(row)

    def read_physical(self):
        self.physical = read_json(self.physical_path)
        if 'samples' in self.physical:
            self.physical = self.physical['samples']
        self.record_input(self.physical_path)
        for index, item in self.physical.items():
            i = int(index)
            for field in ('sample_id', 'actor_norad_cat_id', 'counterpart_type'):
                expected = self.identity(i)[field]
                if str(item[field]) != str(expected):
                    raise ValueError('Physical record identity differs: ' + index + '/' + field)
            if item['counterpart'] != self.meta[i].get('counterpart_id', self.meta[i]['counterpart']):
                raise ValueError('Physical counterpart identity differs: ' + index)
            if not np.allclose(item['y'], self.y[i], atol=1e-6, rtol=0):
                raise ValueError('Physical record labels differ: ' + index)
            time = np.asarray(item['time_offset_s'])
            if len(time) < 2 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
                raise ValueError('Invalid physical time grid: ' + index)
            if self.meta[i]['maneuver_count'] and not item['counterfactual_valid']:
                raise ValueError('Missing counterfactual for maneuver case: ' + index)
            for branch in ('factual', 'counterfactual'):
                for field in ('distance_km', 'range_rate_km_s', 'elevation_deg', 'access_flag'):
                    values = np.asarray(item[branch][field])
                    if values.shape != time.shape or not np.isfinite(values).all():
                        raise ValueError('Incomplete physical series: ' + index + '/' + field)

    @staticmethod
    def style():
        plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 10, 'axes.labelsize': 11,
            'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9,
            'axes.linewidth': .65, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

    def ambiguity_figure(self):
        fig, axes = plt.subplots(1, 2, figsize=(7.48, 3.65), sharey=True)
        fig.subplots_adjust(left=.092, right=.987, bottom=.22, top=.76, wspace=.13)
        lookup = {(r['model'], r['counterpart_type'], r['group']): r for r in self.summary_rows}
        max_y = max(r['mean_js'] + r['sd_js'] for r in self.summary_rows) * 1.15
        handles = []
        for ax, typ, title in zip(axes, TYPES, TYPE_TITLES):
            groups = [g for g in GROUPS if (typ, g) in self.group_members]
            x = np.arange(len(groups))
            for mi, (key, label, _, color, marker) in enumerate(MODELS):
                records = [lookup[key, typ, group] for group in groups]
                h = ax.errorbar(x + (mi - 2.5) * .10, [r['mean_js'] for r in records],
                    yerr=[r['sd_js'] for r in records], fmt=marker, linestyle='none',
                    color=color, markersize=5.2 if key == 'full' else 4.1,
                    elinewidth=1.5 if key == 'full' else .95, capsize=2.2, label=label)
                if ax is axes[0]:
                    handles.append(h)
            ax.set_xticks(x, [g + '\n' + f'n = {len(self.group_members[typ, g])}' for g in groups])
            ax.set(xlim=(-.5, len(groups) - .5), ylim=(0, max_y), xlabel='Reference-label ambiguity')
            ax.set_title(title, fontsize=11, pad=9)
            ax.grid(axis='y', color='.9', linewidth=.6)
            ax.set_axisbelow(True)
            ax.spines[['top', 'right']].set_visible(False)
        axes[0].set_ylabel('Mean JS divergence')
        fig.legend(handles, [m[1] for m in MODELS], loc='upper center', bbox_to_anchor=(.52, .96),
                   ncol=6, frameon=False, columnspacing=.85, handletextpad=.25, fontsize=8.5)
        return fig

    def physical_axes(self, fig, spec, case_key):
        i = self.selected[case_key]
        data = self.physical[str(i)]
        grid = spec.subgridspec(2, 1, hspace=.16)
        axes = [fig.add_subplot(grid[0]), fig.add_subplot(grid[1])]
        typ = data['counterpart_type']
        variables = [('distance_km', 'Distance (km)'), ('range_rate_km_s', 'Range rate (km/s)')] if typ == 'satellite' else [
            ('elevation_deg', 'Elevation (deg)'), ('access_flag', 'Access (0/1)')]
        x = np.asarray(data['time_offset_s']) / 60
        for row, (ax, (field, ylabel)) in enumerate(zip(axes, variables)):
            for branch, label, style, color in (('factual', 'Factual', '-', '#0072B2'),
                                               ('counterfactual', 'Counterfactual', '--', '#D55E00')):
                drawstyle = 'steps-post' if field == 'access_flag' else 'default'
                ax.plot(x, data[branch][field], style, color=color, label=label, lw=1.15, drawstyle=drawstyle)
            for event in data['maneuvers']:
                ax.axvline(event['time_offset_s'] / 60, color='.35', ls=':', lw=.9)
            if field == 'elevation_deg':
                ax.axhline(10, color='.6', ls=':', lw=.7)
                ax.text(.02, .96, 'Access: \u2265 10\u00b0', transform=ax.transAxes, fontsize=8,
                        ha='left', va='top')
            if field == 'access_flag':
                ax.set_yticks([0, 1])
                ax.set_ylim(-.12, 1.12)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_xlim(x[0], x[-1])
            ax.grid(color='.91', linewidth=.5)
            ax.tick_params(labelsize=8.5)
            if row == 0:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel('Time from window start (min)', fontsize=9)
        label = 'C4 / C2: spacecraft' if typ == 'satellite' else 'C7 / C1: ground region'
        axes[0].set_title(label, fontsize=11, pad=7)
        return axes

    def probability_axis(self, fig, spec, case_key):
        i = self.selected[case_key]
        pos = self.positions[i]
        values = np.vstack([self.exported_targets[pos]] + [self.predictions[key, self.seed][pos] for key, *_ in MODELS])
        feasible = FEASIBLE[self.meta[i]['counterpart_type']]
        mask = np.ones_like(values, dtype=bool)
        mask[:, feasible] = False
        cmap = plt.get_cmap('cividis').copy()
        cmap.set_bad('#dddddd')
        ax = fig.add_subplot(spec)
        mesh = ax.pcolormesh(np.ma.array(values, mask=mask), cmap=cmap, vmin=0, vmax=1,
                            edgecolors='white', linewidth=.4, rasterized=False)
        ax.set(xlim=(0, 8), ylim=(7, 0))
        ax.set_xticks(np.arange(8) + .5, CLASSES)
        ax.set_yticks(np.arange(7) + .5, ['Reference'] + [m[1] for m in MODELS])
        ax.tick_params(length=0, labelsize=8.5, pad=3)
        ax.set_xlabel('Intention class', fontsize=9)
        for r in range(7):
            for c in range(8):
                if c in feasible:
                    ax.text(c + .5, r + .5, f'{values[r,c]:.2f}', fontsize=7.7, ha='center', va='center',
                            color='white' if values[r, c] < .48 else 'black')
                else:
                    ax.text(c + .5, r + .5, '\u2014', fontsize=7.7, ha='center', va='center', color='.4')
        return ax, mesh

    def case_figure(self, cases=('c4_c2', 'c7_c1')):
        n = len(cases)
        fig = plt.figure(figsize=(7.48 if n == 2 else 4.7, 6.45))
        outer = fig.add_gridspec(2, n, left=.17 if n == 2 else .255, right=.97,
                                bottom=.21, top=.89, hspace=.33, wspace=.66,
                                height_ratios=[1.2, 1])
        for column, case in enumerate(cases):
            physical = self.physical_axes(fig, outer[0, column], case)
            _, mesh = self.probability_axis(fig, outer[1, column], case)
        handles, labels = physical[0].get_legend_handles_labels()
        handles.append(plt.Line2D([], [], color='.35', ls=':', lw=.9))
        labels.append('Maneuver')
        fig.legend(handles, labels, ncol=3, loc='upper center', bbox_to_anchor=(.54, .995), frameon=False,
                   columnspacing=1.1, handlelength=2.2, fontsize=9)
        bar = fig.colorbar(mesh, cax=fig.add_axes([.38 if n == 2 else .36, .085, .37 if n == 2 else .45, .018]),
                          orientation='horizontal', ticks=[0, .25, .5, .75, 1])
        bar.solids.set_rasterized(False)
        bar.set_label('Probability (seed 42)', fontsize=9, labelpad=3)
        bar.ax.tick_params(labelsize=8)
        fig.text(.54 if n == 2 else .58, .13, 'Grey cells: infeasible classes', ha='center', fontsize=8.5)
        return fig

    def physical_preview_figure(self):
        """Dataset-only evidence, independent of the missing updated model predictions."""
        fig = plt.figure(figsize=(7.48, 3.9))
        grid = fig.add_gridspec(1, 2, left=.085, right=.985, bottom=.15, top=.81, wspace=.32)
        for column, case in enumerate(('c4_c2', 'c7_c1')):
            axes = self.physical_axes(fig, grid[column], case)
        handles, labels = axes[0].get_legend_handles_labels()
        handles.append(plt.Line2D([], [], color='.35', ls=':', lw=.9))
        labels.append('Maneuver')
        fig.legend(handles, labels, ncol=3, loc='upper center', bbox_to_anchor=(.54, .99),
                   frameon=False, columnspacing=2, fontsize=10)
        fig.text(.995, .006, 'Frozen-dataset physical evidence; model comparison pending',
                 ha='right', va='bottom', fontsize=7, color='.35')
        return fig

    def export_tables(self):
        tables = {'ambiguity_groups': self.group_rows, 'ambiguity_metrics_by_seed': self.metric_rows,
                  'ambiguity_summary': self.summary_rows, 'ambiguity_group_composition': self.composition,
                  'case_candidates': self.candidate_rows}
        probabilities, physical_rows = [], []
        for case, i in self.selected.items():
            pos = self.positions[i]
            for seed in SEEDS:
                for key, label, *_ in MODELS:
                    row = self.identity(i)
                    row.update(case=case, model=key, model_label=label, seed=seed,
                               js=float(self.js[key, seed][pos]), reference_class=CLASSES[self.y[i].argmax()],
                               predicted_class=CLASSES[self.predictions[key, seed][pos].argmax()])
                    row.update({'y_' + c: float(self.exported_targets[pos, k]) for k, c in enumerate(CLASSES)})
                    row.update({'p_' + c: float(self.predictions[key, seed][pos, k]) for k, c in enumerate(CLASSES)})
                    probabilities.append(row)
            data = self.physical[str(i)]
            for t, seconds in enumerate(data['time_offset_s']):
                row = dict(case=case, sample_index=i, sample_id=self.labels[i]['sample_id'],
                           time_utc=data['timestamps_utc'][t], time_offset_s=seconds, time_minutes=seconds / 60)
                for branch in ('factual', 'counterfactual'):
                    for field, values in data[branch].items():
                        if isinstance(values, list) and len(values) == len(data['time_offset_s']):
                            row[branch + '_' + field] = values[t]
                physical_rows.append(row)
        tables.update(case_probabilities=probabilities, case_physical_series=physical_rows)
        for name, rows in tables.items():
            write_csv(self.output / (name + '.csv'), rows)

    def save_figure(self, fig, name, source_status):
        if source_status == 'local_preview':
            fig.text(.995, .006, 'Local-record preview; manuscript version pending',
                     ha='right', va='bottom', fontsize=7, color='.35')
        for extension in ('png', 'pdf'):
            fig.savefig(self.output / (name + '.' + extension), dpi=350, facecolor='white')

    def model_observations(self):
        """Summarize the displayed predictions, without preselecting favorable claims."""
        lines = ['## 本次预览的模型表现', '',
                 '| 类型 / 歧义组 | RF JS，mean ± SD | Full JS，mean ± SD | 均值较优模型 |',
                 '|---|---:|---:|---|']
        lookup = {(r['model'], r['counterpart_type'], r['group']): r for r in self.summary_rows}
        winners = []
        for typ in TYPES:
            for group in GROUPS:
                if (typ, group) not in self.group_members:
                    continue
                rf, full = [lookup[key, typ, group] for key in ('analytic_random_forest', 'full')]
                winner = 'Full' if full['mean_js'] < rf['mean_js'] else 'RF' if rf['mean_js'] < full['mean_js'] else '相同'
                winners.append(winner)
                lines.append(f'| {typ} / {group} | {rf["mean_js"]:.6f} ± {rf["sd_js"]:.6f} | '
                             f'{full["mean_js"]:.6f} ± {full["sd_js"]:.6f} | {winner} |')
        lines += ['', f'Full 在 {len(winners)} 个分组中的 {winners.count("Full")} 组平均 JS 低于 RF；'
                  f'RF 在 {winners.count("RF")} 组更低。误差棒表示五个种子的样本标准差。', '',
                  '分组同时改变了主参考类别和机动样本的组成，具体比例见 ambiguity_group_composition.csv。'
                  '因此，组间 JS 的变化需要结合类别组成分析；歧义分组本身不提供模块作用的因果检验。', '',
                  '### 两个个例的 seed 42 对比', '',
                  '| 模型 | C4/C2 个例：JS / 主预测类 | C7/C1 个例：JS / 主预测类 |',
                  '|---|---:|---:|']
        for key, label, *_ in MODELS:
            cells = []
            for case in ('c4_c2', 'c7_c1'):
                pos = self.positions[self.selected[case]]
                p = self.predictions[key, self.seed][pos]
                cells.append(f'{self.js[key, self.seed][pos]:.6f} / {CLASSES[p.argmax()]}')
            lines.append('| ' + label + ' | ' + ' | '.join(cells) + ' |')
        lines.append('')
        for case, i in self.selected.items():
            pos = self.positions[i]
            best = min(MODELS, key=lambda m: self.js[m[0], self.seed][pos])[1]
            full_class = CLASSES[self.predictions['full', self.seed][pos].argmax()]
            ref_class = CLASSES[self.exported_targets[pos].argmax()]
            lines.append(f'样本 {i} 的参考主类为 {ref_class}，Full 的主预测类为 {full_class}；'
                         f'六个模型中该个例的 JS 最低者是 {best}。该结果来自统一 seed {self.seed}，'
                         '完整五种子概率保存在 case_probabilities.csv。')
            lines.append('')
        return lines

    def write_report(self, source_status, exported):
        status = ('已按本机完整逐样本结果导出待核对预览图。' if source_status == 'local_preview' else
                  '模型比较图待新版逐样本数据；已导出冻结数据集的物理证据准备图。')
        if source_status == 'verified_current':
            status = '已按匹配的逐样本导出及来源清单生成模型比较图。'
        lines = ['# 5.3 测试集歧义与典型样本：结果说明', '', status, '',
            '## 数据来源与当前缺项', '',
            '本目录的 CSV 来自 manifest.json 逐一记录的预测文件。数据为冻结 18,000 条记录；Test 为 2,500 条，按 sample_index 对齐，并核对参考标签、actor、counterpart、类型和数据文件哈希。', '']
        if source_status != 'verified_current':
            lines += ['本机预测对应原始正式评估，尚未对应后来更新的论文图表。RF 的本机 JS 为 0.01039540，用户最新表为 0.01859540；Full 的本机 C3/C4/C5 Recall 为 65.2239%/36.0406%/57.3913%，当前绘图手填值为 75.2%/66.0%/72.4%。', '',
                '**需要补入：新版 Analytic RF、Full 的 seeds 7、17、42、73、101，各自 Test 的逐样本预测和与之匹配的选模/拟合记录。** NPZ 至少含 sample_indices、targets、probabilities；需要对应 dataset/labels/split 的哈希及模型来源。其他四模型本机导出已齐全，若另一主机也重测过它们，应一并提供新版。', '',
                '当前本机派生统计用于数据核查，不作为最新版论文数值。分组的 SD 必须从相应五种子的逐样本预测重新计算，不能把总表 SD 直接套给每个歧义组。', '']
        lines += ['## 歧义分组', '',
            '按 counterpart 类型分别使用 Train 原始参考标签的 top-1 minus top-2 margin。线性插值求 1/3 和 2/3 分位数，再以固定阈值分 Test。High: margin≤a；Moderate: a<margin≤b；Low: margin>b。', '',
            '| 类型 | a | b | High n / actors | Moderate n / actors | Low n / actors |',
            '|---|---:|---:|---:|---:|---:|']
        for typ in TYPES:
            cells = []
            for group in GROUPS:
                ids = self.group_members.get((typ, group), [])
                cells.append(f'{len(ids)} / {len({self.meta[i]["actor_norad_cat_id"] for i in ids})}')
            lines.append(f'| {typ} | {self.thresholds[typ]["a"]:.8f} | {self.thresholds[typ]["b"]:.8f} | ' + ' | '.join(cells) + ' |')
        lines += ['', '六组均非空，没有拆散同值标签或调整阈值。JS 使用自然对数；零概率项贡献为零。先计算每样本 JS，再每种子组均值，最后求五种子均值及样本 SD（ddof=1）。参考标签取预测文件实际使用的 float32 转 float64 值；分组取冻结 JSON 原始值，两者精度来源记录在 manifest 中。', '',
            'ambiguity_group_composition.csv 同时记录各组主参考类别组成和有机动样本比例。组间表现差异需要结合这些组成信息解释。', '',
            '## 选样与物理证据', '',
            'C4/C2 严格 top-2 候选池有 114 条，C7/C1 有 41 条。每池按参考 margin 距离中位数的绝对差排序，同值按稳定 sample_id 排序；保留前 10 条，取物理字段完整的首条。未以 Full 对 baseline 的优势排序。', '']
        for case, i in self.selected.items():
            d = self.physical[str(i)]
            lines += [f'### {case}：sample_index={i}', '', f'- sample_id：`{d["sample_id"]}`',
                f'- actor：{self.meta[i]["actor"]}（NORAD {d["actor_norad_cat_id"]}）；counterpart：{self.meta[i]["counterpart"]}。',
                f'- UTC 窗口：{d["timestamps_utc"][0]} 至 {d["timestamps_utc"][-1]}；共 {len(d["time_offset_s"])} 点。']
            for e in d['maneuvers']:
                lines.append(f'- 机动：窗口起点后 {e["time_offset_s"]/60:g} min，Δv={1000*e["delta_v_norm_km_s"]:g} m/s。')
            if case == 'c4_c2':
                f, c = np.asarray(d['factual']['distance_km']), np.asarray(d['counterfactual']['distance_km'])
                lines += [f'- 事实/反事实最近距离：{f.min():.3f}/{c.min():.3f} km。参考 C4={self.y[i][4]:.4f}、C2={self.y[i][2]:.4f}。',
                    '- 曲线呈现一次接近后再远离。C4/C2 的参考支持接近，但该个例不构成长期稳定驻留的物理证据。']
            else:
                f, c = d['factual'], d['counterfactual']
                lines += [f'- 事实/反事实最大仰角：{max(f["elevation_deg"]):.3f}/{max(c["elevation_deg"]):.3f} deg；仰角≥10 deg 的采样点数为 {int(sum(f["access_flag"]))}/{int(sum(c["access_flag"]))}。参考 C7={self.y[i][7]:.4f}、C1={self.y[i][1]:.4f}。',
                    '- 数据是地面中心点的访问几何。图显示访问时刻和持续采样点数变化；没有区域边界或传感器 footprint，不能由此计算覆盖面积扩展。阶梯线按60秒采样显示，区间边界不是精确求根的出入时刻。']
            lines.append('')
        if exported:
            lines += self.model_observations()
        lines += ['反事实使用正式数据处理中的逆脉冲和两体传播重建，与训练使用的定义一致。事件线标原始机动时间；采样点上的施加时刻另保存在物理 JSON。物理量从原始 km、km/s 状态计算，不来自归一化网络输入。这些证据不是所有模型的共同输入，事实序列 baseline 未直接使用反事实和解析特征。', '',
            '## 图注草案', '',
            '**Figure A, once the result version is matched.** Test-set JS divergence across reference-label ambiguity groups for spacecraft and ground-region counterparts. Groups are defined using the one-third and two-thirds quantiles of the training-label margin within each counterpart type. Points and error bars denote the mean and sample standard deviation of five seed-specific group means. Sample counts are shown below each group. JS divergence uses natural logarithms.', '',
            '**Figure B, once the result version is matched.** Physical evidence and predicted probability distributions for two reference-selected test cases. The spacecraft case contrasts approach with proximity stay; the ground-region case compares access to the counterpart point under factual and counterfactual trajectories. Dotted vertical lines mark recorded maneuver times. Counterfactual trajectories use the same inverse-impulse, two-body reconstruction as the model input pipeline. Probability maps show the reference distribution and six models at seed 42. Grey cells denote classes excluded by the counterpart-type mask.', '',
            '**Physical preparation figure.** Physical evidence for two test cases selected by the median reference-label margin within the C4/C2 and C7/C1 candidate pools. Solid and dashed lines show factual and reconstructed counterfactual trajectories. Dotted vertical lines mark recorded maneuver times. Ground-point access is defined by an elevation angle of at least 10 degrees on the one-minute time grid.', '',
            '## 文件与运行', '',
            '- `scripts/phase3_test_analysis.py`：分组、逐种子统计、选样、绘图和CSV导出；使用基础 Anaconda Python，在文件末尾设置 IDE 运行项。',
            '- `scripts/phase3_case_physical.py`：物理序列提取；使用 intention-ir 环境。已有缓存可直接供绘图脚本读取。',
            '- `preparation/physical_cases.json`：20个候选的原始单位事实/反事实序列、时间、事件及定义。',
            '- 新模型预测通过 `prediction_paths` 指定；正式版还需匹配的 `source_manifest`，不会把外部预测自动关联到本机checkpoint。',
            '- `run()` 保留已有目录成果；再次运行写入新的版本目录。IDE 入口已按用户确认设置为导出本机结果预览，图面保留版本说明。', '',
            '图注已按 academic-humanizer 和 anti-defensive-writing 检查，保留证据范围和统计定义；未改动 Overleaf。']
        (self.output / '结果说明.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    def run(self, export_model_figures=False, source_status='awaiting_updated_exports', show=False, data_only=False):
        """Prepare reproducible tables; data_only skips all figure and caption exports."""
        if data_only and export_model_figures:
            raise ValueError('data_only cannot export model figures.')
        if export_model_figures and source_status not in ('local_preview', 'verified_current'):
            raise ValueError('Resolve data version or explicitly choose a local-record preview.')
        if (self.output / 'manifest.json').exists():
            base = self.output
            version = 2
            while self.output.exists():
                self.output = base.with_name(base.name + f'_v{version}')
                version += 1
        self.output.mkdir(parents=True, exist_ok=True)
        self.read_data()
        if source_status == 'verified_current':
            self.verify_current_sources()
        self.make_groups()
        self.read_physical()
        self.select_cases()
        self.export_tables()
        manifest = dict(source_status=source_status, manuscript_version_verified=source_status == 'verified_current',
            dataset=str(self.dataset_path), dataset_identity=self.split_manifest['dataset'],
            split='test', seeds=SEEDS, displayed_seed=self.seed, class_order=CLASSES,
            models=[{'key': x[0], 'label': x[1]} for x in MODELS], inputs=self.inputs,
            predictions=self.prediction_sources, js='natural-log divergence; zero terms contribute zero; no probability correction',
            aggregation='per-sample JS -> per-seed group mean -> five-seed mean and sample SD (ddof=1)',
            thresholds=self.thresholds, case_selection=self.case_rules,
            label_precision='margin and selection use frozen JSON float64 labels; JS and probability heatmaps use common exported targets (local float32-to-float64)',
            physical_sources={case: self.physical[str(i)]['metadata'] for case, i in self.selected.items()},
            unresolved=['RF and Full updated external per-sample exports and checkpoint provenance are not yet matched to current manuscript']
                if source_status != 'verified_current' else [],
            exported_model_figures=export_model_figures, data_only=data_only)
        manifest['script_sha256'] = sha256(__file__)
        (self.output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        if data_only:
            print(self.output)
            return self.output
        self.style()
        figures = []
        if export_model_figures:
            figures = [(self.ambiguity_figure(), 'results_baseline_test_ambiguity'),
                       (self.case_figure(), 'results_baseline_test_cases'),
                       (self.case_figure(('c4_c2',)), 'results_baseline_case_c4_c2'),
                       (self.case_figure(('c7_c1',)), 'results_baseline_case_c7_c1')]
            for fig, name in figures:
                self.save_figure(fig, name, source_status)
            if show:
                plt.show()
            for fig, _ in figures:
                plt.close(fig)
        else:
            fig = self.physical_preview_figure()
            self.save_figure(fig, 'physical_evidence_preparation', source_status)
            if show:
                plt.show()
            plt.close(fig)
        self.write_report(source_status, export_model_figures)
        print(self.output)
        return self.output


if __name__ == '__main__':
    # 用户已确认先使用本机完整结果生成预览；图面标注与论文新版尚待核对。
    # 在 PyCharm 中想弹窗查看，将 show 改为 True。
    # 正式版须替换 prediction_paths，并提供相符的外部选模与数据版本记录。
    analysis = Phase3TestAnalysis()
    analysis.run(export_model_figures=True, source_status='local_preview', show=False)
