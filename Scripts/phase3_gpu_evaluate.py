"""Evaluate 60 new Phase 3 neural runs, five forests and four Phase 2 references.

Run in PyCharm after training, or call evaluate(). All neural checkpoints and
the fixed 300-tree forests are sealed before loading Test examples. Existing
Phase 2 results are verified references, never relabeled as new training runs.
"""
from __future__ import annotations

import expansion_training as shared

import csv
import gc
import hashlib
import json
from pathlib import Path

from project_paths import resolve_project_path
import pickle
import tempfile
import time
import traceback

import numpy as np
import torch

import phase2_gpu_evaluate as phase2
from expansion_classical_baselines import TREE_COUNTS, _forest_probabilities, run_classical_baselines
from model_evaluate import CLASS_NAMES, compute_metrics
from phase3_models import _NO_CLASS_MASK_VARIANTS
import phase3_rf_baseline as rf
from phase3_setup import RUN_SPECS

ROOT = shared.IDEA / 'experiment_data/phase3_scratch_18000_gpu_v1'
SEEDS = phase2.SEEDS
METRICS = phase2.METRICS
SPECS = {spec.model_variant: spec for spec in RUN_SPECS
         if spec.trainer == 'torch' and spec.comparison_group in ('e3_unified_baselines', 'e5_structure_ablation')}
RF_VARIANT = 'analytic_random_forest'
REFERENCES = (*phase2.VARIANTS, phase2.PRIOR)


def check_masks(variant, samples, targets, probabilities):
    """Targets always obey type masks; unmasked ablations may predict any class."""
    phase2.check_class_masks(samples, targets, targets)
    if variant not in _NO_CLASS_MASK_VARIANTS:
        phase2.check_class_masks(samples, targets, probabilities)


def seal_json(path, payload):
    if path.exists() and shared.read(path) != payload:
        raise ValueError(f'Frozen evaluation artifact changed: {path}')
    shared.write(path, payload)
    return shared.sha256_file(path)


class Phase3Evaluation(phase2.Phase2Evaluation):
    """Reuse the verified Phase 2 FP32 inference loop without changing it."""

    def __init__(self, root=ROOT, device=None):
        super().__init__(root, device)

    def gate(self):
        expected = {(variant, seed) for variant in SPECS for seed in SEEDS}
        runs = self.protocol['neural_runs']
        if (len(SPECS) != 12 or len(runs) != 60
                or {(r['model_variant'], r['seed']) for r in runs} != expected
                or len({r['run_id'] for r in runs}) != 60):
            raise ValueError('Expected exactly 12 new neural variants times five seeds.')
        training = self.protocol['training']
        if (training['max_epochs'] != 50 or training['initialization'] != 'random'
                or training.get('early_stopping') is not False):
            raise ValueError('Expected random initialization and complete 50-epoch training.')
        logs = {}
        for spec in runs:
            canonical = SPECS[spec['model_variant']]
            if (spec['run_id'] != f"{spec['model_variant']}__seed{spec['seed']}"
                    or spec['run_key'] != canonical.run_key
                    or spec['group'] != ('baseline' if canonical.comparison_group == 'e3_unified_baselines' else 'ablation')
                    or spec['normalization_variant'] != canonical.normalization_variant):
                raise ValueError('Neural run matrix differs from the registered Phase 3 conditions.')
            path = self.root / 'neural' / spec['run_id'] / 'training.json'
            if not path.is_file():
                raise ValueError(f"Training is incomplete: {spec['run_id']}")
            log = shared.read(path)
            if log['status'] != 'completed' or [r['epoch'] for r in log['history']] != list(range(51)):
                raise ValueError(f"Training is incomplete: {spec['run_id']}")
            if log['experiment']['protocol_sha256'] != self.protocol_hash:
                raise ValueError(f"Training protocol mismatch: {spec['run_id']}")
            logs[spec['run_id']] = log
        self.split_path = resolve_project_path(self.protocol['split_manifest'])
        self.dataset_path = resolve_project_path(self.protocol['dataset_path'])
        if shared.sha256_file(self.split_path) != self.protocol['split_manifest_sha256']:
            raise ValueError('Frozen split hash mismatch.')
        self.manifest, self.split = shared.load_split_indices(self.split_path)
        if (not self.manifest.get('frozen') or self.manifest['dataset'] != self.protocol['dataset']
                or {k: len(v) for k, v in self.split.items()} != dict(train=13000, validation=2500, test=2500)):
            raise ValueError('Frozen dataset identity or partition sizes differ.')
        shared.verify_manifest_dataset(self.manifest, self.dataset_path)
        if shared.sha256_file(shared.FROZEN / 'freeze_manifest.json') != self.protocol['dataset_freeze_manifest_sha256']:
            raise ValueError('Dataset freeze manifest changed.')
        if not {Path(__file__).name, 'phase2_gpu_evaluate.py'}.issubset(self.protocol['code_sha256']):
            raise ValueError('Both evaluator source files must be frozen in the training protocol.')
        for name, digest in self.protocol['code_sha256'].items():
            if shared.sha256_file(shared.IDEA / 'scripts' / name) != digest:
                raise ValueError(f'Frozen source changed: {name}')
        normalizations = {}
        for variant, artifact in self.protocol['normalization_files'].items():
            path = resolve_project_path(artifact['path'])
            if shared.sha256_file(path) != artifact['sha256']:
                raise ValueError('Normalization hash mismatch.')
            payload = shared.read(path)
            if (payload['sample_indices'] != self.split['train'] or payload['sample_count'] != 13000
                    or payload['source'] != 'current_training_split_only'):
                raise ValueError('Normalization must use exactly the frozen Train split.')
            normalizations[variant] = payload['statistics']
        self.selection = []
        for spec in runs:
            log = logs[spec['run_id']]
            experiment = log['experiment']
            if any(experiment[key] != spec[key] for key in ('run_id', 'model_variant', 'seed')):
                raise ValueError('Training identity mismatch.')
            if experiment['initialization'] != 'random' or experiment.get('initial_checkpoint') is not None:
                raise ValueError('A new neural run loaded historical weights.')
            best = min(log['history'][1:], key=lambda r: r['validation']['loss'])
            if best['epoch'] != log['best_epoch'] or best['validation']['loss'] != log['best_validation_loss']:
                raise ValueError('Best weights must be selected only by validation NLL.')
            folder = self.root / 'neural' / spec['run_id']
            for key, name in (('best_checkpoint', 'best.pt'), ('last_checkpoint', 'last.pt'), ('initial_checkpoint', 'initial.pt')):
                artifact = log[key]
                if (resolve_project_path(artifact['path']).resolve() != (folder / name).resolve()
                        or shared.sha256_file(folder / name) != artifact['sha256']):
                    raise ValueError(f"Checkpoint path/hash mismatch: {spec['run_id']}/{name}")
            state = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
            if (state['split_sample_indices'] != self.split or state['epoch'] != best['epoch']
                    or state['experiment']['protocol_sha256'] != self.protocol_hash
                    or state['config']['model_variant'] != spec['model_variant']
                    or state['config']['random_seed'] != spec['seed']
                    or state['experiment']['model_kwargs'] != experiment['model_kwargs']
                    or state['normalization'] != normalizations[spec['normalization_variant']]):
                raise ValueError('Checkpoint identity or normalization differs from its run.')
            del state
            self.selection.append({**spec, 'best_epoch': best['epoch'], 'best_checkpoint': str(folder / 'best.pt'),
                                   'checkpoint_sha256': log['best_checkpoint']['sha256'],
                                   'training_log_sha256': shared.sha256_file(folder / 'training.json')})
        self.verify_reference()
        payload = dict(schema_version=1, protocol_sha256=self.protocol_hash,
                       evaluator_sha256=shared.sha256_file(Path(__file__)),
                       inherited_evaluator_sha256=shared.sha256_file(Path(phase2.__file__)),
                       all_neural_runs_completed=True, neural_run_count=60, test_used_for_selection=False,
                       selected_checkpoints=self.selection, reference_phase2=self.protocol['reference_phase2'])
        self.freeze_hash = seal_json(self.output / 'selected_checkpoints.json', payload)
        print('Evaluation gate passed: all 60 neural runs completed and selected weights sealed.', flush=True)

    def reference_path(self, relative):
        path = self.reference_root / relative
        key = path.resolve()
        hashes = {resolve_project_path(k).resolve(): v for k, v in self.protocol['reference_phase2']['artifacts_sha256'].items()}
        if key not in hashes or shared.sha256_file(path) != hashes[key]:
            raise ValueError(f'Reference artifact missing from snapshot or changed: {relative}')
        return path

    def verify_reference(self):
        reference = self.protocol['reference_phase2']
        self.reference_root = resolve_project_path(reference['path']).resolve()
        for path, digest in reference['artifacts_sha256'].items():
            if shared.sha256_file(resolve_project_path(path)) != digest:
                raise ValueError(f'Phase 2 reference changed: {path}')
        protocol_path = self.reference_path('training_protocol.json')
        if shared.sha256_file(protocol_path) != reference['protocol_sha256']:
            raise ValueError('Reference protocol hash mismatch.')
        protocol = shared.read(protocol_path)
        if (protocol['dataset'] != self.protocol['dataset']
                or protocol['split_manifest_sha256'] != self.protocol['split_manifest_sha256']
                or protocol['training']['max_epochs'] != 50 or protocol['training']['initialization'] != 'random'):
            raise ValueError('Reference models must use the same frozen data and scratch-training regime.')
        for name, frozen_hash in protocol['code_sha256'].items():
            source = shared.IDEA / 'scripts' / name
            current_hash = shared.sha256_file(source)
            if current_hash == frozen_hash:
                continue
            change = reference.get('source_changes_after_reference', {}).get(name, {})
            if (name != 'phase2_gpu_figures.py' or change.get('frozen_sha256') != frozen_hash
                    or change.get('current_sha256') != current_hash):
                raise ValueError(f'Reference source changed outside the verified plotting exception: {name}')
            restored = source.read_bytes().replace(b'# import matplotlib', b'import matplotlib', 1)
            restored = restored.replace(b"# matplotlib.use('Agg')", b"matplotlib.use('Agg')", 1)
            if hashlib.sha256(restored).hexdigest() != frozen_hash:
                raise ValueError('The reference plotting change is not exactly the declared Agg-backend edit.')
        for variant in phase2.VARIANTS:
            for seed in SEEDS:
                run_id = f'{variant}__seed{seed}'
                log = shared.read(self.reference_path(f'neural/{run_id}/training.json'))
                if (log['status'] != 'completed' or [r['epoch'] for r in log['history']] != list(range(51))
                        or log['experiment']['protocol_sha256'] != reference['protocol_sha256']):
                    raise ValueError('Reference training must already be complete under its frozen protocol.')
                checkpoint = self.reference_path(f'neural/{run_id}/best.pt')
                if shared.sha256_file(checkpoint) != log['best_checkpoint']['sha256']:
                    raise ValueError('Reference selected weight differs from its log.')
        self.reference_path('evaluation/summary.json')
        self.reference_path('evaluation/selected_checkpoints.json')
        self.reference_path(f'evaluation/{phase2.PRIOR}/prior.json')

    def load_train_validation(self):
        cache = resolve_project_path(self.protocol['raw_input_cache']['path'])
        meta = shared.read(cache.with_suffix('.json'))
        self.cache_identity = dict(dataset_sha256=self.protocol['dataset']['sha256'],
                                  split_sha256=self.protocol['split_manifest_sha256'],
                                  code_sha256={name: shared.sha256_file(shared.IDEA / 'scripts' / name)
                                               for name in ('data_processing.py', 'analytic_layer.py', 'orbit_core.py')})
        if (meta['identity'] != self.cache_identity or shared.sha256_file(cache) != meta['sha256']
                or meta['sha256'] != self.protocol['raw_input_cache']['sha256']):
            raise ValueError('Train/Validation input cache provenance mismatch.')
        with cache.open('rb') as stream:
            samples = pickle.load(stream)
        expected = set(self.split['train'] + self.split['validation'])
        if len(samples) != 15500 or {int(s['sample_index']) for s in samples} != expected:
            raise ValueError('Prepared cache must contain exactly Train and Validation.')
        train = shared.positions_for_sample_indices(samples, self.split['train'])
        validation = shared.positions_for_sample_indices(samples, self.split['validation'])
        shared.write(self.output / 'status.json', dict(status='fitting_random_forests', test_examples_loaded=False,
                     fixed_tree_count=300, seeds=list(SEEDS), updated_at=shared.now()))
        self.classical = run_classical_baselines(samples, train, validation, self.manifest, self.root,
                         dict(experiment_id=self.protocol['experiment_id'], protocol_sha256=self.protocol_hash,
                              dataset_sha256=self.protocol['dataset']['sha256'],
                              split_manifest_sha256=self.protocol['split_manifest_sha256']), seeds=SEEDS)
        root = resolve_project_path(self.classical['output_dir'])
        reference_prior = shared.read(self.reference_path(f'evaluation/{phase2.PRIOR}/prior.json'))
        fitted_prior = shared.read(root / 'counterpart_type_prior/metadata.json')
        if reference_prior['probabilities'] != fitted_prior['probabilities']:
            raise ValueError('The helper prior does not reproduce the frozen Train-only reference prior.')
        forest_runs = [r for r in self.classical['runs'] if r['seed'] is not None]
        if sorted(r['seed'] for r in forest_runs) != list(SEEDS):
            raise ValueError('All five RandomForest seeds must finish before loading Test.')
        self.forest_selection = []
        for row in forest_runs:
            path = resolve_project_path(row['output_dir']) / 'model.pkl'
            model = rf.load(path)
            if model.n_estimators != 300 or len(model.estimators_) != 300:
                raise ValueError('RandomForest selection must use the fixed 300-tree model.')
            self.forest_selection.append(dict(run_id=f"{RF_VARIANT}__seed{row['seed']}", model_variant=RF_VARIANT,
                        run_key='b1_analytic_random_forest', group='baseline', trainer='random_forest',
                        seed=row['seed'], best_checkpoint=str(path), checkpoint_sha256=shared.sha256_file(path)))
            del model
        self.forest_freeze_hash = seal_json(self.output / 'selected_random_forests.json',
                    dict(protocol_sha256=self.protocol_hash, selected_forests=self.forest_selection,
                         selection='fixed_300_trees_without_validation_or_test_selection',
                         manifest_path=str(root / 'manifest.json'), manifest_sha256=shared.sha256_file(root / 'manifest.json'),
                         artifact_sha256=self.classical['artifact_sha256'],
                         helper_prior='verified against Phase 2; final summary reuses the Phase 2 reference'))
        self.write_forest_curves(forest_runs)
        self.samples = {'validation': [samples[p] for p in validation]}
        del samples
        gc.collect()

    def write_forest_curves(self, runs):
        histories = [shared.read(resolve_project_path(run['output_dir']) / 'history.json') for run in runs]
        if any([r['tree_count'] for r in history['history']] != list(TREE_COUNTS) for history in histories):
            raise ValueError('Incomplete RandomForest tree-count histories.')
        partitions = {}
        for partition in ('train', 'validation'):
            partitions[partition] = dict(metrics={key: {
                'mean': [phase2.mean_std([history['history'][i][partition][source] for history in histories])['mean']
                         for i in range(len(TREE_COUNTS))],
                'std': [phase2.mean_std([history['history'][i][partition][source] for history in histories])['std']
                        for i in range(len(TREE_COUNTS))]} for key, source in METRICS.items()})
        shared.write(self.output / RF_VARIANT / 'tree_curve_summary.json',
                     dict(schema_version=1, protocol_sha256=self.protocol_hash, std_ddof=1, n_runs=5,
                          seeds=sorted(r['seed'] for r in runs), step_axis='tree_count',
                          tree_counts=list(TREE_COUNTS), partitions=partitions,
                          note='Actual staged fitting of 50,100,...,300 trees; not neural epochs.'))

    def load_test(self):
        if not hasattr(self, 'freeze_hash') or not hasattr(self, 'forest_freeze_hash'):
            raise RuntimeError('Neural weights and all five forests must be sealed before Test loading.')
        cache = self.output / 'cache/test_inputs.pkl'
        metadata_path = cache.with_suffix('.json')
        identity = dict(identity=self.cache_identity, selection_freeze_sha256=self.freeze_hash,
                        random_forest_freeze_sha256=self.forest_freeze_hash)
        if cache.exists() and metadata_path.exists():
            metadata = shared.read(metadata_path)
            if any(metadata.get(k) != v for k, v in identity.items()) or shared.sha256_file(cache) != metadata['sha256']:
                raise ValueError('Test cache provenance mismatch.')
            with cache.open('rb') as stream:
                samples = pickle.load(stream)
        else:
            samples = list(shared.iter_training_data(self.dataset_path, sample_indices=set(self.split['test']),
                                                     allow_sidecar_labels=False))
            cache.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache.with_suffix('.pkl.tmp')
            with temporary.open('wb') as stream:
                pickle.dump(samples, stream, protocol=pickle.HIGHEST_PROTOCOL)
            temporary.replace(cache)
            shared.write(metadata_path, dict(**identity, sha256=shared.sha256_file(cache), sample_count=len(samples)))
        if len(samples) != 2500 or {int(s['sample_index']) for s in samples} != set(self.split['test']):
            raise ValueError('Test sample identities differ from the frozen split.')
        self.samples['test'] = [samples[p] for p in shared.positions_for_sample_indices(samples, self.split['test'])]
        self.targets = {name: np.stack([s['y'] for s in rows]).astype(np.float64) for name, rows in self.samples.items()}
        self.sample_indices = {name: np.asarray([s['sample_index'] for s in rows], dtype=np.int64)
                               for name, rows in self.samples.items()}

    def verify_predictions(self, result, folder, partition):
        for name, digest in result['prediction_artifacts'].items():
            if shared.sha256_file(folder / name) != digest:
                raise ValueError(f'Preserved prediction artifact changed: {folder / name}')
        with np.load(folder / 'predictions.npz', allow_pickle=False) as data:
            if (not np.array_equal(data['sample_indices'], self.sample_indices[partition])
                    or not np.array_equal(data['targets'], self.targets[partition])):
                raise ValueError('Preserved prediction samples differ from the frozen partition.')
            check_masks(result['model_variant'], self.samples[partition], data['targets'], data['probabilities'])
            metrics = compute_metrics(data['targets'], data['probabilities'])
        if (result['metrics'] != {key: metrics[source] for key, source in METRICS.items()}
                or result['confusion_matrix'] != metrics['confusion_matrix']
                or result['per_class'] != metrics['per_class'] or result['sample_count'] != metrics['sample_count']):
            raise ValueError('Preserved metrics do not reproduce their probability arrays.')

    def read_completed(self, spec, partition, model_hash):
        path = self.output / 'runs' / spec['run_id'] / partition / 'metrics.json'
        if not path.exists():
            return None
        result = shared.read(path)
        expected = dict(status='completed', protocol_sha256=self.protocol_hash, selection_freeze_sha256=self.freeze_hash,
                        checkpoint_sha256=model_hash, partition=partition,
                        **{key: spec[key] for key in ('run_id', 'model_variant', 'seed')})
        if any(result.get(k) != v for k, v in expected.items()):
            raise ValueError('Completed evaluation identity mismatch.')
        self.verify_predictions(result, path.parent, partition)
        return result

    def save_result(self, spec, partition, model_hash, probabilities, seconds):
        targets = self.targets[partition]
        probabilities = np.asarray(probabilities, dtype=np.float64)
        check_masks(spec['model_variant'], self.samples[partition], targets, probabilities)
        metrics = compute_metrics(targets, probabilities)
        folder = self.output / 'runs' / spec['run_id'] / partition
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / 'predictions.npz'
        temporary = path.with_suffix('.npz.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, sample_indices=self.sample_indices[partition], targets=targets,
                                probabilities=probabilities, reference_classes=targets.argmax(axis=1),
                                predicted_classes=probabilities.argmax(axis=1))
        temporary.replace(path)
        names = ['predictions.npz', 'predictions.csv', 'confusion_matrix.csv', 'confusion_matrix_row_normalized.csv']
        with (folder / names[1]).open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.writer(stream)
            writer.writerow(['sample_index', 'counterpart_type', 'reference_class', 'predicted_class']
                            + [f'y_C{i}' for i in range(8)] + [f'p_C{i}' for i in range(8)])
            for i, (sample, target, predicted) in enumerate(zip(self.samples[partition], targets, probabilities)):
                writer.writerow([int(self.sample_indices[partition][i]), phase2.type_name(sample), int(target.argmax()),
                                 int(predicted.argmax()), *target.tolist(), *predicted.tolist()])
        counts = np.asarray(metrics['confusion_matrix'], dtype=np.int64)
        normalized = phase2.row_normalized(counts)
        for name, matrix in ((names[2], counts), (names[3], normalized)):
            with (folder / name).open('w', newline='', encoding='utf-8-sig') as stream:
                writer = csv.writer(stream)
                writer.writerow(['reference/predicted', *CLASS_NAMES])
                writer.writerows([[CLASS_NAMES[i], *row.tolist()] for i, row in enumerate(matrix)])
        forest = spec['model_variant'] == RF_VARIANT
        result = dict(schema_version=1, status='completed', protocol_sha256=self.protocol_hash,
                      selection_freeze_sha256=self.freeze_hash, checkpoint_sha256=model_hash,
                      **{key: spec[key] for key in ('run_id', 'model_variant', 'seed', 'run_key', 'group')},
                      partition=partition, best_epoch=spec.get('best_epoch'), sample_count=len(targets),
                      target_source=phase2.TARGET_SOURCE, class_mask_applied=spec['model_variant'] not in _NO_CLASS_MASK_VARIANTS,
                      metrics={key: metrics[source] for key, source in METRICS.items()}, per_class=metrics['per_class'],
                      confusion_matrix=counts.tolist(), confusion_matrix_row_normalized=normalized.tolist(),
                      class_support=counts.sum(axis=1).tolist(),
                      prediction_artifacts={name: shared.sha256_file(folder / name) for name in names},
                      inference_seconds=seconds, inference_precision='RandomForest probabilities' if forest else 'FP32 without autocast',
                      inference_batch_size=None if forest else phase2.BATCH_SIZE,
                      source='new_random_forest_fit' if forest else 'new_neural_training', completed_at=shared.now())
        shared.write(folder / 'metrics.json', result)
        shared.write(self.output / 'status.json', dict(status='evaluating', run_id=spec['run_id'], partition=partition,
                     completed_partitions=len(self.records) + 1, expected_new_partitions=130, updated_at=shared.now()))
        return result

    def forest_evaluation(self):
        features = {name: rf.extract_feature_matrix(rows) for name, rows in self.samples.items()}
        masks = {name: np.stack([rf._counterpart_class_mask(s) for s in rows]) for name, rows in self.samples.items()}
        for spec in self.forest_selection:
            if shared.sha256_file(resolve_project_path(spec['best_checkpoint'])) != spec['checkpoint_sha256']:
                raise ValueError('A forest changed after its selection was sealed.')
            model = rf.load(resolve_project_path(spec['best_checkpoint']))
            for name in self.samples:
                result = self.read_completed(spec, name, spec['checkpoint_sha256'])
                if result is None:
                    started = time.perf_counter()
                    probabilities = _forest_probabilities(model, features[name], masks[name])
                    result = self.save_result(spec, name, spec['checkpoint_sha256'], probabilities, time.perf_counter() - started)
                self.records.append(result)
            del model

    def merge_references(self, summary):
        reference_path = self.reference_path('evaluation/summary.json')
        reference = shared.read(reference_path)
        protocol_hash = self.protocol['reference_phase2']['protocol_sha256']
        if (reference['status'] != 'completed' or reference['protocol_sha256'] != protocol_hash
                or reference['std_ddof'] != 1 or not reference['all_models_selected_without_test']):
            raise ValueError('Reference evaluation is incomplete or uses a different protocol.')
        rows = []
        for name in ('validation', 'test'):
            for variant in REFERENCES:
                seeds = [None] if variant == phase2.PRIOR else SEEDS
                for seed in seeds:
                    run_id = variant if seed is None else f'{variant}__seed{seed}'
                    path = self.reference_path(f'evaluation/runs/{run_id}/{name}/metrics.json')
                    result = shared.read(path)
                    if (result['status'] != 'completed' or result['protocol_sha256'] != protocol_hash
                            or result['model_variant'] != variant or result['seed'] != seed or result['partition'] != name):
                        raise ValueError('Reference per-run identity mismatch.')
                    self.verify_predictions(result, path.parent, name)
                    rows.append(result)
        checked = phase2.aggregate_results(rows, protocol_hash)
        for name in ('validation', 'test'):
            if checked['partitions'][name]['models'] != reference['partitions'][name]['models']:
                raise ValueError('Reference summary does not reproduce the frozen per-seed results.')
            support = next(iter(summary['partitions'][name]['models'].values()))['class_support']
            for variant, model in checked['partitions'][name]['models'].items():
                if model['class_support'] != support:
                    raise ValueError('Reference and new experiments use different evaluation samples.')
                summary['partitions'][name]['models'][variant] = {**model, 'source': 'reference_phase2',
                          'source_protocol_sha256': protocol_hash, 'source_summary_path': str(reference_path.relative_to(shared.IDEA)),
                          'source_summary_sha256': shared.sha256_file(reference_path)}
        summary['partitions']['train'] = reference['partitions']['train']
        summary['partitions']['train']['models'][phase2.PRIOR] = {
            **reference['partitions']['train']['models'][phase2.PRIOR], 'source': 'reference_phase2',
            'source_protocol_sha256': protocol_hash, 'source_summary_path': str(reference_path.relative_to(shared.IDEA)),
            'source_summary_sha256': shared.sha256_file(reference_path)}
        summary['reference_phase2'] = {**self.protocol['reference_phase2'], 'reference_models': list(REFERENCES),
                                      'new_training_runs': False, 'predictions_verified_against_current_partitions': True}

    def write_summary(self, summary):
        shared.write(self.output / 'summary.json', summary)
        models = summary['partitions']['test']['models']
        for variant in models:
            shared.write(self.output / variant / 'confusion_summary.json',
                         {name: value['models'][variant] for name, value in summary['partitions'].items()
                          if variant in value['models']})
        with (self.output / 'metrics_mean_std.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.writer(stream)
            writer.writerow(['partition', 'model_variant', 'source', 'n_runs', 'metric', 'mean', 'std', 'std_ddof'])
            for name, partition in summary['partitions'].items():
                for variant, model in partition['models'].items():
                    for metric, value in model['metrics'].items():
                        writer.writerow([name, variant, model['source'], model['n_runs'], metric, value['mean'], value['std'], 1])
        lines = ['# Phase 3 selected-checkpoint results', '',
                 'Five independent seeds: mean ± sample SD (ddof=1). The fixed prior has no seed SD.', '',
                 'Reference Phase 2 models are reused frozen results, not new training runs.', '']
        for name in ('validation', 'test'):
            lines += [f'## {name.title()}', '', '| Model | Source | ' + ' | '.join(METRICS) + ' |',
                      '|---|---|' + '---|' * len(METRICS)]
            for variant, model in summary['partitions'][name]['models'].items():
                values = [f"{v['mean']:.6f}" + (f" ± {v['std']:.6f}" if v['std'] is not None else ' (fixed)')
                          for v in model['metrics'].values()]
                lines.append('| ' + variant + ' | ' + model['source'] + ' | ' + ' | '.join(values) + ' |')
            lines.append('')
        (self.output / 'metrics_mean_std.md').write_text('\n'.join(lines), encoding='utf-8')

    def run(self):
        self.gate()
        self.load_train_validation()
        self.load_test()
        self.neural()
        self.forest_evaluation()
        if len(self.records) != 130:
            raise ValueError('Expected 60 neural runs plus 5 forests on both Validation and Test.')
        summary = phase2.aggregate_results(self.records, self.protocol_hash)
        for partition in summary['partitions'].values():
            for variant, model in partition['models'].items():
                spec = SPECS.get(variant)
                model.update(source='new_random_forest_fit' if variant == RF_VARIANT else 'new_neural_training',
                             run_key='b1_analytic_random_forest' if spec is None else spec.run_key,
                             group='baseline' if spec is None or spec.comparison_group == 'e3_unified_baselines' else 'ablation',
                             comparison_group='e3_unified_baselines' if spec is None else spec.comparison_group,
                             class_mask_applied=variant not in _NO_CLASS_MASK_VARIANTS)
        self.merge_references(summary)
        expected = {*SPECS, RF_VARIANT, *REFERENCES}
        for name in ('validation', 'test'):
            if set(summary['partitions'][name]['models']) != expected or len(expected) != 17:
                raise ValueError('Final comparison requires all 17 requested model conditions.')
        summary.update(completed_at=shared.now(), selection_freeze_sha256=self.freeze_hash,
                       random_forest_freeze_sha256=self.forest_freeze_hash, new_neural_run_count=60,
                       new_random_forest_run_count=5, referenced_model_count=4, new_evaluation_partitions=130,
                       all_models_selected_without_test=True,
                       neural_inference=dict(batch_size=phase2.BATCH_SIZE, precision='FP32', autocast=False,
                                             device=self.inference_device, cuda_backend=self.protocol['training']['backend'],
                                             training_curve_precision=self.protocol['training']['amp'],
                                             training_batch_size=self.protocol['training']['batch_size']))
        self.write_summary(summary)
        shared.write(self.output / 'status.json', dict(status='completed', completed_new_partitions=130,
                     total_model_conditions=17, test_evaluation_performed=True, updated_at=shared.now()))
        return summary


def self_check():
    """CPU synthetic metrics, seed aggregation and both mask policies; no data read."""
    targets = np.eye(8, dtype=np.float64)
    samples = [dict(counterpart_type=np.array([1, 0]), y=targets[0]),
               dict(counterpart_type=np.array([0, 1]), y=targets[6])]
    y = np.stack([targets[0], targets[6]])
    unmasked = np.stack([targets[6], targets[2]])
    check_masks('a0_factual_only_no_mask', samples, y, unmasked)
    try:
        check_masks('bilstm_attention', samples, y, unmasked)
    except ValueError:
        pass
    else:
        raise AssertionError('Masked models must reject inapplicable predicted classes.')
    try:
        check_masks('a0_factual_only_no_mask', samples, unmasked, unmasked)
    except ValueError:
        pass
    else:
        raise AssertionError('Unmasked models must still reject invalid target masks.')
    raw = compute_metrics(targets, targets)
    rows = [dict(model_variant='bilstm_attention', partition='test', run_id=f'bilstm_attention__seed{seed}', seed=seed,
                 metrics={**{key: raw[source] for key, source in METRICS.items()}, 'loss': float(i)},
                 confusion_matrix=raw['confusion_matrix'], per_class=raw['per_class']) for i, seed in enumerate(SEEDS)]
    stats = phase2.aggregate_results(rows, 'synthetic')['partitions']['test']['models']['bilstm_attention']
    assert stats['metrics']['loss']['mean'] == 2
    assert np.isclose(stats['metrics']['loss']['std'], np.sqrt(2.5))
    try:
        phase2.aggregate_results(rows[:-1] + [rows[0]], 'synthetic')
    except ValueError:
        pass
    else:
        raise AssertionError('Duplicate seeds must fail.')
    qa = ROOT / 'qa'
    qa.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix='evaluation_fixture_', dir=qa))
    protocol = dict(training=dict(max_epochs=50, initialization='random', early_stopping=False),
                    neural_runs=[dict(run_id=f'{variant}__seed{seed}', model_variant=variant, seed=seed,
                                      run_key=spec.run_key,
                                      group='baseline' if spec.comparison_group == 'e3_unified_baselines' else 'ablation',
                                      normalization_variant=spec.normalization_variant)
                                 for variant, spec in SPECS.items() for seed in SEEDS])
    shared.write(folder / 'training_protocol.json', protocol)
    evaluator = Phase3Evaluation(folder, device='cpu')
    try:
        evaluator.gate()
    except ValueError as error:
        assert 'incomplete' in str(error)
    else:
        raise AssertionError('Incomplete training must block access to Test.')
    assert not hasattr(evaluator, 'dataset_path')
    evaluator.freeze_hash = 'synthetic_selection'
    try:
        evaluator.load_test()
    except RuntimeError:
        pass
    else:
        raise AssertionError('Missing RF selection must block Test loading.')
    evaluator.samples = dict(validation=samples)
    evaluator.targets = dict(validation=y)
    evaluator.sample_indices = dict(validation=np.array([100, 200]))
    spec = dict(run_id='a0_factual_only_no_mask__seed7', model_variant='a0_factual_only_no_mask', seed=7,
                run_key='a0_factual_only_no_mask', group='ablation')
    result = evaluator.save_result(spec, 'validation', 'synthetic_weights', unmasked, 0.0)
    assert evaluator.read_completed(spec, 'validation', 'synthetic_weights') == result
    assert not result['class_mask_applied'] and result['confusion_matrix'][0][6] == 1
    report = dict(status='passed', formal_data_read=False, fixture_only=True, fixture_directory=str(folder),
                  checks=['60-run incomplete gate', 'RF freeze before Test', 'sample SD ddof=1',
                          'duplicate-seed rejection', 'unmasked predictions allowed', 'target masks always enforced',
                          'masked predictions enforced', 'unmasked NPZ/CSV/JSON round trip'])
    shared.write(qa / 'evaluation_self_check.json', report)
    return report


def evaluate(root=ROOT):
    import msvcrt
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'evaluation.lock').open('a+b') as lock:
        if lock.tell() == 0:
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            return Phase3Evaluation(root).run()
        except Exception:
            shared.write(root / 'evaluation/failure.json', dict(status='failed', time=shared.now(), traceback=traceback.format_exc()))
            raise
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == '__main__':
    evaluate()
