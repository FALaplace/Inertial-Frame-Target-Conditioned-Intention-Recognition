"""Evaluate the four Phase 2 models after all 15 scratch runs finish.

Run directly in PyCharm or call evaluate() from the training queue. Selection
uses validation NLL only and is sealed before any Test examples are loaded.
"""
from __future__ import annotations

import expansion_training as shared  # Set deterministic runtime flags first.

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
from torch.utils.data import DataLoader

from model_evaluate import CLASS_NAMES, compute_metrics
from phase2_experiment import _prior_probabilities

ROOT = shared.IDEA / 'experiment_data/phase2_scratch_18000_gpu_v1'
SEEDS = (7, 17, 42, 73, 101)
VARIANTS = ('factual_only', 'full_without_analytic', 'full')
PRIOR = 'counterpart_type_prior'
BATCH_SIZE = 32
METRICS = dict(loss='soft_nll', js='js_divergence', brier='brier_score_sum_over_classes',
               l1='l1_distance', agreement='dominant_label_agreement',
               balanced_accuracy='balanced_accuracy', macro_f1='macro_f1_all_8_classes')
TARGET_SOURCE = 'embedded_soft_pseudo_label'


def mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError('Aggregation requires finite, nonempty seed values.')
    return dict(mean=values.mean(axis=0).tolist(),
                std=values.std(axis=0, ddof=1).tolist() if len(values) > 1 else None)


def row_normalized(counts):
    counts = np.asarray(counts, dtype=np.float64)
    support = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, support, out=np.zeros_like(counts), where=support > 0)


def type_name(sample):
    value = np.asarray(sample['counterpart_type'])
    if value.shape != (2,) or not np.array_equal(np.sort(value), [0, 1]):
        raise ValueError('Counterpart type must be a two-element one-hot vector.')
    return 'satellite' if int(value.argmax()) == 0 else 'ground_region'


def prior_predictions(priors, samples):
    return np.asarray([priors[type_name(sample)] for sample in samples], dtype=np.float64)


def check_class_masks(samples, targets, probabilities):
    masks = np.ones_like(targets, dtype=bool)
    for index, sample in enumerate(samples):
        masks[index, (6, 7) if type_name(sample) == 'satellite' else (2, 3, 4)] = False
    if np.any(targets[~masks] != 0) or np.any(probabilities[~masks] != 0):
        raise ValueError('A target or prediction violates the counterpart class mask.')


def aggregate_results(results, protocol_hash):
    partitions = {}
    for partition, variant in dict.fromkeys((r['partition'], r['model_variant']) for r in results):
        rows = [r for r in results if (r['partition'], r['model_variant']) == (partition, variant)]
        rows.sort(key=lambda r: -1 if r['seed'] is None else r['seed'])
        seeds = [row['seed'] for row in rows]
        expected = [None] if variant == PRIOR else list(SEEDS)
        if seeds != expected or len({r['run_id'] for r in rows}) != len(rows):
            raise ValueError(f'Incomplete or duplicate seeds: {variant}/{partition}.')
        counts = [np.asarray(row['confusion_matrix'], dtype=np.int64) for row in rows]
        support = counts[0].sum(axis=1)
        if not all(np.array_equal(matrix.sum(axis=1), support) for matrix in counts):
            raise ValueError('Seed predictions must refer to the same partition support.')
        count_stats = mean_std(counts)
        normalized_stats = mean_std([row_normalized(matrix) for matrix in counts])
        model = dict(n_runs=len(rows), seeds=seeds, run_ids=[r['run_id'] for r in rows],
                     sample_count=int(support.sum()), class_support=support.tolist(),
                     metrics={key: mean_std([r['metrics'][key] for r in rows]) for key in METRICS},
                     confusion_matrix_mean=count_stats['mean'], confusion_matrix_std=count_stats['std'],
                     confusion_matrix_row_normalized_mean=normalized_stats['mean'],
                     confusion_matrix_row_normalized_std=normalized_stats['std'],
                     per_class={name: {**{key: mean_std([r['per_class'][name][key] for r in rows])
                                              for key in ('precision', 'recall', 'f1')},
                                       'support': int(support[i])} for i, name in enumerate(CLASS_NAMES)})
        partitions.setdefault(partition, dict(models={}))['models'][variant] = model
    return dict(schema_version=1, status='completed', protocol_sha256=protocol_hash, std_ddof=1,
                aggregation_unit='independent_training_seed', target_source=TARGET_SOURCE,
                reference_class_definition='argmax of embedded soft pseudo label',
                confusion_matrix_axes=dict(rows='reference_pseudo_class', columns='predicted_class'),
                confusion_matrix_aggregation='elementwise mean and sample SD of five seed matrices; not an ensemble',
                metric_definitions=dict(logarithm='natural', brier='sum over 8 classes then sample mean',
                                        macro_f1='unweighted mean over all 8 classes',
                                        balanced_accuracy='mean recall over observed reference classes'),
                deterministic_prior_std=None, partitions=partitions)


class Phase2Evaluation:
    def __init__(self, root=ROOT, device=None):
        self.root = Path(root).resolve()
        self.output = self.root / 'evaluation'
        self.protocol_path = self.root / 'training_protocol.json'
        self.protocol = shared.read(self.protocol_path)
        self.protocol_hash = shared.sha256_file(self.protocol_path)
        self.device_name = device
        self.records = []

    def gate(self):
        """Check completion before opening dataset metadata or any Test example."""
        expected = {(variant, seed) for variant in VARIANTS for seed in SEEDS}
        runs = self.protocol['neural_runs']
        if (len(runs) != 15 or {(r['model_variant'], r['seed']) for r in runs} != expected
                or len({r['run_id'] for r in runs}) != 15):
            raise ValueError('Expected exactly three neural variants times five seeds.')
        training = self.protocol['training']
        if (training['max_epochs'] != 50 or training['initialization'] != 'random'
                or training.get('early_stopping') is not False):
            raise ValueError('Expected random initialization and all 50 epochs.')
        logs = {}
        for spec in runs:
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
        # Hashing verifies bytes without extracting/evaluating Test examples.
        shared.verify_manifest_dataset(self.manifest, self.dataset_path)
        if shared.sha256_file(shared.FROZEN / 'freeze_manifest.json') != self.protocol['dataset_freeze_manifest_sha256']:
            raise ValueError('Dataset freeze manifest changed.')
        for name, digest in self.protocol['code_sha256'].items():
            if shared.sha256_file(shared.IDEA / 'scripts' / name) != digest:
                raise ValueError(f'Frozen source changed: {name}')
        normalizations = {}
        for variant, artifact in self.protocol['normalization_files'].items():
            path = resolve_project_path(artifact['path'])
            if shared.sha256_file(path) != artifact['sha256']:
                raise ValueError(f'Normalization changed: {variant}')
            payload = shared.read(path)
            if (payload['sample_indices'] != self.split['train'] or payload['sample_count'] != 13000
                    or payload['source'] != 'current_training_split_only'):
                raise ValueError('Normalization must use exactly the frozen Train samples.')
            normalizations[variant] = payload['statistics']
        self.selection = []
        for spec in runs:
            log = logs[spec['run_id']]
            if any(log['experiment'][key] != spec[key] for key in ('run_id', 'model_variant', 'seed')):
                raise ValueError(f"Run identity mismatch: {spec['run_id']}")
            if log['experiment']['initialization'] != 'random' or log['experiment'].get('initial_checkpoint') is not None:
                raise ValueError('A formal run loaded historical model weights.')
            best = min(log['history'][1:], key=lambda row: row['validation']['loss'])
            if best['epoch'] != log['best_epoch'] or best['validation']['loss'] != log['best_validation_loss']:
                raise ValueError('Best checkpoint must be selected solely by validation NLL.')
            checkpoint = resolve_project_path(log['best_checkpoint']['path'])
            expected_path = self.root / 'neural' / spec['run_id'] / 'best.pt'
            if checkpoint.resolve() != expected_path.resolve():
                raise ValueError('Selected checkpoint is outside its formal run directory.')
            digest = shared.sha256_file(checkpoint)
            if digest != log['best_checkpoint']['sha256']:
                raise ValueError('Selected weight hash mismatch.')
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if (state['split_sample_indices'] != self.split or state['epoch'] != best['epoch']
                    or state['experiment']['protocol_sha256'] != self.protocol_hash
                    or state['config']['model_variant'] != spec['model_variant']
                    or state['config']['random_seed'] != spec['seed']
                    or state['normalization'] != normalizations[spec.get('normalization_variant', spec['model_variant'])]):
                raise ValueError('Checkpoint metadata/normalization mismatch.')
            del state
            self.selection.append(dict(run_id=spec['run_id'], model_variant=spec['model_variant'], seed=spec['seed'],
                                       best_epoch=best['epoch'], best_checkpoint=str(checkpoint), checkpoint_sha256=digest,
                                       training_log_sha256=shared.sha256_file(expected_path.parent / 'training.json')))
        freeze = dict(schema_version=1, protocol_sha256=self.protocol_hash,
                      evaluator_sha256=shared.sha256_file(Path(__file__)), test_used_for_selection=False,
                      all_neural_runs_completed=True, neural_run_count=15, selected_checkpoints=self.selection)
        path = self.output / 'selected_checkpoints.json'
        if path.exists() and shared.read(path) != freeze:
            raise ValueError('Selected checkpoints changed after the evaluation freeze.')
        shared.write(path, freeze)
        self.freeze_hash = shared.sha256_file(path)
        print('Evaluation gate passed: 15 completed runs and selected weights sealed.', flush=True)

    def load_train_validation(self):
        cache = resolve_project_path(self.protocol['raw_input_cache']['path'])
        meta = shared.read(cache.with_suffix('.json'))
        self.cache_identity = dict(dataset_sha256=self.protocol['dataset']['sha256'],
                                  split_sha256=self.protocol['split_manifest_sha256'],
                                  code_sha256={name: shared.sha256_file(shared.IDEA / 'scripts' / name)
                                               for name in ('data_processing.py', 'analytic_layer.py', 'orbit_core.py')})
        if (meta['identity'] != self.cache_identity or shared.sha256_file(cache) != meta['sha256']
                or meta['sha256'] != self.protocol['raw_input_cache']['sha256']):
            raise ValueError('Prepared Train/Validation cache provenance mismatch.')
        with cache.open('rb') as stream:
            samples = pickle.load(stream)
        expected = set(self.split['train'] + self.split['validation'])
        if len(samples) != 15500 or {int(s['sample_index']) for s in samples} != expected:
            raise ValueError('Cache must contain exactly Train and Validation samples.')
        self.samples = {name: [samples[p] for p in shared.positions_for_sample_indices(samples, self.split[name])]
                        for name in ('train', 'validation')}
        priors, counts = _prior_probabilities(self.samples['train'])
        prior = dict(schema_version=1, model_variant=PRIOR, protocol_sha256=self.protocol_hash,
                     fit_split='train', train_sample_indices=self.split['train'], counts=counts,
                     probabilities=priors, validation_used_for_fitting=False, test_used_for_fitting=False)
        path = self.output / PRIOR / 'prior.json'
        if path.exists() and shared.read(path) != prior:
            raise ValueError('The frozen Train-only prior changed.')
        shared.write(path, prior)
        self.prior_hash, self.priors = shared.sha256_file(path), priors
        del samples
        gc.collect()

    def load_test(self):
        if not hasattr(self, 'freeze_hash') or not hasattr(self, 'prior_hash'):
            raise RuntimeError('Neural selection and Train-only prior must be sealed before Test loading.')
        cache = self.output / 'cache/test_inputs.pkl'
        meta_path = cache.with_suffix('.json')
        if cache.exists() and meta_path.exists():
            meta = shared.read(meta_path)
            if (meta['identity'] != self.cache_identity or shared.sha256_file(cache) != meta['sha256']
                    or meta['selection_freeze_sha256'] != self.freeze_hash):
                raise ValueError('Prepared Test cache provenance mismatch.')
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
            shared.write(meta_path, dict(identity=self.cache_identity, sha256=shared.sha256_file(cache),
                                        sample_count=len(samples), selection_freeze_sha256=self.freeze_hash))
        if len(samples) != 2500 or {int(s['sample_index']) for s in samples} != set(self.split['test']):
            raise ValueError('Test sample identities differ from the frozen partition.')
        self.samples['test'] = [samples[p] for p in shared.positions_for_sample_indices(samples, self.split['test'])]
        self.targets = {name: np.stack([s['y'] for s in rows]).astype(np.float64) for name, rows in self.samples.items()}
        self.sample_indices = {name: np.asarray([s['sample_index'] for s in rows], dtype=np.int64)
                               for name, rows in self.samples.items()}

    def read_completed(self, spec, partition, model_hash):
        path = self.output / 'runs' / spec['run_id'] / partition / 'metrics.json'
        if not path.exists():
            return None
        result = shared.read(path)
        expected = dict(status='completed', protocol_sha256=self.protocol_hash, selection_freeze_sha256=self.freeze_hash,
                        checkpoint_sha256=model_hash, partition=partition,
                        **{key: spec[key] for key in ('run_id', 'model_variant', 'seed')})
        if any(result.get(key) != value for key, value in expected.items()):
            raise ValueError(f'Existing evaluation identity mismatch: {path}')
        for name, digest in result['prediction_artifacts'].items():
            if shared.sha256_file(path.parent / name) != digest:
                raise ValueError(f'Existing evaluation artifact changed: {name}')
        with np.load(path.parent / 'predictions.npz', allow_pickle=False) as data:
            if (not np.array_equal(data['sample_indices'], self.sample_indices[partition])
                    or not np.array_equal(data['targets'], self.targets[partition])):
                raise ValueError('Stored prediction rows no longer match the frozen partition.')
            check_class_masks(self.samples[partition], data['targets'], data['probabilities'])
            metrics = compute_metrics(data['targets'], data['probabilities'])
        if (result['metrics'] != {key: metrics[source] for key, source in METRICS.items()}
                or result['confusion_matrix'] != metrics['confusion_matrix']
                or result['per_class'] != metrics['per_class']
                or result['sample_count'] != metrics['sample_count']):
            raise ValueError('Stored metrics do not match the preserved sample probabilities.')
        return result

    def save_result(self, spec, partition, model_hash, probabilities, seconds):
        targets = self.targets[partition]
        probabilities = np.asarray(probabilities, dtype=np.float64)
        check_class_masks(self.samples[partition], targets, probabilities)
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
            for index, (sample, target, predicted) in enumerate(zip(self.samples[partition], targets, probabilities)):
                writer.writerow([int(self.sample_indices[partition][index]), type_name(sample), int(target.argmax()),
                                 int(predicted.argmax()), *target.tolist(), *predicted.tolist()])
        counts = np.asarray(metrics['confusion_matrix'], dtype=np.int64)
        normalized = row_normalized(counts)
        for name, matrix in ((names[2], counts), (names[3], normalized)):
            with (folder / name).open('w', newline='', encoding='utf-8-sig') as stream:
                writer = csv.writer(stream)
                writer.writerow(['reference/predicted', *CLASS_NAMES])
                writer.writerows([[CLASS_NAMES[i], *row.tolist()] for i, row in enumerate(matrix)])
        result = dict(schema_version=1, status='completed', protocol_sha256=self.protocol_hash,
                      selection_freeze_sha256=self.freeze_hash, checkpoint_sha256=model_hash,
                      **{key: spec[key] for key in ('run_id', 'model_variant', 'seed')}, partition=partition,
                      best_epoch=spec.get('best_epoch'), sample_count=len(targets), target_source=TARGET_SOURCE,
                      metrics={key: metrics[source] for key, source in METRICS.items()}, per_class=metrics['per_class'],
                      confusion_matrix=counts.tolist(), confusion_matrix_row_normalized=normalized.tolist(),
                      class_support=counts.sum(axis=1).tolist(),
                      prediction_artifacts={name: shared.sha256_file(folder / name) for name in names},
                      inference_seconds=seconds, inference_precision='float32 without autocast' if spec['seed'] is not None else 'float64 prior',
                      inference_batch_size=BATCH_SIZE if spec['seed'] is not None else None, completed_at=shared.now())
        shared.write(folder / 'metrics.json', result)
        shared.write(self.output / 'status.json', dict(status='evaluating', run_id=spec['run_id'], partition=partition,
                     completed_partitions=len(self.records) + 1, expected_partitions=33, updated_at=shared.now()))
        return result

    def prior_evaluation(self):
        spec = dict(run_id=PRIOR, model_variant=PRIOR, seed=None)
        for partition, rows in self.samples.items():
            result = self.read_completed(spec, partition, self.prior_hash)
            if result is None:
                started = time.perf_counter()
                probabilities = prior_predictions(self.priors, rows)
                result = self.save_result(spec, partition, self.prior_hash, probabilities, time.perf_counter() - started)
            self.records.append(result)
        del self.samples['train'], self.targets['train'], self.sample_indices['train']
        gc.collect()

    def neural(self):
        torch.set_num_threads(1)
        device = torch.device(self.device_name or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.inference_device = device.type
        bank_identity, banks = None, {}
        for spec in self.selection:
            saved = {name: self.read_completed(spec, name, spec['checkpoint_sha256']) for name in self.samples}
            if all(saved.values()):
                self.records.extend(saved.values())
                continue
            if shared.sha256_file(resolve_project_path(spec['best_checkpoint'])) != spec['checkpoint_sha256']:
                raise ValueError('Selected checkpoint changed after the Test gate.')
            checkpoint = torch.load(resolve_project_path(spec['best_checkpoint']), map_location='cpu', weights_only=False)
            shared.set_random_seed(spec['seed'])
            if device.type == 'cuda':
                backend = self.protocol['training']['backend']
                torch.use_deterministic_algorithms(backend['deterministic_algorithms'])
                torch.backends.cudnn.deterministic = backend['cudnn_deterministic']
                torch.backends.cudnn.benchmark = backend['cudnn_benchmark']
            model, stats = shared.source_model(checkpoint)
            keys = tuple(model.model_input_keys)
            identity = hashlib.sha256(json.dumps(dict(stats=checkpoint['normalization'], keys=keys), sort_keys=True).encode()).hexdigest()
            if bank_identity != identity:
                banks = {name: shared.CachedInputs(rows, list(range(len(rows))), stats, keys)
                         for name, rows in self.samples.items()}
                bank_identity = identity
            model.to(device).eval()
            for name in self.samples:
                if saved[name]:
                    self.records.append(saved[name])
                    continue
                started = time.perf_counter()
                probabilities = []
                loader = DataLoader(banks[name], batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                                    pin_memory=device.type == 'cuda')
                with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
                    for batch in loader:
                        inputs = {key: batch[key].to(device, non_blocking=True) for key in keys}
                        probabilities.append(model(inputs)['probabilities'].float().cpu().numpy())
                result = self.save_result(spec, name, spec['checkpoint_sha256'], np.concatenate(probabilities),
                                          time.perf_counter() - started)
                self.records.append(result)
                print(f"Evaluated {spec['run_id']} / {name}: NLL={result['metrics']['loss']:.6f}", flush=True)
            del model, checkpoint
            gc.collect()
        banks.clear()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    def run(self):
        self.gate()
        self.load_train_validation()
        self.load_test()
        self.prior_evaluation()
        self.neural()
        if len(self.records) != 33:
            raise ValueError('Expected 15 neural runs on Validation/Test plus prior on Train/Validation/Test.')
        summary = aggregate_results(self.records, self.protocol_hash)
        for partition in ('validation', 'test'):
            if set(summary['partitions'][partition]['models']) != {PRIOR, *VARIANTS}:
                raise ValueError('Final comparison must include exactly the four requested models.')
        summary.update(completed_at=shared.now(), selection_freeze_sha256=self.freeze_hash,
                       evaluation_run_count=33, all_models_selected_without_test=True,
                       neural_inference=dict(batch_size=BATCH_SIZE, precision='FP32', autocast=False,
                                             device=self.inference_device,
                                             cuda_backend=self.protocol['training']['backend'],
                                             cpu_policy='deterministic safe defaults',
                                             training_curve_precision=self.protocol['training']['amp'],
                                             training_batch_size=self.protocol['training']['batch_size']))
        shared.write(self.output / 'summary.json', summary)
        for variant in (PRIOR, *VARIANTS):
            shared.write(self.output / variant / 'confusion_summary.json',
                         {name: value['models'][variant] for name, value in summary['partitions'].items()
                          if variant in value['models']})
        with (self.output / 'metrics_mean_std.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.writer(stream)
            writer.writerow(['partition', 'model_variant', 'n_runs', 'metric', 'mean', 'std', 'std_ddof'])
            for name, partition in summary['partitions'].items():
                for variant, model in partition['models'].items():
                    for metric, value in model['metrics'].items():
                        writer.writerow([name, variant, model['n_runs'], metric, value['mean'], value['std'], 1])
        lines = ['# Phase 2 selected-checkpoint metrics', '',
                 'Mean ± sample SD across 5 seeds; the deterministic Train-only prior has no seed SD.', '',
                 'Reference classes are the argmax of embedded soft pseudo labels. Test is evaluated after checkpoint selection is sealed.', '']
        for name in ('validation', 'test'):
            lines += [f'## {name.title()}', '', '| Model | ' + ' | '.join(METRICS) + ' |',
                      '|---|' + '---|' * len(METRICS)]
            for variant in (PRIOR, *VARIANTS):
                model = summary['partitions'][name]['models'][variant]
                cells = [f"{v['mean']:.6f}" + (f" ± {v['std']:.6f}" if v['std'] is not None else ' (fixed)')
                         for v in model['metrics'].values()]
                lines.append('| ' + variant + ' | ' + ' | '.join(cells) + ' |')
            lines.append('')
        (self.output / 'metrics_mean_std.md').write_text('\n'.join(lines), encoding='utf-8')
        shared.write(self.output / 'status.json', dict(status='completed', completed_partitions=33,
                     test_evaluation_performed=True, updated_at=shared.now()))
        return summary


def self_check():
    """Small CPU-only fixtures; never opens formal data, logs, or weights."""
    targets = np.eye(8, dtype=np.float64)
    raw = compute_metrics(targets, targets)
    assert raw['soft_nll'] == raw['js_divergence'] == raw['l1_distance'] == 0
    assert raw['balanced_accuracy'] == raw['macro_f1_all_8_classes'] == 1
    np.testing.assert_array_equal(row_normalized(raw['confusion_matrix']), np.eye(8))
    base = dict(partition='validation', model_variant='full', metrics={k: raw[v] for k, v in METRICS.items()},
                confusion_matrix=raw['confusion_matrix'], per_class=raw['per_class'])
    rows = [{**base, 'seed': seed, 'run_id': f'full__seed{seed}', 'metrics': {**base['metrics'], 'loss': float(i)}}
            for i, seed in enumerate(SEEDS)]
    model = aggregate_results(rows, 'synthetic')['partitions']['validation']['models']['full']
    assert model['metrics']['loss']['mean'] == 2
    assert np.isclose(model['metrics']['loss']['std'], np.sqrt(2.5))
    prior = aggregate_results([{**base, 'seed': None, 'model_variant': PRIOR, 'run_id': PRIOR}], 'synthetic')
    assert prior['partitions']['validation']['models'][PRIOR]['metrics']['loss']['std'] is None
    try:
        aggregate_results(rows[:-1] + [rows[0]], 'synthetic')
    except ValueError:
        pass
    else:
        raise AssertionError('Duplicate seeds must fail.')
    synthetic = [dict(counterpart_type=np.array([1, 0]), y=targets[0]),
                 dict(counterpart_type=np.array([0, 1]), y=targets[6])]
    fit, counts = _prior_probabilities(synthetic)
    assert counts == dict(satellite=1, ground_region=1)
    np.testing.assert_array_equal(prior_predictions(fit, synthetic), np.stack([targets[0], targets[6]]))
    # An unusable held-out row must never enter the Train-only fit.
    train_and_holdout = synthetic + [None]
    assert _prior_probabilities([train_and_holdout[i] for i in (0, 1)]) == (fit, counts)
    check_class_masks(synthetic, np.stack([targets[0], targets[6]]), prior_predictions(fit, synthetic))
    qa = ROOT / 'qa'
    qa.mkdir(parents=True, exist_ok=True)
    # Preserve the tiny fixture for inspection; Windows indexers can hold files open.
    folder = Path(tempfile.mkdtemp(prefix='evaluation_fixture_', dir=qa))
    protocol = dict(training=dict(max_epochs=50, initialization='random', early_stopping=False),
                    neural_runs=[dict(run_id=f'{variant}__seed{seed}', model_variant=variant, seed=seed)
                                 for variant in VARIANTS for seed in SEEDS])
    shared.write(folder / 'training_protocol.json', protocol)
    evaluator = Phase2Evaluation(folder, device='cpu')
    try:
        evaluator.gate()
    except ValueError as error:
        assert 'incomplete' in str(error)
    else:
        raise AssertionError('Incomplete training must never reach the Test dataset.')
    assert not hasattr(evaluator, 'dataset_path')
    evaluator.freeze_hash = 'synthetic_selection'
    evaluator.samples = dict(validation=synthetic)
    evaluator.targets = dict(validation=np.stack([targets[0], targets[6]]))
    evaluator.sample_indices = dict(validation=np.array([100, 200]))
    spec = dict(run_id=PRIOR, model_variant=PRIOR, seed=None)
    saved = evaluator.save_result(spec, 'validation', 'synthetic_prior', prior_predictions(fit, synthetic), 0.0)
    assert evaluator.read_completed(spec, 'validation', 'synthetic_prior') == saved
    (evaluator.output / 'runs' / PRIOR / 'validation/predictions.csv').write_text('corrupt', encoding='utf-8')
    try:
        evaluator.read_completed(spec, 'validation', 'synthetic_prior')
    except ValueError:
        pass
    else:
        raise AssertionError('Tampered evaluation artifacts must fail.')
    result = dict(status='passed', formal_data_read=False, fixture_only=True, fixture_directory=str(folder),
                  checks=['metric identities', 'sample SD ddof=1', 'duplicate seeds rejected',
                          'prior has no seed SD', 'Train-only prior fit', 'class masks',
                          'incomplete matrix blocks Test access', 'NPZ/CSV/JSON round trip', 'artifact tamper detection'])
    shared.write(qa / 'evaluation_self_check.json', result)
    return result


def evaluate(root=ROOT):
    """Run the guarded evaluation once; reuse only verified completed outputs."""
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
            return Phase2Evaluation(root).run()
        except Exception:
            shared.write(root / 'evaluation/failure.json', dict(status='failed', time=shared.now(), traceback=traceback.format_exc()))
            raise
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == '__main__':
    evaluate()
