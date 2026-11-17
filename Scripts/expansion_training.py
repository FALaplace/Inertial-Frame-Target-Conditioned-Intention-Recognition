"""Freeze and run the 18,000-sample continuation matrix without making figures."""
from __future__ import annotations

import os
for _name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

from dataclasses import asdict
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import pickle
import random
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data_processing import iter_training_data
from experiment_setup import load_split_indices, sha256_file, verify_manifest_dataset
from model import build_intention_model
from model_train import (NormalizationStats, normalize_model_sample, positions_for_sample_indices,
                         run_epoch, runtime_environment, set_random_seed)
from phase3_models import build_phase3_model

IDEA = Path(__file__).resolve().parents[1]
ROOT = IDEA / 'experiment_data/expansion_18000_v1'
FROZEN = IDEA / 'dataset'
DATA = FROZEN / 'x_samples_frozen_18000_v1.json'
SPLIT = IDEA / 'experiment_data/splits/actor_isolated_13000_2500_2500_v1__labels_18000_v1.json'
PROTOCOL = ROOT / 'training_protocol.json'
SEEDS = (7, 17, 42, 73, 101)
TRAINING = {'batch_size': 32, 'max_epochs': 30, 'learning_rate': 1e-4,
            'weight_decay': 1e-4, 'optimizer': 'AdamW', 'optimizer_state': 'reset_for_continuation',
            'minimum_epochs': 10, 'early_stopping_patience': 8,
            'selection_metric': 'validation_soft_nll', 'selection_mode': 'min',
            'include_initial_checkpoint_in_selection': True,
            'normalization_policy': 'exact corresponding old checkpoint statistics; never refit',
            'loss': 'soft_cross_entropy', 'amp': 'cuda_float16', 'torch_cpu_threads': 1}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for attempt in range(6):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.1 * 2 ** attempt)


def save_torch(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.pt.tmp')
    torch.save(payload, tmp)
    for attempt in range(6):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.1 * 2 ** attempt)


def source_model(checkpoint):
    variant = checkpoint['config']['model_variant']
    kwargs = checkpoint['experiment']['model_kwargs']
    factory = build_intention_model if variant in ('full', 'factual_only', 'full_without_analytic') else build_phase3_model
    model = factory(variant, **kwargs)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    stats = NormalizationStats.from_checkpoint_dict(checkpoint['normalization'])
    for key, value in vars(stats).items():
        if not np.isfinite(value).all() or ('scale' in key and not (value > 0).all()):
            raise ValueError(f'Invalid source normalization: {key}')
    return model, stats


def freeze_protocol():
    if PROTOCOL.exists():
        protocol = read(PROTOCOL)
        verify_protocol(protocol)
        return protocol
    manifest, _ = load_split_indices(SPLIT)
    if not manifest.get('frozen') or manifest['dataset']['version'] != 'sgp4_1min_18000_rareclass_v1_20260910':
        raise ValueError('The approved 18000 dataset must be frozen first.')
    verify_manifest_dataset(manifest, DATA)
    phase3 = read(IDEA / 'experiment_data/configs/phase3_extended_v3.json')
    specs = [dict(run_key=k, model_variant=k, source_phase='phase2_core_v2', group=g)
             for k, g in (('full', 'main'), ('factual_only', 'baseline'), ('full_without_analytic', 'ablation'))]
    for spec in phase3['run_matrix']['conditions']:
        if spec['trainer'] == 'torch' and spec['comparison_group'] in ('e3_unified_baselines', 'e5_structure_ablation'):
            specs.append(dict(run_key=spec['run_key'], model_variant=spec['model_variant'],
                              source_phase='phase3_extended_v3',
                              group='baseline' if spec['comparison_group'] == 'e3_unified_baselines' else 'ablation'))
    order = {'main': 0, 'baseline': 1, 'ablation': 2}
    specs.sort(key=lambda s: order[s['group']])
    runs = []
    for spec in specs:
        for seed in SEEDS:
            stem = f"sgp4_1min_10400_labels_v5_clean_20260817__{spec['source_phase']}__actor_70_15_15_calibration_train_v1__{spec['run_key']}__seed{seed}"
            checkpoint = IDEA / 'experiment_data/checkpoints' / (stem + '.pt')
            log = IDEA / 'experiment_data/logs' / (stem + '__training.json')
            if not checkpoint.is_file() or not log.is_file():
                raise FileNotFoundError(f'Missing source weight or history: {stem}')
            runs.append({**spec, 'seed': seed, 'run_id': f"{spec['run_key']}__seed{seed}",
                         'initial_checkpoint': str(checkpoint.relative_to(IDEA)),
                         'initial_checkpoint_sha256': sha256_file(checkpoint),
                         'initial_log': str(log.relative_to(IDEA)), 'initial_log_sha256': sha256_file(log)})
    code = ['expansion_training.py', 'expansion_classical_baselines.py', 'data_processing.py',
            'analytic_layer.py', 'orbit_core.py', 'model.py', 'model_train.py',
            'phase3_models.py', 'phase3_rf_baseline.py', 'phase3_setup.py',
            'phase2_experiment.py', 'model_evaluate.py', 'experiment_setup.py']
    protocol = {'schema_version': 1, 'experiment_id': 'expansion_18000_v1',
                'status': 'frozen_before_training', 'created_at': now(),
                'user_authorization': 'Approved frozen 18000 selection and model training; defer all figures; record baseline loss.',
                'dataset': manifest['dataset'], 'dataset_path': str(DATA.relative_to(IDEA)),
                'split_id': manifest['split_id'], 'split_manifest': str(SPLIT.relative_to(IDEA)),
                'split_manifest_sha256': sha256_file(SPLIT),
                'dataset_freeze_manifest_sha256': sha256_file(FROZEN / 'freeze_manifest.json'),
                'sample_counts': manifest['audit']['sample_counts'],
                'training': TRAINING, 'neural_runs': runs,
                'classical': {'seeds': list(SEEDS), 'random_forest': 'refit_all_300_trees_on_new_train',
                              'tree_curve_steps': [50, 100, 150, 200, 250, 300], 'type_prior': 'single_fit_point'},
                'evaluation': {'test_used_for_training_or_selection': False,
                               'test_evaluation': 'separate later evaluation after all selected checkpoints are fixed',
                               'scope': manifest['config']['evaluation_scope'],
                               'human_blind_labels_accessed': False, 'figures_generated': False},
                'historical_protocol_sha256': {phase: sha256_file(IDEA / f'experiment_data/configs/{phase}.json')
                                              for phase in ('phase2_core_v2', 'phase3_extended_v3')},
                'code_sha256': {name: sha256_file(IDEA / 'scripts' / name) for name in code}}
    write(PROTOCOL, protocol)
    return protocol


def verify_protocol(protocol):
    for name, digest in protocol['code_sha256'].items():
        if sha256_file(IDEA / 'scripts' / name) != digest:
            raise ValueError(f'Code changed after protocol freeze: {name}')
    if sha256_file(SPLIT) != protocol['split_manifest_sha256'] or sha256_file(DATA) != protocol['dataset']['sha256']:
        raise ValueError('Frozen dataset or split hash mismatch.')
    if sha256_file(FROZEN / 'freeze_manifest.json') != protocol['dataset_freeze_manifest_sha256']:
        raise ValueError('Dataset freeze manifest changed.')
    for run in protocol['neural_runs']:
        for field in ('initial_checkpoint', 'initial_log'):
            if sha256_file(IDEA / run[field]) != run[field + '_sha256']:
                raise ValueError(f'Source artifact changed: {run[field]}')


class CachedInputs(Dataset):
    """Normalize once per checkpoint, preserving the existing input transformation."""
    def __init__(self, samples, positions, stats, keys):
        self.rows = []
        for position in positions:
            normalized = normalize_model_sample(samples[position], stats)
            self.rows.append({key: torch.from_numpy(normalized[key]) for key in (*keys, 'y')})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class ExpansionTraining:
    def __init__(self):
        torch.set_num_threads(1)
        self.protocol = freeze_protocol()
        self.protocol_hash = sha256_file(PROTOCOL)
        self.manifest, self.split = load_split_indices(SPLIT)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metadata = {'experiment_id': self.protocol['experiment_id'],
                         'dataset_version': self.protocol['dataset']['version'],
                         'label_version': self.protocol['dataset']['label_version'],
                         'dataset_sha256': self.protocol['dataset']['sha256'],
                         'split_id': self.protocol['split_id'],
                         'split_manifest_sha256': self.protocol['split_manifest_sha256'],
                         'protocol_sha256': self.protocol_hash}

    def load_samples(self):
        requested = set(self.split['train'] + self.split['validation'])
        cache = IDEA / 'experiment_data/cache/train_validation_inputs.pkl'
        cache_meta = cache.with_suffix('.json')
        identity = {'dataset_sha256': self.metadata['dataset_sha256'],
                    'split_sha256': self.metadata['split_manifest_sha256'],
                    'code_sha256': {k: self.protocol['code_sha256'][k]
                                   for k in ('data_processing.py', 'analytic_layer.py', 'orbit_core.py')}}
        if cache.is_file() and cache_meta.is_file():
            meta = read(cache_meta)
            if meta['identity'] != identity or sha256_file(cache) != meta['sha256']:
                raise ValueError('Prepared input cache provenance mismatch.')
            with cache.open('rb') as fp:
                self.samples = pickle.load(fp)
        else:
            self.samples = []
            start = time.perf_counter()
            for sample in iter_training_data(DATA, sample_indices=requested, allow_sidecar_labels=False):
                self.samples.append(sample)
                if len(self.samples) % 500 == 0:
                    write(ROOT / 'status.json', {'status': 'preparing_inputs', 'pid': os.getpid(), 'prepared': len(self.samples),
                                                'expected': len(requested), 'updated_at': now()})
                    print(f'Prepared {len(self.samples)}/{len(requested)} train/validation inputs', flush=True)
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix('.tmp')
            with tmp.open('wb') as fp:
                pickle.dump(self.samples, fp, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(cache)
            write(cache_meta, {'identity': identity, 'sha256': sha256_file(cache),
                               'sample_count': len(self.samples), 'preparation_seconds': time.perf_counter() - start})
        if {int(s['sample_index']) for s in self.samples} != requested or len(self.samples) != 15500:
            raise ValueError('Prepared data must contain train and validation exactly once.')
        self.train_positions = positions_for_sample_indices(self.samples, self.split['train'])
        self.validation_positions = positions_for_sample_indices(self.samples, self.split['validation'])

    def run_one(self, spec):
        folder = ROOT / 'neural' / spec['run_id']
        log_path, best_path, last_path = (folder / name for name in ('training.json', 'best.pt', 'last.pt'))
        run_identity = {'protocol_sha256': self.protocol_hash, 'run_spec': spec}
        identity_path = folder / 'run_manifest.json'
        if identity_path.exists() and read(identity_path) != run_identity:
            raise ValueError('Existing run identity mismatch.')
        if log_path.is_file():
            log = read(log_path)
            if log['experiment']['protocol_sha256'] != self.protocol_hash or log['experiment']['run_id'] != spec['run_id']:
                raise ValueError('Existing run belongs to another protocol.')
            if log['status'] == 'completed':
                if sha256_file(best_path) != log['best_checkpoint']['sha256']:
                    raise ValueError('Completed checkpoint hash mismatch.')
                print(f"Already completed {spec['run_id']}", flush=True)
                return log
        elif folder.exists() and any(folder.iterdir()) and not identity_path.exists():
            raise FileExistsError(f'Unrecognized partial run: {folder}')
        folder.mkdir(parents=True, exist_ok=True)
        write(identity_path, run_identity)
        set_random_seed(spec['seed'])
        source = torch.load(IDEA / spec['initial_checkpoint'], map_location='cpu', weights_only=False)
        source_log = read(IDEA / spec['initial_log'])
        if source['config']['random_seed'] != spec['seed'] or source['config']['model_variant'] != spec['model_variant']:
            raise ValueError('Initial weight seed or variant mismatch.')
        _, old_split = load_split_indices(IDEA / 'experiment_data/splits/actor_70_15_15_calibration_train_v1__labels_v5_clean.json')
        if source['split_sample_indices'] != old_split:
            raise ValueError('Initial checkpoint uses unexpected original split.')
        model, stats = source_model(source)
        model.to(self.device)
        model_keys = tuple(model.model_input_keys)
        train_data = CachedInputs(self.samples, self.train_positions, stats, model_keys)
        val_data = CachedInputs(self.samples, self.validation_positions, stats, model_keys)
        generator = torch.Generator().manual_seed(spec['seed'])
        options = {'batch_size': TRAINING['batch_size'], 'num_workers': 0, 'pin_memory': self.device.type == 'cuda'}
        train_loader = DataLoader(train_data, shuffle=True, generator=generator, **options)
        train_eval_loader = DataLoader(train_data, shuffle=False, **options)
        val_loader = DataLoader(val_data, shuffle=False, **options)
        optimizer = torch.optim.AdamW(model.parameters(), lr=TRAINING['learning_rate'], weight_decay=TRAINING['weight_decay'])
        scaler = torch.amp.GradScaler('cuda', enabled=self.device.type == 'cuda')
        experiment = {**self.metadata, **spec, 'model_kwargs': model.model_kwargs, 'model_input_keys': list(model_keys),
                      'initial_best_epoch': source['epoch'], 'initial_training_epochs': len(source_log['training']['history']),
                      'normalization_source_sha256': spec['initial_checkpoint_sha256'],
                      'runtime_environment': runtime_environment(self.device),
                      'train_loss_definition': 'sample-weighted online soft cross entropy with dropout; epoch 0 is eval-mode baseline',
                      'validation_loss_definition': 'eval-mode sample-weighted soft cross entropy',
                      'test_samples_loaded': False}
        write(folder / 'legacy_history.json', {'source_path': spec['initial_log'],
                                              'source_sha256': spec['initial_log_sha256'],
                                              'dataset': 'original_10400',
                                              'history': source_log['training']['history'],
                                              'plot_policy': 'different data phase; do not splice into one uninterrupted loss curve'})
        history, best_epoch, stale, begin = [], 0, 0, 1
        started_at, start = now(), time.perf_counter()
        best_loss = float('inf')

        def payload(epoch, train_metrics, val_metrics):
            return {'schema_version': 1, 'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 'scaler_state_dict': scaler.state_dict(),
                    'normalization': stats.checkpoint_dict(), 'split_sample_indices': self.split,
                    'config': {**TRAINING, 'random_seed': spec['seed'], 'model_variant': spec['model_variant']},
                    'experiment': experiment, 'train_metrics': asdict(train_metrics), 'validation_metrics': asdict(val_metrics),
                    'history': history, 'best_epoch': best_epoch, 'best_validation_loss': best_loss, 'stale_epochs': stale,
                    'rng': {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
                            'cuda': torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else [],
                            'loader': generator.get_state()}}

        if last_path.exists():
            last = torch.load(last_path, map_location='cpu', weights_only=False)
            if (last['experiment']['protocol_sha256'] != self.protocol_hash
                    or last['experiment']['run_id'] != spec['run_id']
                    or last['experiment']['initial_checkpoint_sha256'] != spec['initial_checkpoint_sha256']
                    or last['normalization'] != source['normalization']
                    or last['split_sample_indices'] != self.split):
                raise ValueError('Partial checkpoint protocol mismatch.')
            model.load_state_dict(last['model_state_dict'], strict=True)
            optimizer.load_state_dict(last['optimizer_state_dict'])
            scaler.load_state_dict(last['scaler_state_dict'])
            history, best_epoch, best_loss, stale = last['history'], last['best_epoch'], last['best_validation_loss'], last['stale_epochs']
            begin = last['epoch'] + 1
            rng = last['rng']
            random.setstate(rng['python']); np.random.set_state(rng['numpy']); torch.set_rng_state(rng['torch'])
            if self.device.type == 'cuda':
                torch.cuda.set_rng_state_all(rng['cuda'])
            generator.set_state(rng['loader'])
        else:
            initial_train = run_epoch(model, train_eval_loader, self.device)
            initial_val = run_epoch(model, val_loader, self.device)
            best_loss = initial_val.loss
            history.append({'epoch': 0, 'phase': 'initial_weight_evaluation', 'train': asdict(initial_train),
                            'validation': asdict(initial_val), 'learning_rate': 0.0, 'epoch_seconds': time.perf_counter() - start})
            initial = payload(0, initial_train, initial_val)
            save_torch(best_path, initial)
            save_torch(last_path, initial)
        log = {'schema_version': 1, 'status': 'running', 'experiment': experiment, 'training_config': TRAINING,
               'started_at': started_at, 'history': history, 'best_epoch': best_epoch, 'best_validation_loss': best_loss}
        write(log_path, log)
        already_stopped = begin - 1 >= TRAINING['minimum_epochs'] and stale >= TRAINING['early_stopping_patience']
        epochs = () if already_stopped else range(begin, TRAINING['max_epochs'] + 1)
        for epoch in epochs:
            epoch_start = time.perf_counter()
            train_metrics = run_epoch(model, train_loader, self.device, optimizer, scaler)
            val_metrics = run_epoch(model, val_loader, self.device)
            history.append({'epoch': epoch, 'phase': 'continuation_18000', 'train': asdict(train_metrics),
                            'validation': asdict(val_metrics), 'learning_rate': optimizer.param_groups[0]['lr'],
                            'epoch_seconds': time.perf_counter() - epoch_start})
            if val_metrics.loss < best_loss:
                best_loss, best_epoch, stale = val_metrics.loss, epoch, 0
                save_torch(best_path, payload(epoch, train_metrics, val_metrics))
            else:
                stale += 1
            save_torch(last_path, payload(epoch, train_metrics, val_metrics))
            log.update(history=history, best_epoch=best_epoch, best_validation_loss=best_loss, updated_at=now())
            write(log_path, log)
            write(ROOT / 'status.json', {'status': 'training', 'pid': os.getpid(), 'run_id': spec['run_id'], 'epoch': epoch,
                                        'maximum_epochs': TRAINING['max_epochs'], 'train_loss': train_metrics.loss,
                                        'validation_loss': val_metrics.loss, 'best_epoch': best_epoch, 'updated_at': now()})
            print(f"{spec['run_id']} epoch {epoch:02d}/{TRAINING['max_epochs']} train={train_metrics.loss:.6f} val={val_metrics.loss:.6f} best={best_epoch} seconds={history[-1]['epoch_seconds']:.1f}", flush=True)
            if epoch >= TRAINING['minimum_epochs'] and stale >= TRAINING['early_stopping_patience']:
                break
        log.update(status='completed', completed_at=now(), process_wall_seconds=time.perf_counter() - start,
                   stopping_reason='validation_patience' if stale >= TRAINING['early_stopping_patience'] else 'max_epochs',
                   best_checkpoint={'path': str(best_path.relative_to(IDEA)), 'sha256': sha256_file(best_path)},
                   last_checkpoint={'path': str(last_path.relative_to(IDEA)), 'sha256': sha256_file(last_path)})
        write(log_path, log)
        del model, train_data, val_data, train_loader, train_eval_loader, val_loader, optimizer, scaler, source
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        return log

    def run(self):
        ROOT.mkdir(parents=True, exist_ok=True)
        write(ROOT / 'status.json', {'status': 'starting', 'pid': os.getpid(), 'started_at': now(),
                                    'neural_run_count': len(self.protocol['neural_runs']), 'figures_generated': False})
        self.load_samples()
        from expansion_classical_baselines import run_classical_baselines
        run_classical_baselines(self.samples, self.train_positions, self.validation_positions,
                               self.manifest, ROOT, self.metadata, seeds=SEEDS)
        results = []
        for spec in self.protocol['neural_runs']:
            result = self.run_one(spec)
            results.append({'run_id': spec['run_id'], 'best_epoch': result['best_epoch'],
                            'best_validation_loss': result['best_validation_loss'],
                            'initial_validation_loss': result['history'][0]['validation']['loss'],
                            'epochs_completed': result['history'][-1]['epoch'],
                            'checkpoint': result['best_checkpoint']})
            write(ROOT / 'training_summary.json', {'status': 'running', 'completed_neural_runs': len(results),
                                                  'expected_neural_runs': len(self.protocol['neural_runs']), 'runs': results})
        verify_protocol(self.protocol)
        write(ROOT / 'training_summary.json', {'status': 'completed', 'completed_at': now(),
                                              'completed_neural_runs': len(results), 'runs': results,
                                              'figures_generated': False, 'test_evaluation_performed': False})
        write(ROOT / 'status.json', {'status': 'completed', 'completed_at': now(), 'neural_runs': len(results),
                                    'classical_runs': len(SEEDS) + 1, 'figures_generated': False})


if __name__ == '__main__':
    try:
        ExpansionTraining().run()
    except Exception:
        write(ROOT / 'failure.json', {'status': 'failed', 'time': now(), 'traceback': traceback.format_exc()})
        raise
