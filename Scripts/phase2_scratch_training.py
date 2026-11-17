"""Full / seed 7: random initialization, 50 epochs on the frozen 18k split.

Run directly in PyCharm with the intention-ir interpreter. Re-running resumes
only this experiment's last checkpoint; historical model weights are unused.
"""
from __future__ import annotations

import expansion_training as shared  # Set deterministic runtime flags first.

from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader

from expansion_training_balanced import balanced_run_epoch
from model_train import fit_normalization_stats

IDEA = shared.IDEA
ROOT = IDEA / 'experiment_data/scratch_18000_full_seed7_v1'
SEED = 7
EPOCHS = 50
MODEL_KWARGS = dict(trajectory_hidden_dim=128, maneuver_hidden_dim=32,
                    event_hidden_dim=64, analytic_hidden_dim=96,
                    static_hidden_dim=32, fusion_hidden_dim=128, dropout=0.1)
TRAINING = dict(batch_size=32, max_epochs=EPOCHS, learning_rate=1e-3,
                weight_decay=1e-4, optimizer='AdamW', initialization='random',
                optimizer_state='new', early_stopping=False,
                selection_metric='validation_soft_nll', selection_mode='min',
                include_initial_checkpoint_in_selection=False,
                normalization_policy='fit on current 13000 training samples only',
                loss='soft_cross_entropy', amp='cuda_float16', torch_cpu_threads=1)
CACHE = IDEA / 'experiment_data/cache/train_validation_inputs.pkl'
PLOT_PYTHON = Path('D:/ProgramFiles/Anaconda3/python.exe')


def state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class ScratchTraining:
    def __init__(self):
        torch.set_num_threads(1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type != 'cuda':
            raise RuntimeError('This 50-epoch run requires the configured CUDA device.')
        self.manifest, self.split = shared.load_split_indices(shared.SPLIT)
        shared.verify_manifest_dataset(self.manifest, shared.DATA)
        if not self.manifest.get('frozen'):
            raise ValueError('The dataset split must be frozen.')
        assert {k: len(v) for k, v in self.split.items()} == dict(train=13000, validation=2500, test=2500)
        assert not (set(self.split['train']) & set(self.split['validation']))
        assert not (set(self.split['train'] + self.split['validation']) & set(self.split['test']))
        cache_meta = shared.read(CACHE.with_suffix('.json'))
        expected = dict(dataset_sha256=self.manifest['dataset']['sha256'],
                        split_sha256=shared.sha256_file(shared.SPLIT),
                        code_sha256={name: shared.sha256_file(IDEA / 'scripts' / name)
                                     for name in ('data_processing.py', 'analytic_layer.py', 'orbit_core.py')})
        if cache_meta['identity'] != expected or shared.sha256_file(CACHE) != cache_meta['sha256']:
            raise ValueError('Raw prepared input cache provenance mismatch.')
        code = ('phase2_scratch_training.py', 'expansion_training.py',
                'expansion_training_balanced.py', 'model_train.py', 'model.py',
                'data_processing.py', 'analytic_layer.py', 'orbit_core.py', 'experiment_setup.py')
        protocol = dict(schema_version=1, experiment_id=ROOT.name,
                        dataset=self.manifest['dataset'], dataset_path=str(shared.DATA.relative_to(IDEA)),
                        split_id=self.manifest['split_id'], split_manifest=str(shared.SPLIT.relative_to(IDEA)),
                        split_manifest_sha256=expected['split_sha256'],
                        dataset_freeze_manifest_sha256=shared.sha256_file(shared.FROZEN / 'freeze_manifest.json'),
                        sample_counts={k: len(v) for k, v in self.split.items()},
                        model_variant='full', seed=SEED, model_kwargs=MODEL_KWARGS, training=TRAINING,
                        initial_checkpoint=None, raw_input_cache=dict(path=str(CACHE.relative_to(IDEA)), **cache_meta),
                        code_sha256={name: shared.sha256_file(IDEA / 'scripts' / name) for name in code},
                        user_authorization='Full model, one seed, random initialization, complete 50 epochs, then inspect curves.',
                        evaluation=dict(test_used_for_training_or_selection=False, test_evaluation_performed=False),
                        metric_recording=dict(train='online with dropout; epoch 0 eval mode',
                                              validation='eval mode after each epoch',
                                              initial_epoch='random model before any optimizer update'))
        protocol_path = ROOT / 'training_protocol.json'
        if protocol_path.exists():
            if shared.read(protocol_path) != protocol:
                raise ValueError('Existing scratch protocol differs; do not overwrite a run.')
        else:
            shared.write(protocol_path, protocol)
        self.protocol = protocol
        self.protocol_hash = shared.sha256_file(protocol_path)
        self.folder = ROOT / 'neural/full__seed7'
        self.folder.mkdir(parents=True, exist_ok=True)

    def load_inputs(self):
        shared.write(ROOT / 'status.json', dict(status='preparing_inputs', pid=os.getpid(), updated_at=shared.now()))
        with CACHE.open('rb') as stream:
            self.samples = pickle.load(stream)
        requested = set(self.split['train'] + self.split['validation'])
        assert len(self.samples) == 15500
        assert {int(s['sample_index']) for s in self.samples} == requested
        self.train_positions = shared.positions_for_sample_indices(self.samples, self.split['train'])
        self.val_positions = shared.positions_for_sample_indices(self.samples, self.split['validation'])
        self.stats = fit_normalization_stats(self.samples, self.train_positions, 'full')
        for key, value in vars(self.stats).items():
            assert np.isfinite(value).all(), key
            if 'scale' in key:
                assert (value > 0).all(), key
        shared.write(ROOT / 'normalization.json', dict(source='current_training_split_only',
                     sample_count=13000, sample_indices=self.split['train'],
                     statistics=self.stats.checkpoint_dict()))
        print('Prepared inputs verified; normalization fitted on 13000 training samples.', flush=True)

    def preflight(self):
        """Check seed reproducibility, train-only fitting and an actual update."""
        shared.set_random_seed(SEED)
        first = shared.build_intention_model('full', **MODEL_KWARGS)
        initial_digest = state_digest(first)
        shared.set_random_seed(SEED)
        second = shared.build_intention_model('full', **MODEL_KWARGS)
        assert state_digest(second) == initial_digest
        shared.set_random_seed(SEED + 1)
        different = shared.build_intention_model('full', **MODEL_KWARGS)
        assert state_digest(different) != initial_digest
        del second, different
        positions = self.train_positions[:8]
        rows = [self.samples[p] for p in positions]
        stats = fit_normalization_stats(rows, list(range(8)), 'full')
        # An unusable validation row must never enter normalization fitting.
        same_stats = fit_normalization_stats(rows + [None], list(range(8)), 'full')
        assert stats.checkpoint_dict() == same_stats.checkpoint_dict()
        train = shared.CachedInputs(self.samples, positions, self.stats, tuple(first.model_input_keys))
        validation = shared.CachedInputs(self.samples, self.val_positions[:8], self.stats, tuple(first.model_input_keys))
        first.to(self.device)
        optimizer = torch.optim.AdamW(first.parameters(), lr=TRAINING['learning_rate'], weight_decay=TRAINING['weight_decay'])
        scaler = torch.amp.GradScaler('cuda')
        train_metrics = balanced_run_epoch(first, DataLoader(train, batch_size=8), self.device, optimizer, scaler)
        assert state_digest(first) != initial_digest
        val_metrics = balanced_run_epoch(first, DataLoader(validation, batch_size=8), self.device)
        assert sum(train_metrics.class_support) == sum(val_metrics.class_support) == 8
        path = ROOT / 'preflight/one_step.pt'
        shared.save_torch(path, dict(model_state_dict=first.state_dict()))
        restored = shared.build_intention_model('full', **MODEL_KWARGS).to(self.device)
        restored.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)['model_state_dict'], strict=True)
        assert balanced_run_epoch(restored, DataLoader(validation, batch_size=8), self.device) == val_metrics
        shared.write(ROOT / 'preflight/checks.json', dict(status='passed', created_at=shared.now(),
                     seeded_initial_state_sha256=initial_digest, same_seed_reproduces=True,
                     different_seed_changes_weights=True, historical_weights_loaded=False,
                     normalization_excludes_validation=True, optimizer_changes_weights=True,
                     checkpoint_round_trip_equal=True, test_samples_loaded=False,
                     train_metrics=asdict(train_metrics), validation_metrics=asdict(val_metrics)))
        del first, restored, optimizer, scaler, train, validation
        gc.collect()
        torch.cuda.empty_cache()
        print('Preflight passed: random initialization, train-only normalization, CUDA update and reload.', flush=True)

    def run(self):
        log_path, initial_path, best_path, last_path = (self.folder / name for name in ('training.json', 'initial.pt', 'best.pt', 'last.pt'))
        if log_path.exists() and shared.read(log_path)['status'] == 'completed':
            log = shared.read(log_path)
            assert log['experiment']['protocol_sha256'] == self.protocol_hash
            assert shared.sha256_file(best_path) == log['best_checkpoint']['sha256']
            assert shared.sha256_file(last_path) == log['last_checkpoint']['sha256']
            return log
        self.load_inputs()
        self.preflight()
        shared.set_random_seed(SEED)
        model = shared.build_intention_model('full', **MODEL_KWARGS)
        initial_digest = state_digest(model)
        assert initial_digest == shared.read(ROOT / 'preflight/checks.json')['seeded_initial_state_sha256']
        model.to(self.device)
        keys = tuple(model.model_input_keys)
        train_data = shared.CachedInputs(self.samples, self.train_positions, self.stats, keys)
        val_data = shared.CachedInputs(self.samples, self.val_positions, self.stats, keys)
        del self.samples
        gc.collect()
        generator = torch.Generator().manual_seed(SEED)
        options = dict(batch_size=TRAINING['batch_size'], num_workers=0, pin_memory=True)
        train_loader = DataLoader(train_data, shuffle=True, generator=generator, **options)
        train_eval_loader = DataLoader(train_data, shuffle=False, **options)
        val_loader = DataLoader(val_data, shuffle=False, **options)
        optimizer = torch.optim.AdamW(model.parameters(), lr=TRAINING['learning_rate'], weight_decay=TRAINING['weight_decay'])
        scaler = torch.amp.GradScaler('cuda')
        experiment = dict(experiment_id=ROOT.name, run_id='full__seed7', model_variant='full', seed=SEED,
                          model_kwargs=MODEL_KWARGS, model_input_keys=list(keys), initialization='random',
                          initial_checkpoint=None, initial_state_sha256=initial_digest,
                          dataset_sha256=self.protocol['dataset']['sha256'],
                          dataset_version=self.protocol['dataset']['version'],
                          split_id=self.protocol['split_id'], split_manifest_sha256=self.protocol['split_manifest_sha256'],
                          protocol_sha256=self.protocol_hash, runtime_environment=shared.runtime_environment(self.device),
                          normalization_source='current_13000_train_only', test_samples_loaded=False)
        history, best_epoch, best_loss, begin = [], None, None, 1
        started_at = shared.now()

        def payload(epoch):
            return dict(schema_version=1, epoch=epoch, model_state_dict=model.state_dict(),
                        optimizer_state_dict=optimizer.state_dict(), scaler_state_dict=scaler.state_dict(),
                        normalization=self.stats.checkpoint_dict(), split_sample_indices=self.split,
                        config={**TRAINING, 'random_seed': SEED, 'model_variant': 'full'}, experiment=experiment,
                        train_metrics=history[-1]['train'], validation_metrics=history[-1]['validation'],
                        history=history, best_epoch=best_epoch, best_validation_loss=best_loss, started_at=started_at,
                        rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                                 cuda=torch.cuda.get_rng_state_all(), loader=generator.get_state()))

        if last_path.exists():
            last = torch.load(last_path, map_location='cpu', weights_only=False)
            assert last['experiment']['protocol_sha256'] == self.protocol_hash
            assert last['experiment']['initial_state_sha256'] == initial_digest
            assert last['normalization'] == self.stats.checkpoint_dict()
            assert last['split_sample_indices'] == self.split
            model.load_state_dict(last['model_state_dict'], strict=True)
            optimizer.load_state_dict(last['optimizer_state_dict'])
            scaler.load_state_dict(last['scaler_state_dict'])
            history, best_epoch, best_loss = last['history'], last['best_epoch'], last['best_validation_loss']
            started_at, begin = last['started_at'], last['epoch'] + 1
            random.setstate(last['rng']['python'])
            np.random.set_state(last['rng']['numpy'])
            torch.set_rng_state(last['rng']['torch'])
            torch.cuda.set_rng_state_all(last['rng']['cuda'])
            generator.set_state(last['rng']['loader'])
        else:
            start = time.perf_counter()
            initial_train = balanced_run_epoch(model, train_eval_loader, self.device)
            initial_val = balanced_run_epoch(model, val_loader, self.device)
            history.append(dict(epoch=0, phase='random_initialization_evaluation', train=asdict(initial_train),
                                validation=asdict(initial_val), learning_rate=0.0, epoch_seconds=time.perf_counter() - start))
            shared.save_torch(initial_path, payload(0))
            shared.save_torch(last_path, payload(0))
        log = dict(schema_version=1, status='running', experiment=experiment, training_config=TRAINING,
                   started_at=started_at, history=history, best_epoch=best_epoch, best_validation_loss=best_loss)
        shared.write(log_path, log)
        print(f"Random-init epoch 0: validation NLL={history[0]['validation']['loss']:.6f}; starting epoch {begin}.", flush=True)
        for epoch in range(begin, EPOCHS + 1):
            shared.write(ROOT / 'status.json', dict(status='training', pid=os.getpid(), epoch_in_progress=epoch,
                         epochs_completed=epoch - 1, maximum_epochs=EPOCHS, updated_at=shared.now()))
            start = time.perf_counter()
            train = balanced_run_epoch(model, train_loader, self.device, optimizer, scaler)
            validation = balanced_run_epoch(model, val_loader, self.device)
            history.append(dict(epoch=epoch, phase='from_scratch_18000', train=asdict(train), validation=asdict(validation),
                                learning_rate=optimizer.param_groups[0]['lr'], epoch_seconds=time.perf_counter() - start))
            if best_loss is None or validation.loss < best_loss:
                best_epoch, best_loss = epoch, validation.loss
                shared.save_torch(best_path, payload(epoch))
            shared.save_torch(last_path, payload(epoch))
            log.update(history=history, best_epoch=best_epoch, best_validation_loss=best_loss, updated_at=shared.now())
            shared.write(log_path, log)
            print(f'Full seed7 epoch {epoch:02d}/{EPOCHS}: train={train.loss:.6f} val={validation.loss:.6f} best={best_epoch} seconds={history[-1]["epoch_seconds"]:.1f}', flush=True)
        assert [row['epoch'] for row in history] == list(range(EPOCHS + 1))
        assert best_epoch == min(history[1:], key=lambda row: row['validation']['loss'])['epoch']
        log.update(status='completed', completed_at=shared.now(), stopping_reason='completed_50_epochs',
                   initial_checkpoint=dict(path=str(initial_path.relative_to(IDEA)), sha256=shared.sha256_file(initial_path)),
                   best_checkpoint=dict(path=str(best_path.relative_to(IDEA)), sha256=shared.sha256_file(best_path)),
                   last_checkpoint=dict(path=str(last_path.relative_to(IDEA)), sha256=shared.sha256_file(last_path)))
        shared.write(log_path, log)
        shared.write(ROOT / 'training_summary.json', dict(status='completed', epochs_completed=EPOCHS,
                     best_epoch=best_epoch, best_validation=history[best_epoch]['validation'],
                     test_evaluation_performed=False, completed_at=shared.now()))
        shared.write(ROOT / 'status.json', dict(status='training_completed', epochs_completed=EPOCHS,
                     maximum_epochs=EPOCHS, best_epoch=best_epoch, updated_at=shared.now()))
        return log


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    # Hold a Windows file lock so a second IDE launch cannot duplicate this run.
    import msvcrt
    with (ROOT / 'run.lock').open('a+b') as lock:
        if lock.tell() == 0:
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            trainer = ScratchTraining()
            trainer.run()
            del trainer
            gc.collect()
            torch.cuda.empty_cache()
            script = IDEA / 'scripts/phase2_scratch_curve_preview.py'
            subprocess.run([str(PLOT_PYTHON), '-X', 'utf8', str(script)], check=True, cwd=str(IDEA))
            shared.write(ROOT / 'status.json', dict(status='completed', epochs_completed=EPOCHS,
                         maximum_epochs=EPOCHS, figures_generated=True, completed_at=shared.now()))
        except Exception:
            shared.write(ROOT / 'failure.json', dict(status='failed', time=shared.now(), traceback=traceback.format_exc()))
            raise
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == '__main__':
    main()
