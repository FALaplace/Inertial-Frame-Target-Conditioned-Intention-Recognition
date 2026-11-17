"""Run only the four Phase 2 models after the GPU benchmark passes.

Direct PyCharm entry point. Three neural variants x five seeds x 50 epochs,
then fixed-checkpoint evaluation, SD curves and confusion heatmaps.
"""
from __future__ import annotations

import expansion_training as shared

from dataclasses import asdict
import gc
import os
import pickle
import random
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from gpu_training_runtime import GpuBatches, gpu_run_epoch
from gpu_training_benchmark import GpuTelemetry
from model_train import fit_normalization_stats
from phase2_scratch_training import MODEL_KWARGS, CACHE, PLOT_PYTHON, state_digest

IDEA = shared.IDEA
ROOT = IDEA / 'experiment_data/phase2_scratch_18000_gpu_v1'
BENCHMARK = IDEA / 'experiment_data/gpu_tuning_18000_v1/fast_kernels/benchmark_report.json'
VARIANTS = ('factual_only', 'full_without_analytic', 'full')
SEEDS = (7, 17, 42, 73, 101)
EPOCHS = 50


class Phase2Training:
    def configure_backend(self):
        # Apply after set_random_seed, which restores the historical slow flags.
        torch.use_deterministic_algorithms(self.backend['deterministic_algorithms'])
        torch.backends.cudnn.deterministic = self.backend['cudnn_deterministic']
        torch.backends.cudnn.benchmark = self.backend['cudnn_benchmark']

    def __init__(self):
        torch.set_num_threads(1)
        self.device = torch.device('cuda')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for the measured GPU configuration.')
        bench = shared.read(BENCHMARK)
        if bench['status'] != 'passed' or not bench['batch32_equivalence']['passed']:
            raise ValueError('GPU benchmark and numerical equivalence must pass first.')
        for name, digest in bench['code_sha256'].items():
            if shared.sha256_file(IDEA / 'scripts' / name) != digest:
                raise ValueError(f'Benchmark code changed: {name}')
        self.batch_size = int(bench['selected_batch_size'])
        self.backend = bench['backend']
        self.configure_backend()
        self.training = dict(batch_size=self.batch_size, max_epochs=EPOCHS, learning_rate=1e-3,
                             weight_decay=1e-4, optimizer='AdamW', optimizer_state='new',
                             initialization='random', early_stopping=False,
                             include_initial_checkpoint_in_selection=False,
                             selection_metric='validation_soft_nll', selection_mode='min',
                             normalization_policy='fit each variant using current 13000 Train only',
                             loss='soft_cross_entropy', amp='cuda_float16', torch_cpu_threads=1,
                             input_storage='gpu_resident', epoch_metrics='gpu_accumulation',
                             backend=self.backend,
                             reproducibility='fixed seeds, saved RNG and checkpoints; fast cuDNN kernels may differ on repeated runs')
        self.manifest, self.split = shared.load_split_indices(shared.SPLIT)
        shared.verify_manifest_dataset(self.manifest, shared.DATA)
        counts = {k: len(v) for k, v in self.split.items()}
        assert self.manifest.get('frozen') and counts == dict(train=13000, validation=2500, test=2500)
        assert not set(self.split['train']) & set(self.split['validation'])
        assert not set(self.split['train'] + self.split['validation']) & set(self.split['test'])
        if (bench['dataset_sha256'] != self.manifest['dataset']['sha256']
                or bench['split_sha256'] != shared.sha256_file(shared.SPLIT)):
            raise ValueError('Benchmark used a different dataset or split.')
        cache_meta = shared.read(CACHE.with_suffix('.json'))
        assert shared.sha256_file(CACHE) == cache_meta['sha256']
        assert cache_meta['identity']['dataset_sha256'] == self.manifest['dataset']['sha256']
        assert cache_meta['identity']['split_sha256'] == shared.sha256_file(shared.SPLIT)
        for name, digest in cache_meta['identity']['code_sha256'].items():
            assert shared.sha256_file(IDEA / 'scripts' / name) == digest
        shared.write(ROOT / 'status.json', dict(status='preparing_inputs', pid=os.getpid(), updated_at=shared.now()))
        with CACHE.open('rb') as stream:
            self.samples = pickle.load(stream)
        assert len(self.samples) == 15500
        assert {int(s['sample_index']) for s in self.samples} == set(self.split['train'] + self.split['validation'])
        self.train_positions = shared.positions_for_sample_indices(self.samples, self.split['train'])
        self.val_positions = shared.positions_for_sample_indices(self.samples, self.split['validation'])
        self.stats = {}
        normalization_files = {}
        for variant in VARIANTS:
            stats = fit_normalization_stats(self.samples, self.train_positions, variant)
            for key, value in vars(stats).items():
                assert np.isfinite(value).all(), key
                if 'scale' in key:
                    assert (value > 0).all(), key
            self.stats[variant] = stats
            path = ROOT / 'normalization' / f'{variant}.json'
            record = dict(source='current_training_split_only', sample_count=13000,
                          sample_indices=self.split['train'], statistics=stats.checkpoint_dict())
            if path.exists() and shared.read(path) != record:
                raise ValueError(f'Existing normalization differs: {variant}')
            if not path.exists():
                shared.write(path, record)
            normalization_files[variant] = dict(path=str(path.relative_to(IDEA)), sha256=shared.sha256_file(path))
        code = ('phase2_gpu_training.py', 'gpu_training_runtime.py', 'model_train.py', 'model.py',
                'expansion_training.py', 'expansion_training_balanced.py', 'phase2_scratch_training.py',
                'data_processing.py', 'analytic_layer.py', 'orbit_core.py', 'experiment_setup.py',
                'phase2_gpu_evaluate.py', 'phase2_gpu_figures.py', 'model_evaluate.py',
                'phase2_experiment.py', 'gpu_training_preflight.py', 'gpu_training_benchmark.py',
                'phase2_scratch_curve_preview.py')
        self.protocol = dict(schema_version=1, experiment_id=ROOT.name,
                             dataset=self.manifest['dataset'], dataset_path=str(shared.DATA.relative_to(IDEA)),
                             split_id=self.manifest['split_id'], split_manifest=str(shared.SPLIT.relative_to(IDEA)),
                             split_manifest_sha256=shared.sha256_file(shared.SPLIT),
                             dataset_freeze_manifest_sha256=shared.sha256_file(shared.FROZEN / 'freeze_manifest.json'),
                             sample_counts=counts, training=self.training, model_kwargs=MODEL_KWARGS,
                             neural_runs=[dict(run_id=f'{v}__seed{s}', model_variant=v, seed=s) for v in VARIANTS for s in SEEDS],
                             deterministic_baseline='counterpart_type_prior', initial_checkpoint=None,
                             normalization_files=normalization_files, raw_input_cache=dict(path=str(CACHE.relative_to(IDEA)), **cache_meta),
                             gpu_benchmark=dict(path=str(BENCHMARK.relative_to(IDEA)), sha256=shared.sha256_file(BENCHMARK)),
                             code_sha256={name: shared.sha256_file(IDEA / 'scripts' / name) for name in code},
                             evaluation=dict(test_used_for_training_or_selection=False,
                                             test_policy='only after all 15 selected checkpoints are sealed',
                                             metric_std='sample SD across five seeds, ddof=1',
                                             confusion='per-seed counts and row fractions, then elementwise five-seed mean and SD'),
                             required_outputs=['curves_mean_plus_minus_sd', 'four_model_confusion_heatmaps',
                                               'per_seed_probabilities', 'metric_mean_and_sd_tables'],
                             user_authorization='Only Phase 2 prior/factual-only/no-analytic/full; GPU tuning before formal training; five seeds; SD bands and confusion heatmaps required.')
        protocol_path = ROOT / 'training_protocol.json'
        if protocol_path.exists() and shared.read(protocol_path) != self.protocol:
            raise ValueError('Formal protocol changed; existing runs must not be overwritten.')
        if not protocol_path.exists():
            shared.write(protocol_path, self.protocol)
        self.protocol_hash = shared.sha256_file(protocol_path)
        print(f'Frozen Phase 2: 15 random-init runs, batch={self.batch_size}, 50 epochs; no additional baselines.', flush=True)

    def prepare_variant(self, variant):
        model = shared.build_intention_model(variant, **MODEL_KWARGS)
        keys = tuple(model.model_input_keys)
        train_data = shared.CachedInputs(self.samples, self.train_positions, self.stats[variant], keys)
        val_data = shared.CachedInputs(self.samples, self.val_positions, self.stats[variant], keys)
        train = GpuBatches(train_data, self.device, self.batch_size)
        validation = GpuBatches(val_data, self.device, self.batch_size)
        del model, train_data, val_data
        gc.collect()
        return train, validation

    def preflight(self):
        """Test actual full-cache memory, one update and save/reload for each variant."""
        from gpu_training_preflight import verify_tail_equivalence
        tail_check = verify_tail_equivalence(self.samples, self.train_positions, self.stats['full'])
        reports = []
        for variant in VARIANTS:
            shared.write(ROOT / 'status.json', dict(status='gpu_preflight', variant=variant, pid=os.getpid(), updated_at=shared.now()))
            train, validation = self.prepare_variant(variant)
            shared.set_random_seed(7)
            self.configure_backend()
            model = shared.build_intention_model(variant, **MODEL_KWARGS).to(self.device)
            initial_digest = state_digest(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.training['learning_rate'], weight_decay=self.training['weight_decay'])
            scaler = torch.amp.GradScaler('cuda')
            torch.cuda.reset_peak_memory_stats()
            train_batch = next(iter(train))
            val_batch = next(iter(validation))
            trained = gpu_run_epoch(model, [train_batch], self.device, optimizer, scaler)
            checked = gpu_run_epoch(model, [val_batch], self.device)
            assert state_digest(model) != initial_digest
            path = ROOT / 'preflight' / f'{variant}_one_step.pt'
            shared.save_torch(path, dict(model_state_dict=model.state_dict()))
            restored = shared.build_intention_model(variant, **MODEL_KWARGS).to(self.device)
            restored.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)['model_state_dict'], strict=True)
            assert gpu_run_epoch(restored, [val_batch], self.device) == checked
            allocated = torch.cuda.max_memory_allocated() / 2**20
            reserved = torch.cuda.max_memory_reserved() / 2**20
            if reserved > 12000:
                raise RuntimeError(f'{variant} exceeds full-cache memory budget: {reserved:.1f} MiB reserved.')
            reports.append(dict(model_variant=variant, batch_size=self.batch_size,
                                gpu_input_storage_mib=(train.storage_bytes + validation.storage_bytes) / 2**20,
                                peak_allocated_mib=allocated, peak_reserved_mib=reserved,
                                train=asdict(trained), validation=asdict(checked), save_reload_equal=True))
            del train, validation, model, restored, optimizer, scaler, train_batch, val_batch
            gc.collect()
            torch.cuda.empty_cache()
        shared.write(ROOT / 'preflight/checks.json', dict(status='passed', completed_at=shared.now(),
                     batch32_tail_equivalence=tail_check, variants=reports, test_samples_loaded=False,
                     protocol_sha256=self.protocol_hash))
        print('All three variants passed GPU full-cache/update/reload checks.', flush=True)

    def run_one(self, spec, train_columns, val_columns):
        variant, seed, run_id = spec['model_variant'], spec['seed'], spec['run_id']
        folder = ROOT / 'neural' / run_id
        folder.mkdir(parents=True, exist_ok=True)
        log_path, initial_path, best_path, last_path = (folder / n for n in ('training.json', 'initial.pt', 'best.pt', 'last.pt'))
        if log_path.exists():
            old = shared.read(log_path)
            assert old['experiment']['protocol_sha256'] == self.protocol_hash
            if old['status'] == 'completed':
                assert old['history'][-1]['epoch'] == EPOCHS
                for name, path in (('best_checkpoint', best_path), ('last_checkpoint', last_path), ('initial_checkpoint', initial_path)):
                    assert shared.sha256_file(path) == old[name]['sha256']
                return old
        shared.set_random_seed(seed)
        self.configure_backend()
        model = shared.build_intention_model(variant, **MODEL_KWARGS).to(self.device)
        generator = torch.Generator().manual_seed(seed)
        train = train_columns.with_sampling(self.batch_size, shuffle=True, generator=generator)
        train_eval = train_columns.with_sampling(self.batch_size)
        validation = val_columns.with_sampling(self.batch_size)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.training['learning_rate'], weight_decay=self.training['weight_decay'])
        scaler = torch.amp.GradScaler('cuda')
        initial_digest = state_digest(model)
        experiment = dict(experiment_id=ROOT.name, **spec, initialization='random', initial_checkpoint=None,
                          model_kwargs=MODEL_KWARGS, model_input_keys=list(model.model_input_keys),
                          initial_state_sha256=initial_digest, protocol_sha256=self.protocol_hash,
                          dataset_sha256=self.protocol['dataset']['sha256'], dataset_version=self.protocol['dataset']['version'],
                          split_id=self.protocol['split_id'], split_manifest_sha256=self.protocol['split_manifest_sha256'],
                          normalization_source='current_13000_train_only', runtime_environment=shared.runtime_environment(self.device),
                          test_samples_loaded=False)
        history, best_epoch, best_loss, begin = [], None, None, 1
        started_at = shared.now()

        def payload(epoch):
            return dict(schema_version=1, epoch=epoch, model_state_dict=model.state_dict(),
                        optimizer_state_dict=optimizer.state_dict(), scaler_state_dict=scaler.state_dict(),
                        normalization=self.stats[variant].checkpoint_dict(), split_sample_indices=self.split,
                        config={**self.training, 'random_seed': seed, 'model_variant': variant}, experiment=experiment,
                        train_metrics=history[-1]['train'], validation_metrics=history[-1]['validation'],
                        history=history, best_epoch=best_epoch, best_validation_loss=best_loss, started_at=started_at,
                        rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                                 cuda=torch.cuda.get_rng_state_all(), loader=generator.get_state()))

        if last_path.exists():
            last = torch.load(last_path, map_location='cpu', weights_only=False)
            assert last['experiment']['protocol_sha256'] == self.protocol_hash
            assert last['experiment']['initial_state_sha256'] == initial_digest
            assert last['normalization'] == self.stats[variant].checkpoint_dict()
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
            if best_epoch is not None:
                # A crash can occur between writing a new best and last.pt.
                # Restore the candidate named by the committed last checkpoint.
                candidate = folder / 'selection_candidates' / f'epoch_{best_epoch:02d}.pt'
                if not candidate.exists():
                    raise ValueError('Missing committed best checkpoint candidate.')
                committed = torch.load(candidate, map_location='cpu', weights_only=False)
                shared.save_torch(best_path, committed)
                del committed
        else:
            start = time.perf_counter()
            initial_train = gpu_run_epoch(model, train_eval, self.device)
            initial_val = gpu_run_epoch(model, validation, self.device)
            history.append(dict(epoch=0, phase='random_initialization_evaluation', train=asdict(initial_train),
                                validation=asdict(initial_val), learning_rate=0.0, epoch_seconds=time.perf_counter() - start))
            shared.save_torch(initial_path, payload(0))
            shared.save_torch(last_path, payload(0))
        log = dict(schema_version=1, status='running', experiment=experiment, training_config=self.training,
                   started_at=started_at, history=history, best_epoch=best_epoch, best_validation_loss=best_loss)
        shared.write(log_path, log)
        for epoch in range(begin, EPOCHS + 1):
            shared.write(ROOT / 'status.json', dict(status='training', pid=os.getpid(), run_id=run_id,
                         epoch_in_progress=epoch, epochs_completed=epoch - 1, maximum_epochs=EPOCHS,
                         completed_runs=len(self.results), expected_runs=15, updated_at=shared.now()))
            start = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            train_metrics = gpu_run_epoch(model, train, self.device, optimizer, scaler)
            val_metrics = gpu_run_epoch(model, validation, self.device)
            history.append(dict(epoch=epoch, phase='from_scratch_18000', train=asdict(train_metrics),
                                validation=asdict(val_metrics), learning_rate=optimizer.param_groups[0]['lr'],
                                epoch_seconds=time.perf_counter() - start,
                                peak_gpu_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                                peak_gpu_reserved_mib=torch.cuda.max_memory_reserved() / 2**20))
            if best_loss is None or val_metrics.loss < best_loss:
                best_epoch, best_loss = epoch, val_metrics.loss
                candidate = folder / 'selection_candidates' / f'epoch_{epoch:02d}.pt'
                checkpoint = payload(epoch)
                shared.save_torch(candidate, checkpoint)
                shared.save_torch(best_path, checkpoint)
            shared.save_torch(last_path, payload(epoch))
            log.update(history=history, best_epoch=best_epoch, best_validation_loss=best_loss, updated_at=shared.now())
            shared.write(log_path, log)
            print(f'{run_id} epoch {epoch:02d}/50 train={train_metrics.loss:.6f} val={val_metrics.loss:.6f} best={best_epoch} seconds={history[-1]["epoch_seconds"]:.1f}', flush=True)
        assert [row['epoch'] for row in history] == list(range(EPOCHS + 1))
        assert best_epoch == min(history[1:], key=lambda row: row['validation']['loss'])['epoch']
        log.update(status='completed', completed_at=shared.now(), stopping_reason='completed_50_epochs',
                   initial_checkpoint=dict(path=str(initial_path.relative_to(IDEA)), sha256=shared.sha256_file(initial_path)),
                   best_checkpoint=dict(path=str(best_path.relative_to(IDEA)), sha256=shared.sha256_file(best_path)),
                   last_checkpoint=dict(path=str(last_path.relative_to(IDEA)), sha256=shared.sha256_file(last_path)))
        shared.write(log_path, log)
        return log

    def summarize(self):
        curves = []
        for variant in VARIANTS:
            logs = [r for r in self.results if r['experiment']['model_variant'] == variant]
            if not logs:
                continue
            for epoch in range(EPOCHS + 1):
                for partition in ('train', 'validation'):
                    record = dict(model_variant=variant, epoch=epoch, partition=partition,
                                  n_seeds=len(logs), seeds=[r['experiment']['seed'] for r in logs], std_ddof=1, metrics={})
                    for key in ('loss', 'accuracy', 'balanced_accuracy', 'brier_score'):
                        values = np.array([r['history'][epoch][partition][key] for r in logs], dtype=np.float64)
                        record['metrics'][key] = dict(mean=float(values.mean()), std=float(values.std(ddof=1)) if len(values) > 1 else None)
                    curves.append(record)
        shared.write(ROOT / 'curve_statistics.json', dict(status='completed' if len(self.results) == 15 else 'partial',
                     std_ddof=1, rows=curves, protocol_sha256=self.protocol_hash))
        shared.write(ROOT / 'training_summary.json', dict(status='completed' if len(self.results) == 15 else 'running',
                     completed_neural_runs=len(self.results), expected_neural_runs=15,
                     runs=[dict(run_id=r['experiment']['run_id'], best_epoch=r['best_epoch'], best_validation_loss=r['best_validation_loss'],
                                checkpoint=r['best_checkpoint']) for r in self.results],
                     protocol_sha256=self.protocol_hash, test_evaluation_performed=False, updated_at=shared.now()))

    def run(self):
        self.preflight()
        self.results = []
        for variant in VARIANTS:
            train, validation = self.prepare_variant(variant)
            for spec in self.protocol['neural_runs']:
                if spec['model_variant'] == variant:
                    folder = ROOT / 'neural' / spec['run_id']
                    log_path = folder / 'training.json'
                    if log_path.exists() and shared.read(log_path)['status'] == 'completed':
                        self.results.append(self.run_one(spec, train, validation))
                    else:
                        # Keep separate telemetry files when resuming a run.
                        with GpuTelemetry(folder / f'gpu_telemetry_{time.time_ns()}.json') as telemetry:
                            self.results.append(self.run_one(spec, train, validation))
                        shared.write(folder / 'gpu_telemetry_summary.json', telemetry.summary())
                    self.summarize()
            del train, validation
            gc.collect()
            torch.cuda.empty_cache()
        for name, digest in self.protocol['code_sha256'].items():
            assert shared.sha256_file(IDEA / 'scripts' / name) == digest, name
        shared.write(ROOT / 'status.json', dict(status='training_completed', completed_runs=15, expected_runs=15, updated_at=shared.now()))


def main():
    import msvcrt
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'run.lock').open('a+b') as lock:
        if lock.tell() == 0:
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            trainer = Phase2Training()
            trainer.run()
            del trainer
            gc.collect()
            torch.cuda.empty_cache()
            for stage, executable, filename in (('evaluating', sys.executable, 'phase2_gpu_evaluate.py'),
                                                 ('plotting', str(PLOT_PYTHON), 'phase2_gpu_figures.py')):
                shared.write(ROOT / 'status.json', dict(status=stage, completed_runs=15, expected_runs=15, updated_at=shared.now()))
                subprocess.run([executable, '-X', 'utf8', str(IDEA / 'scripts' / filename)], check=True, cwd=str(IDEA))
            shared.write(ROOT / 'status.json', dict(status='completed', completed_runs=15, expected_runs=15,
                         figures_generated=True, sd_curves_generated=True, confusion_heatmaps_generated=True,
                         completed_at=shared.now()))
        except Exception:
            error = dict(status='failed', time=shared.now(), traceback=traceback.format_exc())
            shared.write(ROOT / 'failure.json', error)
            shared.write(ROOT / 'status.json', error)
            raise
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == '__main__':
    main()
