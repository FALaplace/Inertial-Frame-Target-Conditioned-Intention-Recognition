"""Measure real Train-only batches before authorizing the final GPU run."""
from __future__ import annotations

import expansion_training as shared

from dataclasses import asdict
import gc
import json
from pathlib import Path
import pickle
import statistics
import subprocess
import threading
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from expansion_training_balanced import balanced_run_epoch
from gpu_training_runtime import GpuBatches, gpu_run_epoch
from phase2_scratch_training import MODEL_KWARGS, CACHE, state_digest

ROOT = shared.IDEA / 'experiment_data/gpu_tuning_18000_v1'
DEVICE = torch.device('cuda')
TRAIN_SAMPLES = 2048
REPEATS = 2


class GpuTelemetry:
    def __init__(self, path):
        self.path = path
        self.rows = []

    def __enter__(self):
        self.process = subprocess.Popen(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,power.draw',
                                        '--format=csv,noheader,nounits', '-lms', '500'],
                                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
        def collect():
            for line in self.process.stdout:
                try:
                    util, memory, power = map(float, line.strip().split(','))
                    self.rows.append(dict(timestamp=time.time(), gpu_percent=util, memory_mib=memory, power_w=power))
                except ValueError:
                    pass
        self.thread = threading.Thread(target=collect, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.process.terminate()
        self.process.wait(timeout=10)
        self.thread.join(timeout=5)
        self.process.stdout.close()
        shared.write(self.path, self.rows)

    def summary(self):
        return dict(sample_count=len(self.rows),
                    mean_gpu_percent=statistics.mean(r['gpu_percent'] for r in self.rows) if self.rows else None,
                    peak_board_memory_mib=max((r['memory_mib'] for r in self.rows), default=None))


def new_model():
    shared.set_random_seed(7)
    model = shared.build_intention_model('full', **MODEL_KWARGS).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    return model, optimizer, torch.amp.GradScaler('cuda')


def loader(dataset, gpu, batch_size=32, generator=None, shuffle=True):
    if gpu:
        return GpuBatches(dataset, DEVICE, batch_size, shuffle, generator)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator,
                      pin_memory=True, num_workers=0)


def equivalence(dataset):
    """Two shuffled epochs include the RNG and order change at an epoch boundary."""
    results = []
    for gpu in (False, True):
        model, optimizer, scaler = new_model()
        generator = torch.Generator().manual_seed(7)
        batches = loader(dataset, gpu, generator=generator)
        run_epoch = gpu_run_epoch if gpu else balanced_run_epoch
        rows = []
        for _ in range(2):
            rows.append(asdict(run_epoch(model, batches, DEVICE, optimizer, scaler)))
        torch.cuda.synchronize()
        results.append(dict(metrics=rows, weights={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                            optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state_all(), loader_rng=generator.get_state()))
        del model, optimizer, scaler, batches
        gc.collect()
        torch.cuda.empty_cache()
    left, right = results
    weight_equal = all(torch.equal(left['weights'][k], right['weights'][k]) for k in left['weights'])
    metric_delta = max(abs(a[k] - b[k]) for a, b in zip(left['metrics'], right['metrics'])
                       for k in ('loss', 'accuracy', 'brier_score', 'balanced_accuracy'))
    counts_equal = all(a[k] == b[k] for a, b in zip(left['metrics'], right['metrics'])
                       for k in ('class_support', 'class_correct'))
    rng_equal = (torch.equal(left['torch_rng'], right['torch_rng'])
                 and torch.equal(left['loader_rng'], right['loader_rng'])
                 and all(torch.equal(a, b) for a, b in zip(left['cuda_rng'], right['cuda_rng'])))
    optimizer_equal = all(torch.equal(a, right['optimizer']['state'][idx][key]) if isinstance(a, torch.Tensor)
                          else a == right['optimizer']['state'][idx][key]
                          for idx, state in left['optimizer']['state'].items() for key, a in state.items())
    report = dict(epochs=2, batch_size=32, sample_count=len(dataset), weights_bitwise_equal=weight_equal,
                  metric_max_abs_difference=metric_delta, class_counts_equal=counts_equal,
                  rng_equal=rng_equal, optimizer_equal=optimizer_equal,
                  passed=weight_equal and metric_delta < 1e-12 and counts_equal and rng_equal and optimizer_equal)
    shared.write(ROOT / 'equivalence.json', report)
    if not report['passed']:
        raise AssertionError(f'Optimized batch32 must match the original update: {report}')
    print('Two-epoch batch32 equivalence passed.', flush=True)
    return report


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable.')
    manifest, split = shared.load_split_indices(shared.SPLIT)
    shared.verify_manifest_dataset(manifest, shared.DATA)
    cache_meta = shared.read(CACHE.with_suffix('.json'))
    assert shared.sha256_file(CACHE) == cache_meta['sha256']
    assert cache_meta['identity']['dataset_sha256'] == manifest['dataset']['sha256']
    assert cache_meta['identity']['split_sha256'] == shared.sha256_file(shared.SPLIT)
    for name, digest in cache_meta['identity']['code_sha256'].items():
        assert shared.sha256_file(shared.IDEA / 'scripts' / name) == digest
    with CACHE.open('rb') as stream:
        samples = pickle.load(stream)
    positions = shared.positions_for_sample_indices(samples, split['train'])
    # Pilot statistics were fitted only on this exact 13000-row Train partition.
    stats_record = shared.read(shared.IDEA / 'experiment_data/scratch_18000_full_seed7_v1/normalization.json')
    assert stats_record['sample_indices'] == split['train']
    stats = shared.NormalizationStats.from_checkpoint_dict(stats_record['statistics'])
    selected = np.random.default_rng(20260915).choice(positions, TRAIN_SAMPLES, replace=False).tolist()
    selected_ids = [int(samples[position]['sample_index']) for position in selected]
    model = shared.build_intention_model('full', **MODEL_KWARGS)
    dataset = shared.CachedInputs(samples, selected, stats, tuple(model.model_input_keys))
    small = shared.CachedInputs(samples, selected[:256], stats, tuple(model.model_input_keys))
    del samples, model
    gc.collect()
    eq = equivalence(small)
    measurements = []
    for name, gpu, batch_size in (('reference_cpu32', False, 32), ('gpu32', True, 32),
                                  ('gpu64', True, 64), ('gpu128', True, 128), ('gpu256', True, 256)):
        run_epoch = gpu_run_epoch if gpu else balanced_run_epoch
        for repeat in range(REPEATS):
            model, optimizer, scaler = new_model()
            warm = loader(small, gpu, batch_size, torch.Generator().manual_seed(7))
            run_epoch(model, warm, DEVICE, optimizer, scaler)
            del model, optimizer, scaler, warm
            model, optimizer, scaler = new_model()
            batches = loader(dataset, gpu, batch_size, torch.Generator().manual_seed(7))
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            with GpuTelemetry(ROOT / f'{name}_repeat{repeat}_telemetry.json') as telemetry:
                start = time.perf_counter()
                metrics = run_epoch(model, batches, DEVICE, optimizer, scaler)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
            result = dict(candidate=name, gpu_resident=gpu, batch_size=batch_size, repeat=repeat,
                          sample_count=len(dataset), seconds=elapsed, samples_per_second=len(dataset) / elapsed,
                          peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                          peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                          telemetry=telemetry.summary(), metrics=asdict(metrics))
            measurements.append(result)
            shared.write(ROOT / 'measurements.json', measurements)
            print(f'{name} repeat{repeat}: {elapsed:.2f}s, {result["samples_per_second"]:.1f} samples/s, GPU {result["telemetry"]["mean_gpu_percent"]}, allocated {result["peak_allocated_mib"]:.0f}MiB', flush=True)
            del model, optimizer, scaler, batches
            gc.collect()
            torch.cuda.empty_cache()
    summary = {}
    for name in dict.fromkeys(r['candidate'] for r in measurements):
        rows = [r for r in measurements if r['candidate'] == name]
        summary[name] = dict(median_seconds=statistics.median(r['seconds'] for r in rows),
                             median_samples_per_second=statistics.median(r['samples_per_second'] for r in rows),
                             mean_gpu_percent=statistics.mean(r['telemetry']['mean_gpu_percent'] for r in rows),
                             peak_allocated_mib=max(r['peak_allocated_mib'] for r in rows), batch_size=rows[0]['batch_size'])
    # Prefer the highest measured throughput within the GPU memory budget.
    # A changed batch size means every formal seed is trained under a new protocol.
    feasible = [name for name in summary if name.startswith('gpu') and summary[name]['peak_allocated_mib'] < 12000]
    selected = min(feasible, key=lambda name: summary[name]['median_seconds'])
    speedup = summary['reference_cpu32']['median_seconds'] / summary[selected]['median_seconds']
    report = dict(status='passed', completed_at=shared.now(), selected_candidate=selected,
                  selected_batch_size=summary[selected]['batch_size'],
                  selection_policy='highest repeated median throughput below 12000 MiB allocated; retrain all formal seeds if batch changes',
                  batch32_equivalence=eq, speedup_vs_reference=speedup, candidates=summary,
                  dataset_sha256=manifest['dataset']['sha256'], split_sha256=shared.sha256_file(shared.SPLIT),
                  selected_train_sample_indices=selected_ids,
                  test_data_used=False, runtime=shared.runtime_environment(DEVICE),
                  code_sha256={name: shared.sha256_file(shared.IDEA / 'scripts' / name)
                               for name in ('gpu_training_runtime.py', 'gpu_training_benchmark.py')})
    shared.write(ROOT / 'benchmark_report.json', report)
    print(json.dumps(summary, indent=2), flush=True)
    print(f'Selected {selected}: {speedup:.3f}x reference throughput. Benchmark complete; no formal training started.', flush=True)


if __name__ == '__main__':
    main()
