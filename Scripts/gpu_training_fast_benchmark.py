"""Measure sustained fast cuDNN throughput on the complete Train partition."""
import expansion_training as shared
from dataclasses import asdict
import gc
import pickle
import statistics
import time
import torch
from gpu_training_benchmark import GpuTelemetry, new_model, CACHE, MODEL_KWARGS
from gpu_training_runtime import GpuBatches, gpu_run_epoch

ROOT = shared.IDEA / 'experiment_data/gpu_tuning_18000_v1/fast_kernels'


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    manifest, split = shared.load_split_indices(shared.SPLIT)
    shared.verify_manifest_dataset(manifest, shared.DATA)
    meta = shared.read(CACHE.with_suffix('.json'))
    assert shared.sha256_file(CACHE) == meta['sha256']
    assert meta['identity']['dataset_sha256'] == manifest['dataset']['sha256']
    assert meta['identity']['split_sha256'] == shared.sha256_file(shared.SPLIT)
    with CACHE.open('rb') as stream:
        samples = pickle.load(stream)
    positions = shared.positions_for_sample_indices(samples, split['train'])
    record = shared.read(shared.IDEA / 'experiment_data/scratch_18000_full_seed7_v1/normalization.json')
    assert record['sample_indices'] == split['train']
    stats = shared.NormalizationStats.from_checkpoint_dict(record['statistics'])
    model, _, _ = new_model()
    dataset = shared.CachedInputs(samples, positions, stats, tuple(model.model_input_keys))
    del samples, model
    gc.collect()
    results = []
    for batch_size in (32, 64, 128, 256, 512):
        model, optimizer, scaler = new_model()
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        batches = GpuBatches(dataset, torch.device('cuda'), batch_size, True, torch.Generator().manual_seed(7))
        gpu_run_epoch(model, batches, torch.device('cuda'), optimizer, scaler)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        durations = []
        with GpuTelemetry(ROOT / f'batch{batch_size}_telemetry.json') as telemetry:
            while len(durations) < 3 or sum(durations) < 12:
                start = time.perf_counter()
                metrics = gpu_run_epoch(model, batches, torch.device('cuda'), optimizer, scaler)
                torch.cuda.synchronize()
                durations.append(time.perf_counter() - start)
        result = dict(batch_size=batch_size, durations=durations,
                      median_seconds=statistics.median(durations),
                      samples_per_second=13000 / statistics.median(durations),
                      train_sample_count=len(dataset), test_data_used=False,
                      peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                      peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                      gpu_input_storage_mib=batches.storage_bytes/2**20,
                      telemetry=telemetry.summary(), last_train_metrics=asdict(metrics))
        results.append(result)
        shared.write(ROOT/'results.json', results)
        print(result, flush=True)
        del model, optimizer, scaler, batches
        gc.collect()
        torch.cuda.empty_cache()
    shared.write(ROOT/'provenance.json', dict(completed_at=shared.now(), dataset_sha256=manifest['dataset']['sha256'],
                 split_sha256=shared.sha256_file(shared.SPLIT), test_data_used=False,
                 runtime=shared.runtime_environment(torch.device('cuda')),
                 backend=dict(cudnn_benchmark=True,cudnn_deterministic=False,deterministic_algorithms=False),
                 code_sha256={name:shared.sha256_file(shared.IDEA/'scripts'/name) for name in
                              ('gpu_training_fast_benchmark.py','gpu_training_runtime.py','gpu_training_benchmark.py')}))


if __name__ == '__main__':
    main()
