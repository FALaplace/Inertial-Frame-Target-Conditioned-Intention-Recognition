"""Check deterministic CPU-loaded/GPU-resident equivalence with a real tail batch."""
from __future__ import annotations

import expansion_training as shared

from dataclasses import asdict
import gc
import math
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from expansion_training_balanced import balanced_run_epoch
from gpu_training_runtime import GpuBatches, gpu_run_epoch
from phase2_scratch_training import MODEL_KWARGS


def _snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_snapshot(item) for item in value)
    return value


def _equal(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    return left == right


def _rng_state(generator=None):
    result = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                  cuda=torch.cuda.get_rng_state_all())
    if generator is not None:
        result['loader'] = generator.get_state()
    return _snapshot(result)


def verify_tail_equivalence(samples, train_positions, stats):
    """Use the caller's Train-only inputs/stats; return evidence without writing files.

    Both paths run Full, seed 7, batch 32 and two epochs on 257 samples.
    Deterministic execution is limited to this check; caller RNG/backend settings
    are restored even on failure. No benchmark or formal checkpoint is reused.
    """
    if not torch.cuda.is_available():
        raise RuntimeError('Tail equivalence requires CUDA.')
    positions = list(train_positions[:257])
    if len(positions) != 257 or len(set(positions)) != 257:
        raise ValueError('Tail equivalence needs 257 distinct training positions.')
    device = torch.device('cuda')
    caller_rng = _rng_state()
    backend = (torch.are_deterministic_algorithms_enabled(),
               torch.is_deterministic_algorithms_warn_only_enabled(),
               torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    results = []
    try:
        model = shared.build_intention_model('full', **MODEL_KWARGS)
        dataset = shared.CachedInputs(samples, positions, stats, tuple(model.model_input_keys))
        del model
        for index, row in enumerate(dataset.rows):
            row['_preflight_index'] = torch.tensor(index, dtype=torch.int64)
        for gpu in (False, True):
            shared.set_random_seed(7)
            model = shared.build_intention_model('full', **MODEL_KWARGS).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            scaler = torch.amp.GradScaler('cuda')
            generator = torch.Generator().manual_seed(7)
            if gpu:
                batches = GpuBatches(dataset, device, 32, True, generator)
            else:
                batches = DataLoader(dataset, batch_size=32, shuffle=True, generator=generator,
                                     num_workers=0, pin_memory=True)
            trajectory = []
            for epoch in (1, 2):
                order, batch_sizes = [], []

                def observed_batches():
                    for batch in batches:
                        indices = batch['_preflight_index'].cpu().tolist()
                        order.extend(indices)
                        batch_sizes.append(len(indices))
                        yield batch

                run_epoch = gpu_run_epoch if gpu else balanced_run_epoch
                metrics = asdict(run_epoch(model, observed_batches(), device, optimizer, scaler))
                if batch_sizes != [32] * 8 + [1] or sorted(order) != list(range(257)):
                    raise AssertionError('Tail batch, sample coverage or sample uniqueness changed.')
                if sum(metrics['class_support']) != 257:
                    raise AssertionError('Metric sample count differs from the 257 input samples.')
                trajectory.append(dict(epoch=epoch, metrics=metrics, order=order, batch_sizes=batch_sizes,
                                       weights=_snapshot(model.state_dict()), optimizer=_snapshot(optimizer.state_dict()),
                                       scaler=_snapshot(scaler.state_dict()), rng=_rng_state(generator)))
            results.append(trajectory)
            del model, optimizer, scaler, batches
            gc.collect()
            torch.cuda.empty_cache()
        checks = []
        for left, right in zip(*results):
            differences = [abs(left['metrics'][key] - right['metrics'][key])
                           for key in ('loss', 'accuracy', 'brier_score', 'balanced_accuracy')]
            rng_equal = {key: _equal(left['rng'][key], right['rng'][key]) for key in left['rng']}
            check = dict(epoch=left['epoch'], weights_bitwise_equal=_equal(left['weights'], right['weights']),
                         optimizer_equal=_equal(left['optimizer'], right['optimizer']),
                         scaler_equal=_equal(left['scaler'], right['scaler']), rng_equal=rng_equal,
                         batch_order_equal=left['order'] == right['order'],
                         class_counts_equal=all(left['metrics'][key] == right['metrics'][key]
                                                for key in ('class_support', 'class_correct')),
                         metric_max_abs_difference=max(differences))
            check['passed'] = (all(check[key] for key in ('weights_bitwise_equal', 'optimizer_equal',
                                'scaler_equal', 'batch_order_equal', 'class_counts_equal'))
                               and all(rng_equal.values())
                               and all(math.isfinite(value) and value < 1e-12 for value in differences))
            checks.append(check)
        report = dict(passed=all(check['passed'] for check in checks), sample_count=257, batch_size=32,
                      epochs=2, batches_per_epoch=9, tail_batch_size=1, metric_absolute_tolerance=1e-12,
                      deterministic_algorithms=True, cudnn_deterministic=True, cudnn_benchmark=False,
                      selected_train_sample_indices=[int(samples[position]['sample_index']) for position in positions],
                      checks=checks, test_data_used=False)
        if not report['passed']:
            raise AssertionError(f'Two-epoch tail equivalence failed: {report}')
        return report
    finally:
        random.setstate(caller_rng['python'])
        np.random.set_state(caller_rng['numpy'])
        torch.set_rng_state(caller_rng['torch'])
        torch.cuda.set_rng_state_all(caller_rng['cuda'])
        torch.use_deterministic_algorithms(backend[0], warn_only=backend[1])
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = backend[2:]
