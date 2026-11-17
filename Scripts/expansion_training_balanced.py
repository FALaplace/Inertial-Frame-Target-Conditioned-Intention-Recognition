"""Continue the frozen training matrix with complete balanced-accuracy histories.

Run this file directly in an IDE. The original trainer and its outputs are
preserved; the recording revision has a separate protocol and output directory.
"""
from __future__ import annotations

import expansion_training as base  # Set the original runtime environment first.

from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
import traceback

import torch

ORIGINAL_ROOT = base.ROOT
RECORDING_ROOT = base.IDEA / 'experiment_data/expansion_18000_balanced_v1'
ORIGINAL_RUN_EPOCH = base.run_epoch


@dataclass(frozen=True)
class BalancedEpochMetrics:
    loss: float
    accuracy: float
    brier_score: float
    balanced_accuracy: float
    class_support: list[int]
    class_correct: list[int]


def balanced_run_epoch(model, data_loader, device, optimizer=None, scaler=None):
    """Observe existing forward passes without changing loss, RNG, or updates.

Accumulate class counts across the whole epoch, not mean batchwise recalls.
Only classes present in the reference argmax labels enter balanced accuracy,
matching the Phase 2 evaluation definition. Training uses online predictions;
validation and epoch zero use eval-mode predictions, as in the original logs.
"""
    counts = torch.zeros((8, 8), dtype=torch.int64)
    reference = None

    def batches():
        nonlocal reference
        for batch in data_loader:
            reference = batch['y'].detach().argmax(dim=-1).cpu()
            yield batch

    def observe(_model, _inputs, outputs):
        predicted = outputs['probabilities'].detach().argmax(dim=-1).cpu()
        counts.add_(torch.bincount(reference * 8 + predicted, minlength=64).reshape(8, 8))

    handle = model.register_forward_hook(observe)
    try:
        metrics = ORIGINAL_RUN_EPOCH(model, batches(), device, optimizer, scaler)
    finally:
        handle.remove()
    support = counts.sum(dim=1)
    correct = counts.diag()
    observed = support > 0
    balanced = (correct[observed].double() / support[observed]).mean().item()
    return BalancedEpochMetrics(**asdict(metrics), balanced_accuracy=balanced,
                               class_support=support.tolist(), class_correct=correct.tolist())


def prepare_protocol():
    """Freeze an additive recording revision; keep data and training unchanged."""
    old_protocol_path = ORIGINAL_ROOT / 'training_protocol.json'
    original = base.read(old_protocol_path)
    base.verify_protocol(original)
    protocol_path = RECORDING_ROOT / 'training_protocol.json'
    recording = {
        'metric': 'balanced_accuracy',
        'definition': 'mean per-class recall over reference argmax classes present in the partition',
        'aggregation': 'whole-epoch counts, never mean of batch scores',
        'partitions': ['train', 'validation'],
        'epoch_zero': 'eval-mode initial old weights on the frozen 18000 split',
        'train': 'online forward passes with dropout, before each optimizer update',
        'validation': 'eval mode after each training epoch',
        'additional_saved_fields': ['class_support', 'class_correct'],
        'selection_metric_unchanged': 'validation_soft_nll',
        'figures_generated': False,
        'history_policy': 'new coherent runs from original initial weights; never splice old and new histories',
    }
    if protocol_path.exists():
        protocol = base.read(protocol_path)
        base.verify_protocol(protocol)
        if (protocol.get('predecessor_protocol_sha256') != base.sha256_file(old_protocol_path)
                or protocol.get('metric_recording') != recording):
            raise ValueError('Recording revision differs from the existing frozen revision.')
    else:
        protocol = {
            **original,
            'experiment_id': RECORDING_ROOT.name,
            'created_at': base.now(),
            'user_authorization': 'User approved complete per-epoch balanced-accuracy curves and necessary reruns; no figures yet.',
            'predecessor_protocol': str(old_protocol_path.relative_to(base.IDEA)),
            'predecessor_protocol_sha256': base.sha256_file(old_protocol_path),
            'metric_recording': recording,
            'code_sha256': {**original['code_sha256'], Path(__file__).name: base.sha256_file(Path(__file__))},
        }
        base.write(protocol_path, protocol)
    for field in ('training', 'neural_runs', 'dataset', 'split_manifest_sha256'):
        if protocol[field] != original[field]:
            raise ValueError(f'Recording-only revision changed {field}.')
    return protocol


def share_verified_cache():
    """Reuse the immutable prepared inputs; the original loader verifies hashes."""
    destination = RECORDING_ROOT / 'cache'
    destination.mkdir(parents=True, exist_ok=True)
    for name in ('train_validation_inputs.pkl', 'train_validation_inputs.json'):
        source = ORIGINAL_ROOT / 'cache' / name
        target = destination / name
        if not source.is_file():
            raise FileNotFoundError(source)
        if not target.exists():
            temporary = target.with_suffix(target.suffix + '.tmp')
            shutil.copy2(source, temporary)
            temporary.replace(target)


def main():
    prepare_protocol()
    share_verified_cache()
    base.ROOT = RECORDING_ROOT
    base.PROTOCOL = RECORDING_ROOT / 'training_protocol.json'
    base.run_epoch = balanced_run_epoch
    trainer = base.ExpansionTraining()
    trainer.metadata['metric_recording'] = trainer.protocol['metric_recording']
    trainer.run()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        base.write(RECORDING_ROOT / 'failure.json',
                   {'status': 'failed', 'time': base.now(), 'traceback': traceback.format_exc()})
        raise
