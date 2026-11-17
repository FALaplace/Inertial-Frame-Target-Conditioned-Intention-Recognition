"""GPU-resident batches and epoch metrics for the final scratch matrix."""
from __future__ import annotations

import math
import copy

import torch
from torch.utils.data import DataLoader

from expansion_training_balanced import BalancedEpochMetrics
from model import soft_label_cross_entropy


class GpuBatches:
    """Keep normalized inputs on GPU, shuffle only CPU sample indices.

    The standard DataLoader retains the reference sampler's generator use,
    including its base-seed draw and non-dropped final batch.
    """
    def __init__(self, dataset, device, batch_size=32, shuffle=False, generator=None):
        self.device = device
        self.columns = {
            key: torch.stack([row[key] for row in dataset.rows]).to(device)
            for key in dataset.rows[0]
        }
        self.size = len(dataset)
        self.batch_size = batch_size
        self.indices = DataLoader(range(self.size), batch_size=batch_size, shuffle=shuffle,
                                  generator=generator, num_workers=0, pin_memory=device.type == 'cuda')

    def __len__(self):
        return math.ceil(self.size / self.batch_size)

    def with_sampling(self, batch_size=32, shuffle=False, generator=None):
        """Share the immutable GPU columns with a separate standard sampler."""
        result = copy.copy(self)
        result.batch_size = batch_size
        result.indices = DataLoader(range(self.size), batch_size=batch_size, shuffle=shuffle,
                                    generator=generator, num_workers=0, pin_memory=self.device.type == 'cuda')
        return result

    def __iter__(self):
        for indices in self.indices:
            indices = indices.to(self.device, non_blocking=True)
            yield {key: values.index_select(0, indices) for key, values in self.columns.items()}

    @property
    def storage_bytes(self):
        return sum(value.numel() * value.element_size() for value in self.columns.values())


def gpu_run_epoch(model, batches, device, optimizer=None, scaler=None):
    """Use the same loss/update, returning counts and metrics once per epoch."""
    training = optimizer is not None
    model.train(training)
    totals = torch.zeros(2, dtype=torch.float64, device=device)
    support = torch.zeros(8, dtype=torch.int64, device=device)
    correct = torch.zeros_like(support)
    classes = torch.arange(8, device=device)
    sample_count = 0
    keys = tuple(model.model_input_keys)
    with torch.set_grad_enabled(training):
        for batch in batches:
            inputs = {key: batch[key].to(device, non_blocking=True) for key in keys}
            targets = batch['y'].to(device, non_blocking=True)
            count = targets.shape[0]
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                outputs = model(inputs)
                loss = soft_label_cross_entropy(outputs['masked_logits'], targets)
            probabilities = outputs['probabilities'].detach()
            # Keep a synchronous numerical guard before any optimizer update.
            if not (torch.isfinite(loss) & torch.isfinite(probabilities).all()):
                raise FloatingPointError('Non-finite training/validation loss or probabilities.')
            if training:
                if scaler is None:
                    raise RuntimeError('A GradScaler is required for training.')
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            totals[0] += loss.detach().double() * count
            totals[1] += (probabilities - targets).square().sum(dim=-1).sum().double()
            reference = targets.argmax(dim=-1)
            predicted = probabilities.argmax(dim=-1)
            reference_classes = reference[:, None] == classes[None, :]
            support += reference_classes.sum(dim=0)
            correct += (reference_classes & (reference == predicted)[:, None]).sum(dim=0)
            sample_count += count
    if not sample_count:
        raise ValueError('No samples in epoch.')
    sums = totals.cpu().tolist()
    counts = torch.stack((support, correct)).cpu().tolist()
    observed_recall = [right / n for n, right in zip(*counts) if n]
    return BalancedEpochMetrics(loss=sums[0] / sample_count,
                                accuracy=sum(counts[1]) / sample_count,
                                brier_score=sums[1] / sample_count,
                                balanced_accuracy=sum(observed_recall) / len(observed_recall),
                                class_support=counts[0], class_correct=counts[1])
