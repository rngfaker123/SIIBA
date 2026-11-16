"""Exact, unpadded optimizer batches for unequal-size DDP rank batches.

This module performs no training, process initialization, or dataset access.
For global batch 16, three ranks, and a maximum physical microbatch of two,
rank sample counts are 6/5/5 and micro-round sizes are (2,2,2), (2,2,2),
(2,1,1). The final four-sample batch is (2,1,1), not a padded batch of 16.

Each microbatch loss MUST be a mean over its images (with identical per-image
reduction). Multiply it by ``loss_scale`` before backward. Default DDP averages
gradients across ranks, so ``world_size * micro_n / global_n`` produces exactly
the global image-mean gradient, including the short last batch. Do not divide
again by the number of accumulation rounds. Use DDP ``no_sync()`` around BOTH
forward and backward for every microbatch except ``synchronize=True``. Perform
clipping, optimizer/scaler step, and EMA update once after the complete plan.

All ranks perform the same number of nonempty forwards/backwards. Layouts that
cannot satisfy this without padding, dropping, or empty batches fail explicitly.
This assumes sample-independent layers (e.g. GroupNorm), no custom DDP reduction
hook, and synchronized optimizer/AMP overflow decisions across ranks. Unequal
microbatches are not equivalent for training-mode BatchNorm or batchwise losses.

The default ``epoch_order`` is a NEW explicitly named reproducible Python RNG
rule, NOT a claim to reproduce an existing DataLoader shuffle. For historical
matching, pass the exact frozen permutation produced by that audited loader to
``iter_epoch_batches``. In particular, DataLoader generator base-seed draws can
make a plain seed+epoch randperm differ from the historical loader's order.
"""

from dataclasses import dataclass
import hashlib
import random
import struct


ORDER_RULE = "python_random_seed_plus_epoch_shuffle_v1"
BATCHING_RULE = "strided_rank_assignment_balanced_nonempty_microbatches_v1"


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError("%s must be an integer >= %d" % (name, minimum))
    return value


def _indices(values):
    result = tuple(values)
    for value in result:
        _integer(value, "sample index")
    if len(set(result)) != len(result):
        raise ValueError("Duplicate sample indices are not allowed")
    return result


def epoch_order(num_samples, seed, epoch):
    """Return a deterministic permutation; not the historical loader's rule.

    ``epoch`` is zero-based. This function does not touch the global RNG state.
    Save the resulting order/hash, software version, and ORDER_RULE in a run's
    frozen protocol. Use an externally frozen order when matching another rule.
    """
    _integer(num_samples, "num_samples", 1)
    _integer(seed, "seed")
    _integer(epoch, "epoch")
    order = list(range(num_samples))
    random.Random(seed + epoch).shuffle(order)
    return tuple(order)


def order_sha256(order):
    """Hash exact ordered integer indices, independent of JSON formatting."""
    order = _indices(order)
    digest = hashlib.sha256(b"matched_training_order_v1\x00")
    for index in order:
        if index >= 2 ** 64:
            raise ValueError("Sample index exceeds uint64 hash encoding")
        digest.update(struct.pack("<Q", index))
    return digest.hexdigest()


@dataclass(frozen=True)
class MicroBatch:
    indices: tuple
    loss_scale: float
    synchronize: bool


@dataclass(frozen=True)
class RankBatch:
    rank: int
    indices: tuple
    microbatches: tuple


@dataclass(frozen=True)
class GlobalBatch:
    batch_index: int
    indices: tuple
    rank_batches: tuple

    @property
    def num_samples(self):
        return len(self.indices)

    @property
    def micro_rounds(self):
        return len(self.rank_batches[0].microbatches)


def _layout(global_n, world_size, max_microbatch_size):
    _integer(global_n, "global batch sample count", 1)
    _integer(world_size, "world_size", 1)
    _integer(max_microbatch_size, "max_microbatch_size", 1)
    if global_n < world_size:
        raise ValueError(
            "Global batch has fewer samples than ranks; empty ranks, padding, "
            "and dropping are unsupported")
    counts = tuple(len(range(rank, global_n, world_size))
                   for rank in range(world_size))
    rounds = (max(counts) + max_microbatch_size - 1) // max_microbatch_size
    if min(counts) < rounds:
        raise ValueError(
            "Cannot give every rank the same number of nonempty microbatches "
            "within max_microbatch_size; increase the maximum or use a "
            "separately approved layout")
    return counts, rounds


def plan_global_batch(indices, world_size=3, max_microbatch_size=2,
                      batch_index=0):
    """Partition one explicit global batch, preserving every index once.

    Rank ``r`` receives ``indices[r::world_size]``. Rank-local indices are split
    into equally many nearly balanced, nonempty microbatches; extra images are
    assigned to earlier micro-rounds. This function never pads or drops data.
    """
    indices = _indices(indices)
    _integer(batch_index, "batch_index")
    counts, rounds = _layout(len(indices), world_size, max_microbatch_size)
    ranks = []
    for rank, local_n in enumerate(counts):
        local_indices = indices[rank::world_size]
        quotient, remainder = divmod(local_n, rounds)
        cursor = 0
        micros = []
        for micro_index in range(rounds):
            micro_n = quotient + int(micro_index < remainder)
            micros.append(MicroBatch(
                indices=local_indices[cursor:cursor + micro_n],
                loss_scale=world_size * micro_n / len(indices),
                synchronize=(micro_index == rounds - 1),
            ))
            cursor += micro_n
        ranks.append(RankBatch(rank, local_indices, tuple(micros)))
    return GlobalBatch(batch_index, indices, tuple(ranks))


def iter_epoch_batches(order, global_batch_size=16, world_size=3,
                       max_microbatch_size=2, start_batch=0):
    """Return an iterator of plans from an optimizer-batch resume offset.

    All order/layout/offset checks run immediately, before the first yielded
    plan; a bad short final batch therefore fails before any training starts.
    ``start_batch`` is the NEXT batch to attempt, not the count of successful
    optimizer updates (AMP-skipped attempts still consumed data). It can equal
    the number of batches for an already consumed epoch. Save training RNG,
    scaler, optimizer, and statistics separately; this only resumes data order.
    """
    order = _indices(order)
    if not order:
        raise ValueError("Epoch order must not be empty")
    _integer(global_batch_size, "global_batch_size", 1)
    _integer(start_batch, "start_batch")
    _layout(min(global_batch_size, len(order)), world_size, max_microbatch_size)
    tail = len(order) % global_batch_size
    if tail:
        _layout(tail, world_size, max_microbatch_size)
    count = (len(order) + global_batch_size - 1) // global_batch_size
    if start_batch > count:
        raise ValueError("start_batch exceeds the number of epoch batches")

    def plans():
        for batch_index in range(start_batch, count):
            start = batch_index * global_batch_size
            yield plan_global_batch(
                order[start:start + global_batch_size], world_size,
                max_microbatch_size, batch_index)

    return plans()


def resume_metadata(order, next_batch, global_batch_size=16, world_size=3,
                    max_microbatch_size=2):
    """Return JSON-safe data-order identity for a batch-boundary checkpoint."""
    order = _indices(order)
    iter_epoch_batches(order, global_batch_size, world_size,
                       max_microbatch_size, next_batch)
    return {
        "batching_rule": BATCHING_RULE,
        "order_sha256": order_sha256(order),
        "num_samples": len(order),
        "global_batch_size": global_batch_size,
        "world_size": world_size,
        "max_microbatch_size": max_microbatch_size,
        "next_batch": next_batch,
    }


def validate_resume_metadata(order, metadata, global_batch_size=16,
                             world_size=3, max_microbatch_size=2):
    """Reject any changed order/layout instead of silently resuming differently.

    Returns the verified next-batch offset. Do not use this metadata as a full
    training checkpoint: weights, RNG and optimizer state are outside its scope.
    """
    if not isinstance(metadata, dict) or "next_batch" not in metadata:
        raise ValueError("Missing batching resume metadata")
    expected = resume_metadata(order, metadata["next_batch"], global_batch_size,
                               world_size, max_microbatch_size)
    if metadata != expected:
        raise ValueError("Resume order, sample count, or batching protocol mismatch")
    return expected["next_batch"]
