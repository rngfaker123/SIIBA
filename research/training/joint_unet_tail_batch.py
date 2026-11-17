"""Compatibility ONLY for the clean cohort's final two-image/four-rank batch.

Complete batches delegate to the unchanged historical DDP batch planner. The
two-image tail uses loss/2 on the bare module on ranks 0/1 only, then one
packed-gradient SUM with no post-division. This avoids an unnecessary factor
of four in FP16 intermediate backward values. Ranks 2/3 perform no model
forward and see no synthetic/repeated image.
The caller still invokes the original synchronized AMP/optimizer step once.
"""
import math

from models import matched_training_batching as original


BATCHING_RULE = 'original_global16_world4_micro1_with_exact_tail2_manual_gradient_v2'
WORLD, BATCH, MICRO = 4, 16, 1


def _settings(global_batch_size, world_size, max_microbatch_size):
    if (type(global_batch_size), type(world_size), type(max_microbatch_size)) != (int, int, int):
        raise ValueError('Integer batching settings required')
    if (global_batch_size, world_size, max_microbatch_size) != (BATCH, WORLD, MICRO):
        raise ValueError('This compatibility path only supports global16/world4/micro1')


def _tail_plan(indices, batch_index):
    indices = original._indices(indices)
    if len(indices) != 2:
        raise ValueError('Only the exact two-image tail is supported')
    ranks = []
    for rank in range(WORLD):
        local = indices[rank::WORLD]
        micros = (original.MicroBatch(local, 1.0/2.0, False),) if local else ()
        ranks.append(original.RankBatch(rank, local, micros))
    return original.GlobalBatch(batch_index, indices, tuple(ranks))


def is_short_tail(plan):
    return len(plan.indices) == 2


def iter_epoch_batches(order, global_batch_size=16, world_size=4,
                       max_microbatch_size=1, start_batch=0):
    _settings(global_batch_size, world_size, max_microbatch_size)
    order = original._indices(order)
    original._integer(start_batch, 'start_batch')
    if not order:
        raise ValueError('Epoch order must not be empty')
    remainder = len(order) % BATCH
    # Except for the explicitly authorized tail2, preserve the old validation.
    if remainder != 2:
        return original.iter_epoch_batches(order, BATCH, WORLD, MICRO, start_batch)
    count = (len(order)+BATCH-1)//BATCH
    if start_batch > count:
        raise ValueError('start_batch exceeds epoch batches')
    def plans():
        for batch_index in range(start_batch, count):
            indices = order[batch_index*BATCH:(batch_index+1)*BATCH]
            if len(indices) == 2:
                yield _tail_plan(indices, batch_index)
            else:
                yield original.plan_global_batch(indices, WORLD, MICRO, batch_index)
    return plans()


def resume_metadata(order, next_batch, global_batch_size=16, world_size=4,
                    max_microbatch_size=1):
    order = original._indices(order)
    iter_epoch_batches(order, global_batch_size, world_size, max_microbatch_size, next_batch)
    return dict(batching_rule=BATCHING_RULE, order_sha256=original.order_sha256(order),
                num_samples=len(order), global_batch_size=BATCH, world_size=WORLD,
                max_microbatch_size=MICRO, next_batch=next_batch,
                short_tail_samples=len(order) % BATCH,
                tail2_rule='rank0/1_one_real_image;rank2/3_no_forward;normalized_before_backward_loss_div2;scaled_grad_SUM;no_postdivide')


def validate_resume_metadata(order, metadata, global_batch_size=16, world_size=4,
                             max_microbatch_size=1):
    if not isinstance(metadata, dict) or 'next_batch' not in metadata:
        raise ValueError('Missing batching resume metadata')
    expected = resume_metadata(order, metadata['next_batch'], global_batch_size,
                               world_size, max_microbatch_size)
    if expected != metadata:
        raise ValueError('Resume order/count/layout/tail protocol mismatch')
    return expected['next_batch']


def tail_backward(core, plan, rank, scaler, device, loss_closure):
    """Accumulate the tail's synchronized, STILL SCALED global-mean gradients.

    ``loss_closure()`` returns ``(one_image_loss_vector, metrics)`` and must use
    ``core`` (the bare module), NOT its DDP wrapper. It is called only on ranks
    0/1. The returned payload is unchanged on active ranks, None on idle ranks.
    All ranks call this once, then the historical ``step_optimizer`` once.

    There are matched all-rank phase collectives, never collectives nested in
    rank-specific microbatch counts: forward/scale validity, backward validity,
    and one packed gradient SUM. Each active loss is divided by the actual
    global tail size (2) BEFORE backward; bare-module gradients have no DDP
    averaging to compensate. Nonfinite gradients are left for the
    original AMP overflow policy; nonfinite loss/failed backward fail closed.
    """
    import torch
    import torch.distributed as dist
    if type(rank) is not int or rank not in range(WORLD):
        raise ValueError('Invalid rank')
    if not dist.is_initialized() or dist.get_world_size() != WORLD or dist.get_rank() != rank:
        raise RuntimeError('Tail requires the original four-rank process group')
    if plan != _tail_plan(plan.indices, plan.batch_index):
        raise ValueError('Noncanonical tail plan')
    parameters = [parameter for parameter in core.parameters() if parameter.requires_grad]
    if not parameters or len({parameter.dtype for parameter in parameters}) != 1:
        raise ValueError('Nonempty, common-dtype trainable parameters required')
    if any(parameter.device != torch.device(device) for parameter in parameters):
        raise ValueError('Tail parameter device mismatch')
    core.zero_grad(set_to_none=True)
    active = bool(plan.rank_batches[rank].indices)
    payload, failure, valid = None, None, True
    try:
        # Initialize GradScaler's scale/growth tensors even on never-active
        # ranks. This scalar performs NO model forward and NO backward.
        scaler.scale(torch.zeros((), device=device, dtype=torch.float32))
        scale = float(scaler.get_scale())
        if not math.isfinite(scale) or scale <= 0:
            raise FloatingPointError('Invalid AMP scale')
        if active:
            payload = loss_closure()
            if not isinstance(payload, tuple) or len(payload) != 2:
                raise ValueError('Tail closure must return (loss_vector, metrics)')
            loss = payload[0]
            if (not isinstance(loss, torch.Tensor) or loss.shape != (1,) or not loss.is_floating_point()
                    or loss.device != torch.device(device) or not bool(torch.isfinite(loss).all())):
                raise FloatingPointError('Invalid/nonfinite one-image tail loss')
    except Exception as error:
        failure, valid, scale = repr(error), False, 1.0
    # MIN collects validity and minimum scale; MIN(-scale) is negative maximum.
    status = torch.tensor([float(valid), scale, -scale], dtype=torch.float64, device=device)
    dist.all_reduce(status, op=dist.ReduceOp.MIN)
    if not bool(status[0].item()) or status[1].item() != -status[2].item():
        core.zero_grad(set_to_none=True)
        raise FloatingPointError('Tail forward/AMP-scale failed on a rank: ' + (failure or 'remote rank/scale mismatch'))
    backward_ok, failure = True, None
    try:
        if active:
            scaler.scale(payload[0].sum() / 2.0).backward()
            if any(parameter.grad is None for parameter in parameters):
                raise RuntimeError('Active tail parameter missing gradient')
        else:
            for parameter in parameters:
                parameter.grad = torch.zeros_like(parameter)
        if any(parameter.grad.dtype != parameter.dtype or parameter.grad.device != parameter.device
               or parameter.grad.shape != parameter.shape or parameter.grad.is_sparse for parameter in parameters):
            raise RuntimeError('Invalid tail gradient representation')
    except Exception as error:
        failure, backward_ok = repr(error), False
    status = torch.tensor(int(backward_ok), dtype=torch.int32, device=device)
    dist.all_reduce(status, op=dist.ReduceOp.MIN)
    if not bool(status.item()):
        core.zero_grad(set_to_none=True)
        raise RuntimeError('Tail backward failed on a rank: ' + (failure or 'remote rank'))
    with torch.no_grad():
        gradients = torch.cat([parameter.grad.reshape(-1) for parameter in parameters])
        dist.all_reduce(gradients, op=dist.ReduceOp.SUM)
        cursor = 0
        for parameter in parameters:
            count = parameter.numel()
            parameter.grad.copy_(gradients[cursor:cursor+count].view_as(parameter))
            cursor += count
    return payload
