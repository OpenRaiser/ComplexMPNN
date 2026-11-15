# -*- coding: utf-8 -*-
"""Generation: equally spaced flow steps over the design positions.

Generation starts with every design position masked and ends with a complete
sequence. At each step the decoder predicts a distribution at every position;
masked positions are then filled by sampling from it, and positions filled
earlier can be masked again so that later steps can revise them. The last
step fills what is left and keeps what is already there.
"""
from __future__ import annotations

import math

import torch


def transition_probs(clean_probs, state, mask_id, t, next_t, remask=1.0,
                     final=False):
    """One step of the kernel, as a distribution over tokens per position.

    A masked position is filled with probability u = dt (1 + eta t) / (1 - t),
    drawing from `clean_probs`. A filled design position is masked again with
    probability r = dt eta. `final` sets u = 1 and r = 0.
    """
    if not 0 <= t < next_t <= 1:
        raise ValueError('a flow step needs 0 <= t < next_t <= 1')
    if not math.isfinite(remask) or remask < 0:
        raise ValueError('the remask rate must be finite and nonnegative')
    dt = next_t - t
    if final:
        unmask, remask_prob = 1., 0.
    else:
        unmask = min(1., dt * (1. + remask * t) / (1. - t))
        remask_prob = min(1., dt * remask)
    masked = state == mask_id
    stay = torch.nn.functional.one_hot(state, clean_probs.shape[-1]
                                       ).to(clean_probs.dtype)
    probs = torch.where(masked[:, None], unmask * clean_probs,
                        (1. - remask_prob) * stay)
    ones = torch.ones_like(probs[:, 0])
    probs[:, mask_id] += torch.where(masked, (1. - unmask) * ones,
                                     remask_prob * ones)
    return probs


def denoise(model, batch, h_V, h_E, state, time, feedback=None):
    """One decoder call, with the tokens a position cannot take ruled out."""
    logits = model.decoder(h_V, h_E, batch['edge_idx'], state,
                           time[batch['batch_id']], batch['design_mask'],
                           model.tokenizer.mask_id, feedback=feedback)
    return model.mask_illegal(logits, batch['type_vec'])


def _replicate(batch, h_V, h_E, num_samples):
    """Lay out `num_samples` independent copies of a batch, side by side.

    Returns the copied batch, the copied encodings, and how many complexes
    the result holds in total, which is what indexes the flow time.
    """
    n = len(h_V)
    out = dict(batch)
    for key in ('S', 'design_mask', 'type_vec', 'chain_idx', 'res_idx'):
        out[key] = batch[key].repeat(num_samples)
    n_graphs = int(batch['batch_id'].max()) + 1
    out['batch_id'] = torch.cat([batch['batch_id'] + k * n_graphs
                                 for k in range(num_samples)])
    out['edge_idx'] = torch.cat([batch['edge_idx'] + k * n
                                 for k in range(num_samples)], dim=1)
    return (out, h_V.repeat(num_samples, 1), h_E.repeat(num_samples, 1),
            n_graphs * num_samples)


@torch.no_grad()
def sample_flow(model, batch, *, temperature=0.1, num_samples=1, steps=None,
                remask=None, logit_bias=None):
    """Design the masked positions. One sequence per sample per position."""
    steps = model.flow_steps if steps is None else steps
    remask = model.flow_remask if remask is None else remask
    if not isinstance(steps, int) or steps < 1:
        raise ValueError('the number of flow steps must be a positive integer')
    if not math.isfinite(remask) or remask < 0:
        raise ValueError('the remask rate must be finite and nonnegative')
    if not isinstance(num_samples, int) or num_samples < 1:
        raise ValueError('num_samples must be a positive integer')
    if temperature is None or not math.isfinite(temperature) or temperature < 0:
        raise ValueError('the temperature must be finite and nonnegative')

    was_training = model.training
    model.eval()
    try:
        h_V, h_E = model.encode(batch)
        mask_id = model.tokenizer.mask_id
        b, h_V, h_E, n_complexes = _replicate(batch, h_V, h_E, num_samples)
        design = b['design_mask'].bool()
        if not bool(design.any()):
            raise ValueError('nothing is marked for design')
        state = b['S'].masked_fill(design, mask_id)
        bias = None if logit_bias is None else logit_bias.repeat(num_samples, 1)
        legal = model.legal[b['type_vec']]
        feedback = None

        for step in range(steps):
            t, next_t = step / steps, (step + 1) / steps
            time = torch.full((n_complexes,), t, device=state.device)
            logits = denoise(model, b, h_V, h_E, state, time, feedback).float()
            # The feedback the next step reads is the prediction itself,
            # before any bias or temperature.
            feedback = logits.softmax(-1)
            if bias is not None:
                logits = logits + bias
            # MASK is a state of the process and never an endpoint, and the
            # two nucleic alphabets differ, so legality is applied here rather
            # than trusted through the bias.
            logits = logits.masked_fill(~legal, float('-inf'))
            if not bool(torch.isfinite(logits).any(-1)[design].all()):
                raise ValueError('the bias forbids every token allowed at a '
                                 'design position')
            if temperature == 0:
                clean = torch.nn.functional.one_hot(
                    logits.argmax(-1), logits.shape[-1]).float()
            else:
                clean = (logits / temperature).softmax(-1)
            probs = transition_probs(clean, state, mask_id, t, next_t, remask,
                                     final=step == steps - 1)
            # Positions that are not being designed keep their identity.
            fixed = torch.nn.functional.one_hot(state, probs.shape[-1]).float()
            probs = torch.where(design[:, None], probs, fixed)
            state = torch.multinomial(probs, 1).squeeze(-1)

        if bool((design & ~legal.gather(1, state[:, None]).squeeze(1)).any()):
            raise RuntimeError('generation ended on a token the position '
                               'cannot take')
        return dict(S=state, design_mask=design, num_samples=num_samples,
                    flow_steps=steps, flow_remask=remask)
    finally:
        model.train(was_training)
