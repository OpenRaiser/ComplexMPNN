# -*- coding: utf-8 -*-
"""Segment reductions on plain torch."""
from __future__ import annotations

import torch


def _expand(index: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    if index.dim() != 1:
        raise ValueError('index must be one-dimensional')
    if index.shape[0] != like.shape[0]:
        raise ValueError('index and source disagree on the first dimension')
    return index.view(-1, *([1] * (like.dim() - 1))).expand_as(like)


def segment_sum(src: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    out = src.new_zeros((n, *src.shape[1:]))
    return out.index_add_(0, index, src)


def segment_max(src: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    out = src.new_full((n, *src.shape[1:]), float('-inf'))
    return out.scatter_reduce_(0, _expand(index, src), src, reduce='amax',
                               include_self=True)


def segment_softmax(src: torch.Tensor, index: torch.Tensor, n: int,
                    eps: float = 1e-16) -> torch.Tensor:
    """Softmax within each segment, over dimension zero."""
    m = segment_max(src, index, n)
    e = torch.exp(src - m[index])
    s = segment_sum(e, index, n)
    return e / s[index].clamp_min(eps)
