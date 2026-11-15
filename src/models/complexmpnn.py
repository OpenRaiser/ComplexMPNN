# -*- coding: utf-8 -*-
"""The ComplexMPNN network.

The encoder reads the backbone and the cofactors once. The decoder reads that
encoding together with the sequence as it stands, and runs again at every
generation step. Which tokens a position may take follows from its polymer
type and is held in `legal`.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from src.datasets.tokenizer_complex import Unitokenizer_Complex
from src.modules.encoder import Encoder

PROTEIN_TYPE, RNA_TYPE, DNA_TYPE = 0, 1, 2
TYPE_ID_BY_NAME = {'protein': PROTEIN_TYPE, 'rna': RNA_TYPE, 'dna': DNA_TYPE}
N_POLYMER_TYPES = 3

# The smoothing the released weights were trained with.
LABEL_SMOOTHING = 0.05


def legal_token_table(tok: Unitokenizer_Complex) -> torch.Tensor:
    """[n_polymer_types, vocab] bool: which tokens a type may be assigned."""
    table = torch.zeros(N_POLYMER_TYPES, tok.vocab_size, dtype=torch.bool)
    for i in range(20):
        table[PROTEIN_TYPE, i] = True
    for base in ('A', 'C', 'G', 'U'):
        table[RNA_TYPE, tok.nuc_to_id[base]] = True
    for base in ('A', 'C', 'G', 'T'):
        table[DNA_TYPE, tok.nuc_to_id[base]] = True
    return table


class ComplexMPNN(nn.Module):
    def __init__(self, hidden_dim: int = 128, encoder_layers: int = 6,
                 ar_layers: int = 3, n_heads: int = 4, dropout: float = 0.1,
                 n_virtual: int = 3, n_rbf: int = 16, pos_clip: int = 32,
                 fixed_atom_context: bool = False,
                 flow_steps: int = 50, flow_remask: float = 1.0):
        super().__init__()
        from src.modules.flow_decoder import FlowDecoder
        if flow_steps < 1 or not 0 <= flow_remask < float('inf'):
            raise ValueError('flow_steps must be positive; flow_remask finite '
                             'and nonnegative')
        self.tokenizer = Unitokenizer_Complex()
        self.flow_steps, self.flow_remask = int(flow_steps), float(flow_remask)
        self.fixed_atom_context = bool(fixed_atom_context)
        self.encoder = Encoder(hidden_dim=hidden_dim, n_layers=encoder_layers,
                               n_heads=n_heads, dropout=dropout,
                               n_virtual=n_virtual, n_rbf=n_rbf,
                               pos_clip=pos_clip,
                               n_polymer_types=N_POLYMER_TYPES,
                               fixed_atom_context=fixed_atom_context)
        self.decoder = FlowDecoder(hidden_dim, self.tokenizer.vocab_size,
                                   n_layers=ar_layers, dropout=dropout)
        self.register_buffer('legal', legal_token_table(self.tokenizer),
                             persistent=False)

    def encode(self, batch: Dict[str, torch.Tensor]):
        """The structural encoding: one vector per residue and per edge."""
        return self.encoder(batch)

    def mask_illegal(self, logits: torch.Tensor, type_vec: torch.Tensor
                     ) -> torch.Tensor:
        """Push the tokens a position's type cannot take below the softmax."""
        legal = self.legal[type_vec]
        return logits.masked_fill(~legal, torch.finfo(logits.dtype).min / 2)

    def log_probs(self, batch: Dict[str, torch.Tensor], state: torch.Tensor,
                  time: torch.Tensor,
                  feedback: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Log-probabilities at every position, for one sequence state.

        `state` holds the token at fixed positions and either a token or MASK
        at design positions. `time` is one flow time per complex in the batch.
        """
        from src.modules.flow_sampling import denoise
        h_V, h_E = self.encode(batch)
        logits = denoise(self, batch, h_V, h_E, state, time, feedback)
        return torch.log_softmax(logits.float(), dim=-1)

    def marginal_log_probs(self, batch: Dict[str, torch.Tensor]
                           ) -> torch.Tensor:
        """Log-probabilities with every design position masked at once.

        One decoder call at flow time zero, so each design position is scored
        from the structure, the cofactors and the fixed sequence alone. It
        does not condition on the identities chosen at the other design
        positions, which is what makes it a marginal and not the likelihood
        of a complete sequence.
        """
        design = batch['design_mask'].bool()
        state = batch['S'].masked_fill(design, self.tokenizer.mask_id)
        n_graphs = int(batch['batch_id'].max()) + 1
        time = torch.zeros(n_graphs, device=state.device)
        return self.log_probs(batch, state, time)

    def loss_sums(self, log_probs: torch.Tensor,
                  batch: Dict[str, torch.Tensor],
                  label_smoothing: float = LABEL_SMOOTHING
                  ) -> Dict[str, torch.Tensor]:
        """The training objective, as sums and counts rather than as a mean.

        Cross-entropy against the native token, with the smoothing mass spread
        over the legal tokens of the position's own polymer type, so a protein
        position's smoothing never lands on a nucleotide. Only the positions in
        `loss_mask` are counted.

        Sums rather than a mean because the training runs divided by the
        supervised count over all ranks and all accumulated microbatches, not
        per batch. A single reading is `loss / count`.
        """
        if not 0 <= label_smoothing < 1:
            raise ValueError('label_smoothing must be in [0, 1)')
        lp = log_probs.float()
        mask = batch['loss_mask'].to(lp.dtype)
        native = batch['S'][:, None]
        nll = -lp.gather(-1, native).squeeze(-1)
        per_position = nll
        if label_smoothing > 0:
            legal = self.legal[batch['type_vec']].to(lp.dtype)
            spread = -(lp * legal).sum(-1) / legal.sum(-1).clamp_min(1)
            per_position = (1 - label_smoothing) * nll + label_smoothing * spread
        hits = (lp.argmax(-1) == batch['S']).to(lp.dtype)
        return dict(loss=(per_position * mask).sum(),
                    nll=(nll * mask).sum(),
                    hits=(hits * mask).sum(),
                    count=mask.sum())

    def sample(self, batch: Dict[str, torch.Tensor], temperature: float = 0.1,
               num_samples: int = 1,
               logit_bias: Optional[torch.Tensor] = None,
               flow_steps: Optional[int] = None,
               flow_remask: Optional[float] = None) -> Dict[str, object]:
        """Design the masked positions, drawing `num_samples` sequences."""
        from src.modules.flow_sampling import sample_flow
        return sample_flow(self, batch, temperature=temperature,
                           num_samples=num_samples, steps=flow_steps,
                           remask=flow_remask, logit_bias=logit_bias)

    def parameter_report(self) -> Dict[str, int]:
        """Where the parameters are, for the line each run prints."""
        encoder = self.encoder.parameter_report()
        decoder = sum(p.numel() for p in self.decoder.parameters())
        return dict(encoder_geometry=encoder['geometry'],
                    encoder_blocks=encoder['blocks'],
                    cofactor_context=encoder['cofactor_context'],
                    encoder_block_count=encoder['n_blocks'],
                    decoder=decoder,
                    total=sum(p.numel() for p in self.parameters()))
