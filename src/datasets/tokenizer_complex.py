# -*- coding: utf-8 -*-
"""One alphabet for protein, RNA and DNA.

Twenty amino acids, then A, C, G, T and U, then the mask token the decoder
uses for a position it has not decided, then the two unknown-identity tokens,
X for a protein position and N for a nucleotide. Twenty-eight in all. Which
of them a position may actually take is decided by the model, from the
position's polymer type.
"""
from typing import List, Optional, Sequence


class Unitokenizer_Complex:
    """Unified tokenizer with a dedicated mask token for protein / DNA / RNA."""
    def __init__(self, map_N_mode: str = "keep", mask_char: str = "*", unknown_to_mask: bool = False):
        # protein
        self.alphabet_protein = 'ACDEFGHIKLMNPQRSTVWY'         # 0..19
        self.protein_to_id = {ch: i for i, ch in enumerate(self.alphabet_protein)}
        self.protein_X_id = 26                                  # X -> 26

        # nucleotides
        self.nuc5 = ['A', 'C', 'G', 'T', 'U']                  # 20..24
        self.nuc_to_id = {b: 20 + i for i, b in enumerate(self.nuc5)}
        self.nuc_N_id = 27                                      # N -> 27
        self.nuc_to_id['N'] = self.nuc_N_id

        # mask
        self.mask_char = mask_char
        self.mask_id = 25

        # vocab size (20 aa + 5 nuc + MASK + X + N) = 28
        self.vocab_size = 28

        self.map_N_mode = map_N_mode    # 'keep' | 'A' | 'random'
        self.unknown_to_mask = unknown_to_mask

        # convenient id sets
        self.protein_ids = list(range(20)) + [self.protein_X_id]
        self.dna_ids = [self.nuc_to_id[x] for x in ['A','C','G','T']] + [self.nuc_N_id]
        self.rna_ids = [self.nuc_to_id[x] for x in ['A','C','G','U']] + [self.nuc_N_id]

    def _map_Ns(self, s: str) -> str:
        if 'N' not in s: return s
        if self.map_N_mode == 'keep':
            return s
        if self.map_N_mode == 'A':
            return s.replace('N', 'A')
        elif self.map_N_mode == 'random':
            import random
            return ''.join(ch if ch!='N' else random.choice('ACGT') for ch in s)
        else:
            return s

    def _auto_kind(self, s: str) -> str:
        S = set(s)
        S.discard(self.mask_char)
        # allow N in both dna/rna auto-detect
        if S.issubset(set('AUGCN')):   return 'rna'
        if S.issubset(set('ATGCN')):   return 'dna'
        return 'protein'

    def encode(self, seq: str, kind: Optional[str] = None) -> List[int]:
        s = seq.upper()
        if kind is None:
            kind = self._auto_kind(s)

        if kind == 'protein':
            out = []
            for ch in s:
                if ch == self.mask_char:
                    out.append(self.mask_id); continue
                if ch in self.protein_to_id:
                    out.append(self.protein_to_id[ch]); continue
                if ch == 'X':
                    out.append(self.protein_X_id); continue
                # unknown AA
                out.append(self.mask_id if self.unknown_to_mask else self.protein_X_id)
            return out

        if kind == 'dna':
            s = s.replace('U','T')
            s = self._map_Ns(s)
        elif kind == 'rna':
            s = s.replace('T','U')
            s = self._map_Ns(s)
        else:
            raise ValueError(f"Unknown kind: {kind}")

        out = []
        for ch in s:
            if ch == self.mask_char:
                out.append(self.mask_id); continue
            if ch in self.nuc_to_id:
                out.append(self.nuc_to_id[ch])
            else:
                # unknown base -> mask or 'A'
                out.append(self.mask_id if self.unknown_to_mask else self.nuc_to_id['A'])
        return out

    def decode(self, ids: Sequence[int], kind: Optional[str] = None) -> str:
        out = []
        for t in ids:
            if t == self.mask_id:
                out.append(self.mask_char)
            elif 0 <= t < 20:
                out.append(self.alphabet_protein[t])
            elif t == self.protein_X_id:
                out.append('X')
            elif 20 <= t <= 24 or t == self.nuc_N_id:
                if t == self.nuc_N_id:
                    out.append('N'); continue
                base = self.nuc5[t-20]
                if kind == 'dna' and base == 'U': base = 'T'
                if kind == 'rna' and base == 'T': base = 'U'
                out.append(base)
            else:
                out.append(self.mask_char)
        return ''.join(out)
