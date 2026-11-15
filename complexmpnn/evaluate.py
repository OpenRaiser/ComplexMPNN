#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Design the entries of a JSONL test set and report recovery, per molecule type."""
from __future__ import annotations

import argparse
import collections
import csv
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from complexmpnn.design import load_model
from src.datasets.jsonl_dataset import ComplexDataset
from src.datasets.structure_input import TYPE_ID, collate_complex

# The length groups each per-type table is reported in, as the manuscript
# reports them. A chain falls in the group whose [low, high) holds its length,
# and len_all holds every chain. Chains are between 30 and 1,000 residues.
TYPE_BUCKET_SPECS = {
    'protein': [(0, 100, 'len_0_100'), (100, 300, 'len_100_300'),
                (300, 1000, 'len_300_1000'), (-1, 10**9, 'len_all')],
    'rna': [(0, 50, 'len_0_50'), (50, 100, 'len_50_100'),
            (100, 1000, 'len_100_1000'), (-1, 10**9, 'len_all')],
    'dna': [(0, 50, 'len_0_50'), (50, 100, 'len_50_100'),
            (100, 1000, 'len_100_1000'), (-1, 10**9, 'len_all')],
}

TYPE_NAMES = [n for n, _ in sorted(TYPE_ID.items(), key=lambda kv: kv[1])]
TASKS = ('all', 'one_chain', 'nucleic_given_protein', 'protein_given_nucleic')


def move(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def restrict(batch, design_types):
    """Keep only the requested polymer types designable."""
    if not design_types:
        return batch
    keep = torch.zeros_like(batch['design_mask'], dtype=torch.bool)
    for name in design_types:
        keep |= batch['type_vec'] == TYPE_ID[name]
    out = dict(batch)
    out['design_mask'] = batch['design_mask'].bool() & keep
    return out


def per_type(tokens, native, mask, type_vec, chain_idx=None):
    """{type: (hits, positions, {chain: (hits, positions)})} for one sample."""
    out = {}
    right = tokens == native
    for name in TYPE_NAMES:
        here = mask & (type_vec == TYPE_ID[name])
        n = int(here.sum())
        if not n:
            continue
        chains = {}
        if chain_idx is not None:
            for c in chain_idx[here].unique().tolist():
                sel = here & (chain_idx == c)
                chains[int(c)] = (int((right & sel).sum()), int(sel.sum()))
        out[name] = (int((right & here).sum()), n, chains)
    return out


def buckets_for(kind):
    """The evaluation's length buckets, imported rather than restated."""
    return TYPE_BUCKET_SPECS.get(kind, TYPE_BUCKET_SPECS['protein'])


def add_class_counts(cls, kind, pairs, specs):
    """Accumulate (tp, fp, fn) per class into every bucket a chain belongs to."""
    for pred, want, L in pairs:
        for lo, hi, bname in specs:
            if not (lo < 0 or (L >= 0 and lo <= L < hi)):
                continue
            bucket = cls[(kind, bname)]
            for a, b in zip(pred, want):
                bucket[b][0 if a == b else 2] += 1
                if a != b:
                    bucket[a][1] += 1


def macro_f1(counts):
    """Macro F1 over the classes PRESENT in the target of this bucket."""
    scores = []
    for cls, (tp, fp, fn) in sorted(counts.items()):
        if tp + fn == 0:                 # no support in the target
            continue
        denom = 2 * tp + fp + fn
        scores.append(2 * tp / denom if denom else 0.)
    return sum(scores) / len(scores) if scores else float('nan')


def protocols(pooled, entry_means, chain_means, by_sample):
    """The four readings of one accumulation, named."""
    import statistics as st
    hits, count = pooled
    out = dict(pooled=hits / count if count else float('nan'),
               per_entry=(sum(entry_means) / len(entry_means)
                          if entry_means else float('nan')),
               per_chain=(sum(chain_means) / len(chain_means)
                          if chain_means else float('nan')),
               entries=len(entry_means), chains=len(chain_means),
               positions=count)
    out['sample_sd'] = (st.stdev(by_sample) if len(by_sample) > 1
                        else float('nan'))
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--jsonl', required=True)
    p.add_argument('--out', default='', help='per-chain tsv; default none')
    p.add_argument('--device',
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--temperature', type=float, default=0.1)
    p.add_argument('--num-samples', type=int, default=1)
    p.add_argument('--flow-steps', type=int, default=50,
                   help='generation steps; 0 takes the number the checkpoint '
                        'was written with')
    p.add_argument('--task', choices=TASKS, default='all',
                   help='fixed for every entry, unlike the trainer, which '
                        'draws it per entry from the training weights')
    p.add_argument('--design-types', default='',
                   help='comma separated subset of %s; empty designs whatever '
                        'the task made designable' % ','.join(TYPE_NAMES))
    p.add_argument('--crop-budget', type=int, default=3000)
    p.add_argument('--contact-cutoff', type=float, default=12.0)
    p.add_argument('--limit', type=int, default=0, help='first N entries only')
    p.add_argument('--seed', type=int, default=111)
    p.add_argument('--nll', action='store_true',
                   help="also score the native sequence: one decoder call per "
                        "entry with every design position masked, reported as "
                        "mean negative log-likelihood per position")
    return p


def main() -> int:
    args = build_parser().parse_args()
    types = [t.strip() for t in args.design_types.split(',') if t.strip()]
    for name in types:
        if name not in TYPE_ID:
            raise SystemExit('--design-types takes %s, got %r'
                             % (','.join(TYPE_NAMES), name))

    blob = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    saved = dict(blob.get('args') or {})
    del blob
    knn = int(saved.get('knn') or 32)
    print('[infer] %s' % args.ckpt)
    print("[infer] knn %d (the checkpoint's), task %s, design types %s"
          % (knn, args.task,
             ','.join(types) or 'whatever the task designs'))
    print('[infer] T=%.3g, %d sample(s)%s'
          % (args.temperature, args.num_samples,
             '' if not args.flow_steps else ', %d flow steps' % args.flow_steps))

    model, device, _ = load_model(args.ckpt, args.device)
    model.eval()
    # One fixed task for every entry. The trainer's validation draws it from
    # the training weights, so two reads there need not score the same mix.
    weights = {args.task: 1.0}
    data = ComplexDataset(args.jsonl, args.crop_budget, args.contact_cutoff,
                            weights, args.seed, training=False,
                            mixed_task_weights=weights,
                            fixed_mixed_tasks=(args.task,),
                            fixed_atom_context=bool(saved.get('fixed_atom_context')),
                            log=lambda *a, **k: None)
    n = len(data) if not args.limit else min(args.limit, len(data))
    print('[infer] %d entr%s from %s'
          % (n, 'y' if n == 1 else 'ies', args.jsonl))

    # Seeded once for the whole split, after the model is built, so the
    # designs depend on the seed and the entries alone.
    torch.manual_seed(args.seed)
    extra = {'flow_steps': args.flow_steps or None}
    totals = collections.defaultdict(lambda: [0, 0])
    entry_means = collections.defaultdict(list)
    chain_means = collections.defaultdict(list)
    sample_entry = collections.defaultdict(lambda: collections.defaultdict(list))
    # (type, bucket) -> {sample: [segment recovery,...]}, the reported convention
    # plus the token counts and the class counts that macro F1 needs.
    seg = collections.defaultdict(lambda: collections.defaultdict(list))
    tok = collections.defaultdict(lambda: [0, 0])
    cls = collections.defaultdict(lambda: collections.defaultdict(
        lambda: [0, 0, 0]))
    native_nll = collections.defaultdict(lambda: [0., 0.])
    rows, skipped, failed, started = [], [], [], time.monotonic()
    for i in range(n):
        name = str(data[i].get('example_id') or i)
        try:
            batch = restrict(move(collate_complex([data[i]], knn_k=knn),
                                  device), types)
        except (ValueError, RuntimeError, KeyError) as exc:
            failed.append((name, str(exc)))
            continue
        name = batch['ids'][0] if batch.get('ids') else name
        if not int(batch['design_mask'].sum()):
            skipped.append(name)
            continue
        try:
            with torch.no_grad():
                out = model.sample(batch, temperature=args.temperature,
                                   num_samples=args.num_samples, **extra)
        except (ValueError, RuntimeError) as exc:
            failed.append((name, str(exc)))
            continue
        if args.nll:
            # The native sequence under the model, at the positions this task
            # designs. No sampling, so it is the same number every run.
            scored = batch['design_mask'].bool() & batch['loss_mask'].bool()
            with torch.no_grad():
                lp = model.marginal_log_probs(batch)
            for kind in TYPE_NAMES:
                here = scored & (batch['type_vec'] == TYPE_ID[kind])
                if not int(here.sum()):
                    continue
                sums = model.loss_sums(lp, dict(batch, loss_mask=here),
                                       label_smoothing=0.0)
                native_nll[kind][0] += float(sums['nll'])
                native_nll[kind][1] += float(sums['count'])
        tokens = out['S'].detach().view(args.num_samples, -1)
        native = batch['S']
        mask = batch['design_mask'].bool()
        for s in range(args.num_samples):
            got = per_type(tokens[s], native, mask, batch['type_vec'],
                           batch.get('chain_idx'))
            ci = batch.get('chain_idx')
            for kind, (hits, count, chains) in got.items():
                here = mask & (batch['type_vec'] == TYPE_ID[kind])
                specs = buckets_for(kind)
                if ci is None:
                    pairs = [(tokens[s][here].tolist(),
                              native[here].tolist(), -1)]
                else:
                    pairs = []
                    for c in chains:
                        sel = here & (ci == c)
                        pairs.append((tokens[s][sel].tolist(),
                                      native[sel].tolist(), int(sel.sum())))
                add_class_counts(cls, kind, pairs, specs)
                totals[kind][0] += hits
                totals[kind][1] += count
                entry_means[kind].append(hits / count)
                sample_entry[kind][s].append(hits / count)
                for h, c in chains.values():
                    chain_means[kind].append(h / c)
                    for lo, hi, bname in buckets_for(kind):
                        if lo < 0 or lo <= c < hi:
                            seg[(kind, bname)][s].append(h / c)
                            tok[(kind, bname)][0] += h
                            tok[(kind, bname)][1] += c
            rows.append(dict(entry=name, sample=s,
                             **{'%s_recovery' % k: '%.6f' % (h / c)
                                for k, (h, c, _) in got.items()},
                             **{'%s_positions' % k: c
                                for k, (_, c, _) in got.items()}))
        if (i + 1) % 25 == 0 or i + 1 == n:
            print('[infer] %d/%d, %.1f s' % (i + 1, n, time.monotonic() - started),
                  flush=True)

    if native_nll:
        print('[native] mean negative log-likelihood of the native sequence, '
              'every design position masked at once')
        for kind, (total, count) in sorted(native_nll.items()):
            print('[native] %-7s %.4f over %d position(s)'
                  % (kind, total / count, int(count)))
    if skipped:
        print('[infer] %d entr%s had nothing designable under task %r%s'
              % (len(skipped), 'y' if len(skipped) == 1 else 'ies', args.task,
                 ': ' + ', '.join(skipped[:5]) if len(skipped) <= 5 else ''))
    if failed:
        by_reason = collections.defaultdict(list)
        for name, why in failed:
            by_reason[why.split(' (')[0][:90]].append(name)
        print('[infer] %d of %d entr%s could not be built and were SKIPPED; '
              'every figure below is over the rest'
              % (len(failed), n, 'y' if len(failed) == 1 else 'ies'))
        for why, names in sorted(by_reason.items(),
                                 key=lambda kv: -len(kv[1])):
            print('          %3d  %s  e.g. %s'
                  % (len(names), why, ', '.join(names[:4])))
    if not totals:
        raise SystemExit('nothing was designed; check --task and --design-types')
    print('\nrecovery over DESIGNED positions. Three protocols, because one '
          'set of\nsequences gives four different means and the name is part '
          'of the number.')
    print('  %-8s %9s %9s %9s %8s %7s %10s %9s'
          % ('type', 'pooled', 'per entry', 'per chain', 'entries', 'chains',
             'positions', 'sample sd'))
    for kind in TYPE_NAMES:
        if kind not in totals:
            continue
        got = protocols(totals[kind], entry_means[kind], chain_means[kind],
                        [sum(v) / len(v) for v in sample_entry[kind].values()])
        print('  %-8s %9.4f %9.4f %9.4f %8d %7d %10d %9s'
              % (kind, got['pooled'], got['per_entry'], got['per_chain'],
                 got['entries'] // max(args.num_samples, 1), got['chains'],
                 got['positions'],
                 '-' if got['sample_sd'] != got['sample_sd']
                 else '%.4f' % got['sample_sd']))
    print('\nby chain length, the evaluation\'s own buckets. recovery is '
          'that convention:\na chain\'s token accuracy, averaged over a '
          'sample\'s chains, then over samples.')
    print('  %-8s %-14s %9s %9s %11s %7s %9s'
          % ('type', 'bucket', 'recovery', 'pooled', 'f1_present', 'chains',
             'positions'))
    strata = []
    for kind in TYPE_NAMES:
        for _, _, bname in buckets_for(kind):
            key = (kind, bname)
            if key not in seg:
                continue
            per_sample = [sum(v) / len(v) for v in seg[key].values()]
            hits, count = tok[key]
            f1 = macro_f1(cls[key]) if key in cls else float('nan')
            n_chains = sum(len(v) for v in seg[key].values()) // max(args.num_samples, 1)
            print('  %-8s %-14s %9.4f %9.4f %11s %7d %9d'
                  % (kind, bname, sum(per_sample) / len(per_sample),
                     hits / count if count else float('nan'),
                     '-' if f1 != f1 else '%.4f' % f1, n_chains, count))
            strata.append(dict(type=kind, bucket=bname,
                               recovery='%.6f' % (sum(per_sample) / len(per_sample)),
                               pooled='%.6f' % (hits / count) if count else '',
                               f1_present='' if f1 != f1 else '%.6f' % f1,
                               chains=n_chains, positions=count))
    print('\n  recovery   = per chain, then per sample, then across samples')
    print('  pooled     = every token of the bucket at once')
    print('  f1_present = macro F1 over the classes PRESENT in the target.')
    print('               The published tables average over each TYPE\'s own')
    print('               alphabet, 4 bases or 20 amino acids, so the two are')
    print('               the same wherever every class appears: always for')
    print('               nucleic acids, and for proteins long enough to show')
    print('               all 20. On short protein chains this one reads high.')
    print('\n  pooled    = every designed position of every sample at once')
    print('  per entry = an entry\'s positions pooled, then the entries')
    print('  per chain = a chain\'s positions pooled, then the chains')
    print('  sample sd = spread of the per-entry mean across the %d sample(s)'
          % args.num_samples)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        keys = sorted({k for r in rows for k in r})
        with open(args.out, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=['entry', 'sample'] +
                               [k for k in keys if k not in ('entry', 'sample')],
                               delimiter='\t')
            w.writeheader()
            w.writerows(rows)
        print('\n[infer] %d row(s) -> %s' % (len(rows), args.out))
        strat_path = os.path.splitext(args.out)[0] + '_by_length.tsv'
        with open(strat_path, 'w', newline='') as fh:
            w = csv.DictWriter(fh, delimiter='\t', fieldnames=[
                'type', 'bucket', 'recovery', 'pooled', 'f1_present',
                'chains', 'positions'])
            w.writeheader()
            w.writerows(strata)
        print('[infer] %d stratum/strata -> %s' % (len(strata), strat_path))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
