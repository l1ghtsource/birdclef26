from collections import Counter

import torch
from torch.utils.data import BatchSampler, Subset, WeightedRandomSampler


def items_for_sampler(dataset):
    if isinstance(dataset, Subset):
        base = dataset.dataset
        return [base.items[i] for i in dataset.indices]
    return dataset.items


def primary_label_counts(items):
    cnt = Counter()
    for it in items:
        for lab in it.primary_labels:
            cnt[lab] += 1
    return cnt


def per_sample_weights_balanced(items, counts):
    w = []
    for it in items:
        labs = it.primary_labels
        if not labs:
            w.append(1.0)
            continue
        w.append(max(1.0 / max(counts[c], 1) for c in labs))
    return w


def per_sample_weights_square(items, counts):
    denom = float(sum(counts.values()))
    if denom <= 0:
        return [1.0] * len(items)
    w = []
    for it in items:
        labs = it.primary_labels
        if not labs:
            w.append(1.0)
            continue
        factors = []
        for c in labs:
            p = counts[c] / denom
            factors.append(float(p ** (-0.5)))
        w.append(max(factors))
    return w


class SourceBalancedBatchSampler(BatchSampler):
    def __init__(self, items, batch_size: int, ss_fraction: float, *, seed: int, drop_last: bool = True):
        if not 0.0 < ss_fraction < 1.0:
            raise ValueError(f"ss_fraction must be in (0, 1), got {ss_fraction}")
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.ss_per_batch = round(self.batch_size * float(ss_fraction))
        self.ss_per_batch = max(1, min(self.batch_size - 1, self.ss_per_batch))
        self.audio_per_batch = self.batch_size - self.ss_per_batch
        self.audio_idx = [
            i for i, item in enumerate(items) if item.source == "train_audio" or str(item.source).startswith("extra_")
        ]
        self.ss_idx = [i for i, item in enumerate(items) if item.source == "train_soundscapes"]
        if not self.audio_idx:
            raise ValueError("SourceBalancedBatchSampler: no train_audio items")
        if not self.ss_idx:
            raise ValueError("SourceBalancedBatchSampler: no train_soundscapes items")
        self.num_batches = (
            len(items) // self.batch_size if self.drop_last else (len(items) + self.batch_size - 1) // self.batch_size
        )

    def __iter__(self):
        gen = torch.Generator()
        gen.manual_seed(self.seed)
        audio = torch.as_tensor(self.audio_idx, dtype=torch.long)
        ss = torch.as_tensor(self.ss_idx, dtype=torch.long)
        for _ in range(self.num_batches):
            a = audio[torch.randint(0, len(audio), (self.audio_per_batch,), generator=gen)].tolist()
            s = ss[torch.randint(0, len(ss), (self.ss_per_batch,), generator=gen)].tolist()
            batch = a + s
            perm = torch.randperm(len(batch), generator=gen).tolist()
            yield [batch[i] for i in perm]

    def __len__(self):
        return self.num_batches


class PseudoBalancedBatchSampler(BatchSampler):
    def __init__(self, items, batch_size: int, pseudo_fraction: float, *, seed: int, drop_last: bool = True):
        if not 0.0 < pseudo_fraction < 1.0:
            raise ValueError(f"pseudo_fraction must be in (0, 1), got {pseudo_fraction}")
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.pseudo_per_batch = round(self.batch_size * float(pseudo_fraction))
        self.pseudo_per_batch = max(1, min(self.batch_size - 1, self.pseudo_per_batch))
        self.regular_per_batch = self.batch_size - self.pseudo_per_batch
        self.pseudo_idx = [i for i, item in enumerate(items) if bool(getattr(item, "is_pseudo", False))]
        self.regular_idx = [i for i, item in enumerate(items) if not bool(getattr(item, "is_pseudo", False))]
        if not self.pseudo_idx:
            raise ValueError("PseudoBalancedBatchSampler: no pseudo-labeled items")
        if not self.regular_idx:
            raise ValueError("PseudoBalancedBatchSampler: no regular items")
        self.num_batches = (
            len(items) // self.batch_size if self.drop_last else (len(items) + self.batch_size - 1) // self.batch_size
        )

    def __iter__(self):
        gen = torch.Generator()
        gen.manual_seed(self.seed)
        reg = torch.as_tensor(self.regular_idx, dtype=torch.long)
        pse = torch.as_tensor(self.pseudo_idx, dtype=torch.long)
        for _ in range(self.num_batches):
            r = reg[torch.randint(0, len(reg), (self.regular_per_batch,), generator=gen)].tolist()
            p = pse[torch.randint(0, len(pse), (self.pseudo_per_batch,), generator=gen)].tolist()
            batch = r + p
            perm = torch.randperm(len(batch), generator=gen).tolist()
            yield [batch[i] for i in perm]

    def __len__(self):
        return self.num_batches


def get_source_balanced_batch_sampler(cfg: dict, dataset):
    raw = cfg.get("ss_sampling_weight", "none")
    if raw in (None, "none", "None"):
        return None
    items = items_for_sampler(dataset)
    return SourceBalancedBatchSampler(
        items,
        batch_size=int(cfg["bs"]),
        ss_fraction=float(raw),
        seed=int(cfg["seed"]),
        drop_last=True,
    )


def get_pseudo_balanced_batch_sampler(cfg: dict, dataset):
    raw = cfg.get("pseudo_sampling_weight", "none")
    if raw in (None, "none", "None"):
        return None
    items = items_for_sampler(dataset)
    return PseudoBalancedBatchSampler(
        items,
        batch_size=int(cfg["bs"]),
        pseudo_fraction=float(raw),
        seed=int(cfg["seed"]),
        drop_last=True,
    )


def get_sampler(cfg: dict, dataset):
    items = items_for_sampler(dataset)
    n = len(items)
    counts = primary_label_counts(items)
    name = cfg["sampler"]

    if name == "none":
        return None

    if name == "balanced":
        weights_list = per_sample_weights_balanced(items, counts)
    elif name == "square_balanced":
        weights_list = per_sample_weights_square(items, counts)
    else:
        raise NotImplementedError

    weights = torch.as_tensor(weights_list, dtype=torch.double)
    seed = cfg["seed"]
    gen = torch.Generator()
    gen.manual_seed(seed)

    return WeightedRandomSampler(
        weights=weights,
        num_samples=n,
        replacement=True,
        generator=gen,
    )
