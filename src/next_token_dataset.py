from typing import List, Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence


class TextAutocompleteDataset(Dataset):
    def __init__(self, tokenized_texts: List[List[str]], vocab: Dict[str, int], max_len: int = 32):
        self.data = tokenized_texts
        self.vocab = vocab
        self.max_len = max_len
        self.pad_id = vocab["[PAD]"]
        self.bos_id = vocab["[BOS]"]
        self.eos_id = vocab["[EOS]"]
        self.unk_id = vocab["[UNK]"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]
        ids = [self.vocab.get(t, self.unk_id) for t in tokens]
        sequence = [self.bos_id] + ids + [self.eos_id]
        sequence = sequence[: self.max_len + 1]
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        target_ids = torch.tensor(sequence[1:], dtype=torch.long)
        return input_ids, target_ids


class SortedBatchSampler(Sampler):
    def __init__(self, lengths, batch_size, shuffle=True, drop_last=False):
        self.lengths = np.array(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.sorted_idx = np.argsort(self.lengths)
        self.batches = []
        for i in range(0, len(self.sorted_idx), batch_size):
            batch = self.sorted_idx[i:i + batch_size]
            if not drop_last or len(batch) == batch_size:
                self.batches.append(batch)

    def __iter__(self):
        if self.shuffle:
            import random
            random.shuffle(self.batches)
        for batch in self.batches:
            yield batch.tolist()

    def __len__(self):
        return len(self.batches)


def collate_with_padding(pad_id: int):
    def _fn(batch):
        inputs, targets = zip(*batch)
        lengths = torch.tensor([len(x) for x in inputs], dtype=torch.long)
        lengths_sorted, sort_idx = lengths.sort(descending=True)
        inputs = [inputs[i] for i in sort_idx]
        targets = [targets[i] for i in sort_idx]
        inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=pad_id)
        targets_padded = pad_sequence(targets, batch_first=True, padding_value=pad_id)
        return {"input_ids": inputs_padded, "target_ids": targets_padded, "lengths": lengths_sorted}
    return _fn


def build_dataloaders(
    train_tokens: List[List[str]],
    val_tokens: List[List[str]],
    test_tokens: List[List[str]],
    vocab: Dict[str, int],
    max_len: int = 32,
    batch_size: int = 256,
    pin_memory: bool = True,
    num_workers: int = 0
):
    PAD = vocab["[PAD]"]

    train_dataset = TextAutocompleteDataset(train_tokens, vocab, max_len)
    val_dataset = TextAutocompleteDataset(val_tokens, vocab, max_len)
    test_dataset = TextAutocompleteDataset(test_tokens, vocab, max_len)

    train_lengths = [min(len(t) + 2, max_len) for t in train_tokens]
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=SortedBatchSampler(train_lengths, batch_size, shuffle=True),
        collate_fn=collate_with_padding(PAD),
        pin_memory=pin_memory,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_with_padding(PAD),
        pin_memory=pin_memory,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_with_padding(PAD),
        pin_memory=pin_memory,
        num_workers=num_workers
    )
    return train_loader, val_loader, test_loader
