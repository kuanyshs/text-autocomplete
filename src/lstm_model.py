from typing import List, Optional
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from pathlib import Path

class LSTMAutocomplete(nn.Module):
    def __init__(self, vocab_size, emb_dim=512, hidden_dim=512, num_layers=1, dropout=0.2, pad_idx=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size, bias=False)
        # weight tying
        self.fc.weight = self.embedding.weight

    def forward(self, input_ids, lengths):
        emb = self.embedding(input_ids)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_out, _ = self.lstm(packed)
        T = input_ids.size(1)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=T)
        logits = self.fc(self.dropout(out))
        return logits

    @torch.no_grad()
    def generate_batch(
        self,
        prefixes: List[torch.LongTensor],
        max_new_tokens: int = 10,
        eos_id: Optional[int] = None,
        do_sample: bool = True,
        top_p: float = 0.8,
        temperature: float = 1.0,
        max_context: Optional[int] = None,
        device: Optional[torch.device] = None,
        ban_ids: Optional[List[int]] = None
    ):
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        proc = []
        for t in prefixes:
            if max_context is not None and len(t) > max_context:
                t = t[-max_context:]
            proc.append(t.to(device))

        lengths = torch.tensor([len(t) for t in proc], device=device, dtype=torch.long)
        padded = torch.nn.utils.rnn.pad_sequence(proc, batch_first=True, padding_value=self.pad_idx)

        emb = self.embedding(padded)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.lstm(packed)

        last_token = padded[torch.arange(padded.size(0), device=device), lengths - 1].unsqueeze(1)

        B = padded.size(0)
        generated = [[] for _ in range(B)]
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        ban_ids_t = None
        if ban_ids is not None and len(ban_ids) > 0:
            ban_ids_t = torch.tensor(ban_ids, device=device, dtype=torch.long)

        for _ in range(max_new_tokens):
            out, hidden = self.lstm(self.embedding(last_token), hidden)
            logits = self.fc(out[:, -1, :])

            if temperature != 1.0:
                logits = logits / temperature

            if ban_ids_t is not None:
                logits[:, ban_ids_t] = float("-inf")

            if do_sample:
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                    sorted_probs = torch.softmax(sorted_logits, dim=-1)
                    cumprobs = torch.cumsum(sorted_probs, dim=-1)
                    to_remove = cumprobs > top_p
                    to_remove[:, 1:] = to_remove[:, :-1].clone()
                    to_remove[:, 0] = False
                    sorted_logits = sorted_logits.masked_fill(to_remove, float("-inf"))
                    logits = torch.zeros_like(logits).scatter(1, sorted_idx, sorted_logits)
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            next_token_flat = next_token.squeeze(1)
            for b in range(B):
                if not finished[b]:
                    generated[b].append(int(next_token_flat[b].item()))

            if eos_id is not None:
                finished |= (next_token_flat == eos_id)
                if torch.all(finished):
                    break

            if eos_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(next_token, eos_id),
                    next_token
                )

            last_token = next_token

        return generated

def zero_pad_embedding_row(model: LSTMAutocomplete, pad_idx: int):
    with torch.no_grad():
        model.embedding.weight.data[pad_idx].zero_()