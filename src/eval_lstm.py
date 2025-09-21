import math
from typing import List, Optional
import torch
from tqdm import tqdm
import evaluate

from .data_utils import ids_to_text


@torch.no_grad()
def evaluate_rouge_lstm(
    model,
    loader,
    inv_vocab,
    PAD, BOS, EOS, UNK,
    quarter: float = 0.25,
    max_context_tokens: int = 32,
    do_sample: bool = True,
    top_p: float = 0.8,
    temperature: float = 0.9,
    device: Optional[torch.device] = None,
    ban_ids: Optional[List[int]] = None,
    num_show: int = 5,
    max_new_tokens: int = 10
):
    if device is None:
        device = next(model.parameters()).device
    if ban_ids is None:
        ban_ids = [PAD, UNK]

    metric = evaluate.load("rouge")
    examples = []
    model.eval()
    any_added = False

    for batch in tqdm(loader, desc="Eval LSTM"):
        tgt = batch["target_ids"]

        prefixes = []
        ref_texts = []
        ref_lens = []
        prefix_texts_dbg = []

        B = tgt.size(0)
        for i in range(B):
            seq = tgt[i].tolist()
            if PAD in seq:
                seq = seq[: seq.index(PAD)]
            if EOS in seq:
                seq = seq[: seq.index(EOS)]
            if len(seq) < 4:
                continue

            total_len = len(seq)
            prefix_len = max(1, int(math.ceil(total_len * (1 - quarter))))
            target_ids = seq[prefix_len:]
            if len(target_ids) == 0:
                continue
            prefix_ids = torch.tensor([BOS] + seq[:prefix_len], dtype=torch.long, device=device)

            prefixes.append(prefix_ids)
            ref_texts.append(ids_to_text(target_ids, inv_vocab, PAD, BOS, EOS))
            ref_lens.append(len(target_ids))
            if len(prefix_texts_dbg) < num_show:
                prefix_texts_dbg.append(ids_to_text(seq[:prefix_len], inv_vocab, PAD, BOS, EOS))

        if not prefixes:
            continue

        gens = model.generate_batch(
            prefixes=prefixes,
            max_new_tokens=max_new_tokens,
            eos_id=EOS,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            max_context=max_context_tokens,
            device=device,
            ban_ids=ban_ids
        )

        pred_texts = [ids_to_text(gen_ids[:L], inv_vocab, PAD, BOS, EOS) for gen_ids, L in zip(gens, ref_lens)]
        metric.add_batch(predictions=pred_texts, references=ref_texts)
        any_added = True

        for k in range(min(num_show - len(examples), len(pred_texts))):
            examples.append({
                "prefix": prefix_texts_dbg[k],
                "prediction": pred_texts[k],
                "reference": ref_texts[k]
            })
            if len(examples) >= num_show:
                break

    if not any_added:
        return {"rouge1": 0.0, "rouge2": 0.0}, []

    results = metric.compute()
    return results, examples