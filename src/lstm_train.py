import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt
    from IPython.display import clear_output, display
    _HAS_PLOT = True
except Exception:
    _HAS_PLOT = False

import yaml

from .data_utils import set_seed, load_lines, clean_and_tokenize, build_vocabulary, save_summary
from .next_token_dataset import build_dataloaders
from .lstm_model import LSTMAutocomplete, zero_pad_embedding_row
from .eval_lstm import evaluate_rouge_lstm

def fit_one_epoch(model, train_loader, val_loader, optimizer, criterion, device, pad_id, clip_norm=1.0):
    model.train()
    total_train_loss, total_train_tokens = 0.0, 0

    for batch in tqdm(train_loader, desc="Train"):
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        lengths = batch["lengths"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids, lengths)
        loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        loss.backward()
        if clip_norm is not None and clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

        with torch.no_grad():
            if hasattr(model, "embedding"):
                model.embedding.weight.data[pad_id].zero_()

        train_batch_tokens = (target_ids != pad_id).sum().item()
        total_train_loss += loss.item() * train_batch_tokens
        total_train_tokens += train_batch_tokens

    train_loss = total_train_loss / max(1, total_train_tokens)

    model.eval()
    total_val_loss, total_val_tokens = 0.0, 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Val"):
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            lengths = batch["lengths"].to(device)

            logits = model(input_ids, lengths)
            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            val_batch_tokens = (target_ids != pad_id).sum().item()
            total_val_loss += loss.item() * val_batch_tokens
            total_val_tokens += val_batch_tokens

    val_loss = total_val_loss / max(1, total_val_tokens)
    return train_loss, val_loss


def train_with_plotting(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    inv_vocab,
    PAD, BOS, EOS, UNK,
    device,
    max_len,
    num_epochs=10,
    quarter=0.25,
    do_sample=True,
    top_p=0.8,
    temperature=0.9,
    num_show=5,
    ckpt_dir="models",
    save_best=True,
    clip_norm=1.0,
    use_plot=True
):
    os.makedirs(ckpt_dir, exist_ok=True)

    hist = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "rouge1": [],
        "rouge2": [],
    }

    fig = None
    if use_plot and _HAS_PLOT:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax2 = ax.twinx()
        (line_r1,) = ax.plot([], [], label="rouge1", color="tab:blue", lw=2)
        (line_r2,) = ax.plot([], [], label="rouge2", color="tab:green", lw=2)
        (line_trn,) = ax2.plot([], [], label="train_loss", color="tab:red", lw=2, alpha=0.8)
        (line_val,) = ax2.plot([], [], label="val_loss", color="tab:brown", lw=2, alpha=0.8, linestyle="--")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("ROUGE")
        ax2.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        lines_all = [line_r1, line_r2, line_trn, line_val]
        labels_all = [l.get_label() for l in lines_all]
        ax.legend(lines_all, labels_all, loc="upper left")

    best_lstm_results = None
    best_lstm_examples = None
    best_val_loss = float("inf")
    best_ckpt_path = None

    for epoch in range(1, num_epochs + 1):
        train_loss, val_loss = fit_one_epoch(model, train_loader, val_loader, optimizer, criterion, device, PAD, clip_norm)

        lstm_results, lstm_examples = evaluate_rouge_lstm(
            model,
            val_loader,
            inv_vocab,
            PAD, BOS, EOS, UNK,
            quarter=quarter,
            max_context_tokens=max_len,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            device=device,
            num_show=num_show,
            max_new_tokens=10
        )

        hist["epoch"].append(epoch)
        hist["train_loss"].append(train_loss)
        hist["val_loss"].append(val_loss)
        hist["rouge1"].append(float(lstm_results.get("rouge1", np.nan)))
        hist["rouge2"].append(float(lstm_results.get("rouge2", np.nan)))

        if use_plot and _HAS_PLOT:
            ax = fig.axes[0]
            ax2 = fig.axes[1]
            line_r1, line_r2, line_trn, line_val = ax.lines[0], ax.lines[1], ax2.lines[0], ax2.lines[1]
            x = hist["epoch"]
            line_r1.set_data(x, hist["rouge1"])
            line_r2.set_data(x, hist["rouge2"])
            line_trn.set_data(x, hist["train_loss"])
            line_val.set_data(x, hist["val_loss"])
            ax.set_xlim(1, max(3, epoch))

            rouge_vals = np.array(hist["rouge1"] + hist["rouge2"], dtype=float)
            rouge_vals = rouge_vals[~np.isnan(rouge_vals)]
            if rouge_vals.size > 0:
                ymin = max(0.0, rouge_vals.min() - 0.02)
                ymax = min(1.0, rouge_vals.max() + 0.02)
                if ymax - ymin < 0.05:
                    ymax = min(1.0, ymin + 0.05)
                ax.set_ylim(ymin, ymax)

            loss_vals = np.array(hist["train_loss"] + hist["val_loss"], dtype=float)
            if loss_vals.size > 0:
                lmin, lmax = float(np.min(loss_vals)), float(np.max(loss_vals))
                pad = max(1e-6, 0.1 * (lmax - lmin if lmax > lmin else lmax if lmax > 0 else 1.0))
                ax2.set_ylim(max(0.0, lmin - pad), lmax + pad)

            ax.set_title(f"Epoch {epoch}/{num_epochs}")
            clear_output(wait=True)
            display(fig)
            plt.pause(0.001)

        if save_best and val_loss < best_val_loss:
            best_lstm_results = lstm_results
            best_lstm_examples = lstm_examples
            best_val_loss = val_loss
            best_ckpt_path = os.path.join(ckpt_dir, f"lstm_autocomplete_best.pt")
            torch.save({
                "model_state": model.state_dict(),
                "vocab_size": model.vocab_size,
                "pad_idx": PAD,
                "vocab": inv_vocab,  # при желании можно сохранять обе мапы
                "hist": hist
            }, best_ckpt_path)

    return hist, best_lstm_results, fig, best_lstm_examples


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 123))
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu" else "cpu")
    print(f"Using device: {device}")

    train_path = Path(cfg["data"]["train_path"])
    val_path = Path(cfg["data"]["val_path"])
    test_path = Path(cfg["data"]["test_path"])

    train_texts = load_lines(train_path)
    val_texts = load_lines(val_path)
    test_texts = load_lines(test_path)

    train_tokens = clean_and_tokenize(train_texts)
    val_tokens = clean_and_tokenize(val_texts)
    test_tokens = clean_and_tokenize(test_texts)

    vocab, inv_vocab = build_vocabulary(
        train_tokens,
        max_size=cfg["vocab"]["max_size"],
        min_freq=cfg["vocab"]["min_freq"]
    )
    PAD, BOS, EOS, UNK = vocab["[PAD]"], vocab["[BOS]"], vocab["[EOS]"], vocab["[UNK]"]
    vocab_size = len(vocab)
    print(f"Vocab size: {vocab_size}")

    max_len = cfg["train"]["max_len"]
    batch_size = cfg["train"]["batch_size"]
    pin_memory = (device.type == "cuda")

    train_loader, val_loader, test_loader = build_dataloaders(
        train_tokens, val_tokens, test_tokens, vocab,
        max_len=max_len, batch_size=batch_size, pin_memory=pin_memory
    )

    model = LSTMAutocomplete(
        vocab_size=vocab_size,
        emb_dim=cfg["model"]["emb_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        dropout=cfg["model"]["dropout"],
        pad_idx=PAD
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    criterion = nn.CrossEntropyLoss(ignore_index=PAD, reduction="mean")

    hist, best_lstm_results, fig, best_lstm_examples = train_with_plotting(
        model, train_loader, val_loader, optimizer, criterion,
        inv_vocab, PAD, BOS, EOS, UNK, device, max_len,
        num_epochs=cfg["train"]["epochs"],
        quarter=cfg["train"]["quarter"],
        do_sample=cfg["train"]["do_sample"],
        top_p=cfg["train"]["top_p"],
        temperature=cfg["train"]["temperature"],
        num_show=cfg["train"]["num_show"],
        ckpt_dir=cfg["train"]["ckpt_dir"],
        save_best=cfg["train"]["save_best"],
        clip_norm=cfg["train"]["clip_norm"],
        use_plot=cfg["train"].get("use_plot", False)
    )

    print("Best LSTM results:", best_lstm_results)
    if best_lstm_examples:
        print("Examples:")
        for ex in best_lstm_examples[:min(5, len(best_lstm_examples))]:
            print("PREFIX:", ex["prefix"])
            print("PRED:  ", ex["prediction"])
            print("REF:   ", ex["reference"])
            print("---")

    save_summary(best_lstm_results, best_lstm_examples, out_dir="models", filename_prefix="lstm_results", top_k=5)


if __name__ == "__main__":
    main()