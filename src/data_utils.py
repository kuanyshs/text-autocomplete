import re
import random
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split
from nltk.tokenize import TweetTokenizer


def set_seed(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    except Exception:
        pass


def load_tweets(filepath: Path, limit: Optional[int] = None) -> List[str]:
    texts = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if not line or len(line) < 5:
                continue
            if set(line) <= set("-= "):
                continue
            texts.append(line.rstrip("\n"))
            if limit and len(texts) >= limit:
                break
    return texts


def load_lines(filepath: Path, limit: Optional[int] = None) -> List[str]:
    texts = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            line = line.rstrip("\n")
            if not line:
                continue
            texts.append(line)
            if limit and len(texts) >= limit:
                break
    return texts


def save_lines(filepath: Path, lines: List[str]):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.strip() + "\n")


def clean_and_tokenize(texts: List[str]) -> List[List[str]]:
    tokenizer = TweetTokenizer(
        preserve_case=False,
        strip_handles=False,
        reduce_len=True
    )
    tokenized = []
    for text in texts:
        text = re.sub(r"http\S+|www\.\S+", "<url>", text)
        text = re.sub(r"@\w+", "<user>", text)
        text = text.replace("’", "'")
        text = re.sub(r"\s+", " ", text).strip()
        tokens = tokenizer.tokenize(text)
        tokens = ["<user>" if (t.startswith("@") and len(t) > 1) else t for t in tokens]
        tokens = re.sub(r"<\s+(\w+)\s+>", r"<\1>", " ".join(tokens)).split()
        if tokens:
            tokenized.append(tokens)
    return tokenized


def clean_text_for_gpt2(texts: List[str]) -> List[str]:
    cleaned = []
    for text in texts:
        text = re.sub(r"http\S+|www\.\S+", "<url>", text)
        text = re.sub(r"@\w+", "<user>", text)
        text = text.replace("’", "'")
        text = re.sub(r"\s+", " ", text).strip()
        cleaned.append(text)
    return cleaned


def build_vocabulary(token_lists: List[List[str]], max_size: int = 10000, min_freq: int = 5) -> Tuple[Dict[str, int], Dict[int, str]]:
    counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)
    special_tokens = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
    frequent_words = [w for w, c in counter.most_common() if c >= min_freq]
    vocab_words = frequent_words[: max_size - len(special_tokens)]
    vocab = {token: idx for idx, token in enumerate(special_tokens + vocab_words)}
    inv_vocab = {idx: token for token, idx in vocab.items()}
    return vocab, inv_vocab


def ids_to_text(ids, inv_vocab: Dict[int, str], PAD: int, BOS: int, EOS: int, skip_unk: bool = False) -> str:
    import torch
    if torch.is_tensor(ids):
        ids = ids.detach().cpu().tolist()
    words = []
    for i in ids:
        if i in (PAD, BOS, EOS):
            continue
        tok = inv_vocab.get(int(i), "[UNK]")
        if skip_unk and tok == "[UNK]":
            continue
        words.append(tok)
    return " ".join(words)


def split_and_save_texts(
    input_path: Path,
    output_dir: Path,
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 123,
    limit: Optional[int] = None
):
    raw_texts = load_tweets(input_path, limit=limit)
    cleaned_texts = clean_text_for_gpt2(raw_texts)
    all_idx = np.arange(len(cleaned_texts))
    train_idx, temp_idx = train_test_split(all_idx, test_size=(val_size + test_size), random_state=seed)
    rel_val = val_size / (val_size + test_size) if (val_size + test_size) > 0 else 0.0
    val_idx, test_idx = train_test_split(temp_idx, test_size=(1 - rel_val), random_state=seed)

    train_texts = [cleaned_texts[i] for i in train_idx]
    val_texts = [cleaned_texts[i] for i in val_idx]
    test_texts = [cleaned_texts[i] for i in test_idx]

    output_dir.mkdir(parents=True, exist_ok=True)
    save_lines(output_dir / "dataset_processed.txt", cleaned_texts)
    save_lines(output_dir / "train.txt", train_texts)
    save_lines(output_dir / "val.txt", val_texts)
    save_lines(output_dir / "test.txt", test_texts)
    return {
        "train": train_texts,
        "val": val_texts,
        "test": test_texts
    }

def select_examples_by_indices(examples: List[Dict], indices: List[int]) -> List[Dict]:
    return [examples[i] for i in indices]

def print_selected_examples(selected_examples: List[Dict]) -> None:
    for ex in selected_examples:
        print("Examples:")
        print("PREFIX: ", ex["prefix"])
        print("PRED:   ", ex["prediction"])
        print("REF:    ", ex["reference"])
        print("---")


def save_summary(best_results, best_examples, out_dir="models", filename_prefix="results", top_k=5):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / f"{filename_prefix}.txt"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Best results: {best_results}\n")
        if best_examples:
            f.write("Examples:\n")
            for ex in best_examples[:min(top_k, len(best_examples))]:
                f.write("PREFIX: " + ex["prefix"] + "\n")
                f.write("PRED:   " + ex["prediction"] + "\n")
                f.write("REF:    " + ex["reference"] + "\n")
                f.write("---\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Путь к raw датасету (data/raw_dataset.txt)")
    parser.add_argument("--output_dir", type=str, default="data", help="Куда сохранить split")
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    split_and_save_texts(Path(args.input), Path(args.output_dir), args.val_size, args.test_size, args.seed, args.limit)
    print("Готово: train/val/test сохранены.")