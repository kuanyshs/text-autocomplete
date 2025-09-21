import math
from pathlib import Path

import torch
from tqdm import tqdm
import evaluate
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM

from .data_utils import load_lines, select_examples_by_indices, print_selected_examples, save_summary

@torch.no_grad()
def evaluate_rouge_gpt2(
    text_list,
    tokenizer,
    model,
    device,
    quarter=0.25,
    do_sample=True,
    top_k=50,
    top_p=0.8,
    temperature=0.9,
    num_show=256,
    max_length=32,
):
    model.eval()
    metric = evaluate.load("rouge")

    predictions, references, examples = [], [], []

    for text in tqdm(text_list, desc="Eval GPT-2"):
        enc = tokenizer(text, add_special_tokens=False)
        token_ids = enc.input_ids
        if len(token_ids) < 4:
            continue

        total_len = len(token_ids)
        prefix_len = max(1, int(math.ceil(total_len * (1 - quarter))))
        prefix_token_ids = token_ids[:prefix_len]
        target_token_ids = token_ids[prefix_len:]
        if len(target_token_ids) == 0:
            continue

        ctx_ids = prefix_token_ids[-max_length:]
        input_ids = torch.tensor([ctx_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids, device=device)

        target_length = len(target_token_ids)
        max_new_tokens = max(target_length, 10)

        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        gen_tokens = generated[0][input_ids.shape[-1]:]
        if len(gen_tokens) > target_length:
            gen_tokens = gen_tokens[:target_length]

        prediction = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        reference = tokenizer.decode(target_token_ids[:len(gen_tokens)], skip_special_tokens=True).strip()

        predictions.append(prediction)
        references.append(reference)

        if len(examples) < num_show:
            prefix_text = tokenizer.decode(ctx_ids, skip_special_tokens=True)
            full_ref = tokenizer.decode(target_token_ids, skip_special_tokens=True)
            examples.append({"prefix": prefix_text, "prediction": prediction, "reference": full_ref})

    if not predictions:
        print("Нет валидных примеров для оценки")
        return {"rouge1": 0.0, "rouge2": 0.0}, []

    results = metric.compute(predictions=predictions, references=references)
    print(f"Обработано {len(predictions)} примеров")
    return results, examples


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--subset_n", type=int, default=None, help="Сколько первых примеров взять для оценки")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu" else "cpu")
    print(f"Using device: {device}")

    val_path = Path(cfg["data"]["val_path"])
    val_texts = load_lines(val_path)
    
    test_path = Path(cfg["data"]["test_path"])
    test_texts = load_lines(test_path)
    
    if args.subset_n is not None:
        val_texts = val_texts[:args.subset_n]
        test_texts = test_texts[:args.subset_n]
    else:
        subset_n = cfg["transformer_eval"].get("subset_n", None)
        if subset_n is not None:
            val_texts = val_texts[:subset_n]
            test_texts = test_texts[:subset_n] 

    model_name = cfg["transformer_eval"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    val_results, val_examples = evaluate_rouge_gpt2(
        text_list=val_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        quarter=cfg["transformer_eval"]["quarter"],
        do_sample=True,
        top_k=cfg["transformer_eval"]["top_k"],
        top_p=cfg["transformer_eval"]["top_p"],
        temperature=cfg["transformer_eval"]["temperature"],
        num_show=cfg["transformer_eval"]["num_show"],
        max_length=cfg["transformer_eval"]["max_length"]
    )
    
    print("GPT-2 val results:", val_results)
    if val_examples:
        val_selected_examples = select_examples_by_indices(val_examples, cfg["transformer_eval"]["example_ids"])
        print_selected_examples(val_selected_examples)
    
    save_summary(val_results, val_selected_examples, out_dir="models", filename_prefix="gpt2_val_results", top_k=5)

    test_results, test_examples = evaluate_rouge_gpt2(
        text_list=test_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        quarter=cfg["transformer_eval"]["quarter"],
        do_sample=True,
        top_k=cfg["transformer_eval"]["top_k"],
        top_p=cfg["transformer_eval"]["top_p"],
        temperature=cfg["transformer_eval"]["temperature"],
        num_show=cfg["transformer_eval"]["num_show"],
        max_length=cfg["transformer_eval"]["max_length"]
    )
    
    print("GPT-2 test results:", test_results)
    if test_examples:
        print_selected_examples(test_examples)
    
    save_summary(val_results, val_examples, out_dir="models", filename_prefix="gpt2_test_results", top_k=5)

if __name__ == "__main__":
    main()