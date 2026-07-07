import argparse

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from utils import ALL_PERMS, concat_images, load_images, perm_to_answer, parse_answer

MODEL_NAME = "google/siglip2-base-patch16-384"


def predict(row, model, processor, data_dir: str, device: str) -> list[int]:
    images = load_images(data_dir, row)
    sentence = row["Sentence"]

    candidates = []
    for perm in ALL_PERMS:
        ordered = [images[i] for i in perm]
        combined = concat_images(ordered)
        candidates.append(combined)

    inputs = processor(
        text=[sentence] * len(candidates),
        images=candidates,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits_per_image.diag()
    best_idx = logits.argmax().item()
    best_perm = ALL_PERMS[best_idx]

    return perm_to_answer(best_perm)


def run_inference(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    test_df = pd.read_csv(f"{args.data_dir}/test.csv")

    results = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Inference"):
        answer = predict(row, model, processor, f"{args.data_dir}/test", device)
        results.append({"Id": row["Id"], "Answer": answer})

    submission = pd.DataFrame(results)
    submission.to_csv(args.output, index=False)
    print(f"Saved to {args.output} ({len(submission)} samples)")


def validate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    train_df = pd.read_csv(f"{args.data_dir}/train.csv")
    eval_df = train_df[train_df["No_ordering"] == False].head(200).reset_index(drop=True)
    print(f"Evaluating on {len(eval_df)} samples")

    correct = 0
    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Validation"):
        pred = predict(row, model, processor, f"{args.data_dir}/train", device)
        gt = parse_answer(row["Answer"])
        if pred == gt:
            correct += 1

    acc = correct / len(eval_df)
    print(f"Exact Match Accuracy: {acc:.4f} ({correct}/{len(eval_df)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output", type=str, default="submission.csv")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    if args.validate:
        validate(args)
    else:
        run_inference(args)
