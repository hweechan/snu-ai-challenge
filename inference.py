import argparse
import re

import pandas as pd
import torch
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from utils import load_images, parse_answer

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

PROMPT_TEMPLATE = (
    "You are given 4 images labeled Image 1, Image 2, Image 3, Image 4, "
    "and a sentence describing the correct sequence of events.\n"
    "Sentence: {sentence}\n"
    "Arrange the 4 images in the correct order that matches the sentence.\n"
    "Reply with only a Python list like [2, 1, 4, 3] using the image numbers.\n"
    "Do not explain."
)


def parse_model_output(text: str) -> list[int]:
    match = re.search(r'\[(\d)\s*,\s*(\d)\s*,\s*(\d)\s*,\s*(\d)\]', text)
    if match:
        result = [int(match.group(i)) for i in range(1, 5)]
        if sorted(result) == [1, 2, 3, 4]:
            return result
    return [1, 2, 3, 4]


def predict(row, model, processor, data_dir: str) -> list[int]:
    images = load_images(data_dir, row)
    sentence = row["Sentence"]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": images[0]},
                {"type": "image", "image": images[1]},
                {"type": "image", "image": images[2]},
                {"type": "image", "image": images[3]},
                {"type": "text", "text": PROMPT_TEMPLATE.format(sentence=sentence)},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=64)

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

    return parse_model_output(output_text)


def run_inference(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    test_df = pd.read_csv(f"{args.data_dir}/test.csv")

    results = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Inference"):
        if row.get("No_ordering", False):
            answer = [1, 2, 3, 4]
        else:
            answer = predict(row, model, processor, f"{args.data_dir}/test")
        results.append({"Id": row["Id"], "Answer": answer})

    submission = pd.DataFrame(results)
    submission.to_csv(args.output, index=False)
    print(f"Saved to {args.output} ({len(submission)} samples)")


def validate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    train_df = pd.read_csv(f"{args.data_dir}/train.csv")
    eval_df = train_df[train_df["No_ordering"] == False].head(200).reset_index(drop=True)
    print(f"Evaluating on {len(eval_df)} samples")

    correct = 0
    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Validation"):
        pred = predict(row, model, processor, f"{args.data_dir}/train")
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
