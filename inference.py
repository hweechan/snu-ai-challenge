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
    "I am showing you 4 images in order: Image 1, Image 2, Image 3, Image 4.\n\n"
    "This sentence describes the correct chronological order of the events shown in these images:\n"
    "\"{sentence}\"\n\n"
    "Your task: determine the correct chronological order of the 4 images.\n"
    "Think step by step:\n"
    "1. What does each image show?\n"
    "2. What sequence of events does the sentence describe?\n"
    "3. Which image matches each step in the sequence?\n\n"
    "Answer with ONLY a Python list of image numbers (1-4) in chronological order.\n"
    "Example: [3, 1, 4, 2] means Image 3 happens first, then Image 1, then Image 4, then Image 2.\n"
    "Answer:"
)


def parse_model_output(text: str) -> list[int]:
    match = re.search(r'\[(\d)\s*,\s*(\d)\s*,\s*(\d)\s*,\s*(\d)\]', text)
    if match:
        result = [int(match.group(i)) for i in range(1, 5)]
        if sorted(result) == [1, 2, 3, 4]:
            return result
        if sorted(result) == [0, 1, 2, 3]:
            return [x + 1 for x in result]
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
        generated_ids = model.generate(**inputs, max_new_tokens=256)

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

    if hasattr(predict, '_debug_count'):
        predict._debug_count += 1
    else:
        predict._debug_count = 1
    if predict._debug_count <= 10:
        print(f"  raw output: {output_text!r}")

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
    fallback_count = 0
    for i, (_, row) in enumerate(tqdm(eval_df.iterrows(), total=len(eval_df), desc="Validation")):
        pred = predict(row, model, processor, f"{args.data_dir}/train")
        gt = parse_answer(row["Answer"])
        if pred == [1, 2, 3, 4] and gt != [1, 2, 3, 4]:
            fallback_count += 1
        if pred == gt:
            correct += 1
        if i < 10:
            print(f"  [{i}] pred={pred} gt={gt} {'OK' if pred == gt else 'X'}")

    acc = correct / len(eval_df)
    print(f"Exact Match Accuracy: {acc:.4f} ({correct}/{len(eval_df)})")
    print(f"Fallback to [1,2,3,4] (parse fail): {fallback_count}/{len(eval_df)}")


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
