import argparse

import pandas as pd
import torch
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info

from utils import load_images

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

PROMPT_TEMPLATE = (
    "These are 4 images (Image 1, Image 2, Image 3, Image 4) shown in random order.\n"
    "Sentence: \"{sentence}\"\n"
    "Determine the correct chronological order of images based on the sentence.\n"
    "Reply with only a Python list like [3, 1, 4, 2]."
)


def train(args):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        min_pixels=128 * 28 * 28,
        max_pixels=256 * 28 * 28,
    )

    train_df = pd.read_csv(f"{args.data_dir}/train.csv")
    train_df = train_df[train_df["No_ordering"] == False].reset_index(drop=True)
    print(f"Training on {len(train_df)} samples")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )

    model.train()
    for epoch in range(args.epochs):
        df = train_df.sample(frac=1, random_state=epoch).reset_index(drop=True)
        total_loss = 0
        num_steps = 0

        pbar = tqdm(df.iterrows(), total=len(df), desc=f"Epoch {epoch+1}")
        optimizer.zero_grad()

        for idx, (_, row) in enumerate(pbar):
            try:
                images = load_images(f"{args.data_dir}/train", row)
            except Exception:
                continue

            sentence = row["Sentence"]
            answer_str = str(row["Answer"]).strip()

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
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": answer_str}],
                },
            ]

            full_text = processor.apply_chat_template(messages, tokenize=False)
            prompt_text = processor.apply_chat_template(
                messages[:1], tokenize=False, add_generation_prompt=True
            )

            image_inputs, video_inputs = process_vision_info(messages)

            try:
                full_inputs = processor(
                    text=[full_text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(model.device)

                prompt_inputs = processor(
                    text=[prompt_text],
                    images=image_inputs,
                    videos=video_inputs,
                    return_tensors="pt",
                )
                prompt_len = prompt_inputs["input_ids"].shape[1]

                labels = full_inputs["input_ids"].clone()
                labels[:, :prompt_len] = -100

                outputs = model(**full_inputs, labels=labels)
                loss = outputs.loss / args.grad_accum
                loss.backward()

                total_loss += outputs.loss.item()
                num_steps += 1

            except torch.cuda.OutOfMemoryError:
                print(f"\nOOM at sample {idx}, skipping")
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                continue

            if (idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if num_steps > 0:
                pbar.set_postfix(loss=f"{total_loss / num_steps:.4f}")

        if len(df) % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / max(num_steps, 1)
        print(f"Epoch {epoch+1}/{args.epochs}, Avg Loss: {avg_loss:.4f}")

        save_path = f"{args.output_dir}/epoch_{epoch+1}"
        model.save_pretrained(save_path)
        print(f"Saved: {save_path}")

    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Done. Final model: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="lora_adapter")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--grad_accum", type=int, default=4)
    args = parser.parse_args()
    train(args)
