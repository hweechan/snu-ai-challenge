import argparse
import math

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from qwen_vl_utils import process_vision_info

from utils import load_images

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

PROMPT_TEMPLATE = (
    "These are 4 images (Image 1, Image 2, Image 3, Image 4) shown in random order.\n"
    'Sentence: "{sentence}"\n'
    "Determine the correct chronological order of images based on the sentence.\n"
    "Reply with only a Python list like [3, 1, 4, 2]."
)


class TrainDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        df = pd.read_csv(csv_path)
        self.df = df[df["No_ordering"] == False].reset_index(drop=True)
        self.data_dir = data_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            images = load_images(self.data_dir, row)
        except Exception:
            return None
        return {
            "images": images,
            "sentence": row["Sentence"],
            "answer": str(row["Answer"]).strip(),
        }


def collate_fn(batch):
    return [s for s in batch if s is not None]


def find_assistant_start(input_ids, tokenizer):
    """Find the token position where the assistant's answer starts."""
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    assistant_token_ids = tokenizer.encode("assistant", add_special_tokens=False)

    input_list = input_ids[0].tolist()
    last_assistant_pos = -1

    for i in range(len(input_list) - len(assistant_token_ids)):
        if input_list[i] == im_start_id:
            chunk = input_list[i + 1 : i + 1 + len(assistant_token_ids)]
            if chunk == assistant_token_ids:
                last_assistant_pos = i + 1 + len(assistant_token_ids) + 1

    return last_assistant_pos


def train(args):
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="sdpa",
    )

    if args.resume_adapter:
        print(f"Resuming from adapter: {args.resume_adapter}")
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        model = PeftModel.from_pretrained(
            model,
            args.resume_adapter,
            is_trainable=True,
        )
    else:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        min_pixels=args.min_pixels * 28 * 28,
        max_pixels=args.max_pixels * 28 * 28,
    )

    dataset = TrainDataset(f"{args.data_dir}/train.csv", f"{args.data_dir}/train")
    print(f"Training on {len(dataset)} samples")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )

    total_steps = len(dataloader) * args.epochs // args.grad_accum
    warmup_steps = min(int(total_steps * 0.05), 50)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    effective_batch = args.batch_size * args.grad_accum

    model.train()
    global_step = 0
    sample_count = 0

    for epoch in range(args.epochs):
        total_loss = 0
        epoch_samples = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        optimizer.zero_grad()

        for batch in pbar:
            if not batch:
                continue

            for sample in batch:
                images = sample["images"]
                sentence = sample["sentence"]
                answer = sample["answer"]

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": images[0]},
                            {"type": "image", "image": images[1]},
                            {"type": "image", "image": images[2]},
                            {"type": "image", "image": images[3]},
                            {
                                "type": "text",
                                "text": PROMPT_TEMPLATE.format(sentence=sentence),
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": answer}],
                    },
                ]

                full_text = processor.apply_chat_template(
                    messages, tokenize=False
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

                    labels = full_inputs["input_ids"].clone()

                    assistant_start = find_assistant_start(
                        full_inputs["input_ids"], processor.tokenizer
                    )
                    if assistant_start > 0:
                        labels[:, :assistant_start] = -100
                    else:
                        answer_ids = processor.tokenizer.encode(
                            answer, add_special_tokens=False
                        )
                        prompt_len = (
                            full_inputs["input_ids"].shape[1] - len(answer_ids) - 1
                        )
                        labels[:, :prompt_len] = -100

                    outputs = model(**full_inputs, labels=labels)
                    loss = outputs.loss / effective_batch
                    loss.backward()

                    total_loss += outputs.loss.item()
                    epoch_samples += 1
                    sample_count += 1

                except torch.cuda.OutOfMemoryError:
                    print("\nOOM, skipping")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue

            global_step += 1

            if global_step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if epoch_samples > 0:
                current_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix(
                    loss=f"{total_loss / epoch_samples:.4f}",
                    lr=f"{current_lr:.2e}",
                )

            if sample_count > 0 and sample_count % args.save_every == 0:
                save_path = f"{args.output_dir}/step_{sample_count}"
                model.save_pretrained(save_path)
                print(f"\nCheckpoint saved: {save_path}")

            if args.max_steps and global_step >= args.max_steps:
                print(f"\nReached max_steps={args.max_steps}, stopping.")
                break

        if global_step % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = total_loss / max(epoch_samples, 1)
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
    parser.add_argument("--resume_adapter", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--min_pixels", type=int, default=64)
    parser.add_argument("--max_pixels", type=int, default=128)
    args = parser.parse_args()
    train(args)
