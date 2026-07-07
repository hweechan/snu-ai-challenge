import ast
from itertools import permutations
from PIL import Image


ALL_PERMS = list(permutations(range(4)))

TARGET_H = 384
TARGET_W = 384


def parse_answer(answer_str: str) -> list[int]:
    return ast.literal_eval(answer_str)


def load_images(data_dir: str, row) -> list[Image.Image]:
    sample_id = row["Id"]
    images = []
    for col in ["Input_1", "Input_2", "Input_3", "Input_4"]:
        path = f"{data_dir}/{sample_id}/{row[col]}"
        images.append(Image.open(path).convert("RGB"))
    return images


def concat_images(images: list[Image.Image]) -> Image.Image:
    resized = [img.resize((TARGET_W, TARGET_H)) for img in images]
    total_w = TARGET_W * len(resized)
    combined = Image.new("RGB", (total_w, TARGET_H))
    for i, img in enumerate(resized):
        combined.paste(img, (i * TARGET_W, 0))
    return combined


def perm_to_answer(perm: tuple[int, ...]) -> list[int]:
    answer = [0] * 4
    for new_pos, orig_idx in enumerate(perm):
        answer[orig_idx] = new_pos + 1
    return answer
