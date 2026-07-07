"""
Preprocessing pipeline for SNU AI Challenge 2026.
Skeleton for future GPT-4o Vision captioning integration.
"""

import pandas as pd


def generate_captions(csv_path: str, data_dir: str, output_path: str):
    # TODO: GPT-4o Vision API로 각 이미지에 대한 캡션 생성
    # TODO: 생성된 캡션을 Sentence와 결합하여 더 풍부한 텍스트 프롬프트 구성
    raise NotImplementedError


def augment_text_prompts(csv_path: str, output_path: str):
    # TODO: 텍스트 프롬프트 augmentation (패러프레이즈 등)
    raise NotImplementedError


if __name__ == "__main__":
    pass
