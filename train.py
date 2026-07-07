"""
Fine-tuning skeleton for SNU AI Challenge 2026.
"""

import argparse


def train(args):
    # TODO: SigLIP2 fine-tuning with scene ordering objective
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()
    train(args)
