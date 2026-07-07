# SNU AI Challenge 2026 — SigLIP2 Baseline

## 태스크 설명

**주제: 텍스트로 풀어보는 장면의 재구성**

SNU AI Challenge 2026은 텍스트 프롬프트와 순서가 섞인 4장의 이미지가 주어졌을 때, 텍스트 내용에 맞는 올바른 이미지 순서를 예측하는 과제입니다.

- **입력**: 텍스트 프롬프트(Sentence) 1개 + 순서가 섞인 이미지 4장
- **출력**: 올바른 이미지 순서 (예: `[3, 1, 4, 2]`, 1-indexed)
- **평가**: Exact Match Accuracy — 4장의 순서가 완전히 일치해야만 정답으로 인정

## 핵심 아이디어: SigLIP2 + 4! 전수 탐색

[SigLIP2](https://huggingface.co/google/siglip2-base-patch16-384)는 이미지-텍스트 유사도를 계산하는 모델입니다.

1. 4장의 이미지를 특정 순열로 배치한 뒤, 384×384로 리사이즈하여 가로로 이어붙인 combined image를 생성
2. 4! = 24가지 가능한 순열을 모두 시도
3. 각 순열의 combined image와 텍스트 프롬프트 간 유사도(`logits_per_image`)를 계산
4. 유사도가 가장 높은 순열을 정답으로 선택

## 프로젝트 구조

```
snu-ai-challenge/
├── Dockerfile           # Docker 실행 환경
├── requirements.txt     # Python 의존성
├── README.md
├── inference.py         # 추론 및 검증 코드
├── train.py             # Fine-tuning 스켈레톤
├── dataset.py           # PyTorch Dataset 클래스
├── preprocess.py        # 전처리 스켈레톤 (GPT-4o 캡셔닝 예정)
├── utils.py             # 이미지 처리, 순열 변환 유틸리티
└── data/                # 데이터 디렉토리 (.gitignore 처리)
    ├── train.csv
    ├── test.csv
    ├── sample_submission.csv
    ├── train/           # 학습 이미지
    └── test/            # 테스트 이미지
```

## 실행 방법

### 1. RunPod 실행

```bash
# 레포 클론
git clone https://github.com/hweechan1007/snu-ai-challenge.git
cd snu-ai-challenge

# 데이터 업로드 (로컬에서 RunPod으로)
scp -P <PORT> -r data/ root@<RUNPOD_IP>:/workspace/snu-ai-challenge/data/

# Docker 빌드 및 실행
docker build -t snu-ai .
docker run --gpus all -v $(pwd)/data:/workspace/data snu-ai

# 검증 모드 실행
docker run --gpus all -v $(pwd)/data:/workspace/data snu-ai python inference.py --validate
```

### 2. 로컬 실행

```bash
git clone https://github.com/hweechan1007/snu-ai-challenge.git
cd snu-ai-challenge

# 데이터 디렉토리에 train.csv, test.csv, 이미지 폴더 배치
# data/train.csv, data/test.csv, data/train/*, data/test/*

pip install -r requirements.txt

# 추론 (submission.csv 생성)
python inference.py --data_dir data --output submission.csv

# 검증 (train 데이터 200개로 Exact Match Accuracy 측정)
python inference.py --validate --data_dir data
```

## 개발 환경

- Python 3.10
- PyTorch 2.3.0
- CUDA 12.1
- GPU: RTX 3090
- Docker base image: `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime`
