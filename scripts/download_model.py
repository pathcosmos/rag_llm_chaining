#!/usr/bin/env python3
"""
HuggingFace 모델 다운로드 스크립트 (속도 제한 지원)
aria2c를 사용하여 대역폭 제한 다운로드 수행
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_url, snapshot_download
from dotenv import load_dotenv

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# 사전 정의된 모델 목록
MODELS = {
    "solar-10.7b": {
        "repo_id": "upstage/SOLAR-10.7B-Instruct-v1.0",
        "description": "Korean-optimized 10.7B model (Apache 2.0)",
        "size_gb": 21
    },
    "qwen2.5-14b": {
        "repo_id": "Qwen/Qwen2.5-14B-Instruct",
        "description": "Larger Qwen model for better performance",
        "size_gb": 28
    },
    "qwen2.5-7b": {
        "repo_id": "Qwen/Qwen2.5-7B-Instruct",
        "description": "Current baseline model",
        "size_gb": 14
    },
    "llama3.1-8b": {
        "repo_id": "meta-llama/Llama-3.1-8B-Instruct",
        "description": "Meta's Llama 3.1 8B",
        "size_gb": 16
    }
}


def download_with_aria2(repo_id: str, local_dir: Path, max_speed_mbps: int = 80):
    """aria2c를 사용하여 속도 제한 다운로드"""
    api = HfApi()
    token = os.getenv("HF_TOKEN")

    # 파일 목록 가져오기
    files = api.list_repo_files(repo_id)

    # 다운로드 디렉토리 생성
    local_dir.mkdir(parents=True, exist_ok=True)

    # aria2c 입력 파일 생성
    aria2_input = local_dir / "aria2_input.txt"

    # 속도 제한 계산 (Mbps -> bytes/s)
    max_speed = f"{max_speed_mbps // 8}M"  # 80Mbps = 10MB/s

    print(f"📦 Downloading: {repo_id}")
    print(f"📁 Location: {local_dir}")
    print(f"⚡ Speed limit: {max_speed_mbps} Mbps ({max_speed}/s)")
    print(f"📄 Files: {len(files)}")
    print("-" * 50)

    with open(aria2_input, "w") as f:
        for filename in files:
            url = hf_hub_url(repo_id, filename)
            out_path = local_dir / filename
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # 이미 다운로드된 파일 건너뛰기
            if out_path.exists():
                print(f"  ✅ Skip (exists): {filename}")
                continue

            # aria2c 형식으로 URL 작성
            f.write(f"{url}\n")
            f.write(f"  out={filename}\n")
            if token:
                f.write(f"  header=Authorization: Bearer {token}\n")
            f.write("\n")

    # aria2c 실행
    cmd = [
        "aria2c",
        f"--input-file={aria2_input}",
        f"--dir={local_dir}",
        f"--max-download-limit={max_speed}",
        "--max-concurrent-downloads=2",
        "--split=4",
        "--max-connection-per-server=4",
        "--min-split-size=10M",
        "--continue=true",
        "--console-log-level=notice",
        "--summary-interval=30"
    ]

    print(f"\n🚀 Starting download with aria2c...")
    print(f"   Command: {' '.join(cmd[:5])}...")

    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✅ Download complete: {repo_id}")

        # 입력 파일 정리
        aria2_input.unlink(missing_ok=True)
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Download failed: {e}")
        return False


def download_with_huggingface(repo_id: str, local_dir: Path):
    """HuggingFace Hub를 사용한 기본 다운로드 (속도 제한 없음)"""
    token = os.getenv("HF_TOKEN")

    print(f"📦 Downloading: {repo_id}")
    print(f"📁 Location: {local_dir}")
    print("⚠️  No speed limit (using HuggingFace default)")
    print("-" * 50)

    try:
        # HF_HUB_ENABLE_HF_TRANSFER=0으로 느린 다운로드 사용
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            token=token,
            local_dir_use_symlinks=False
        )
        print(f"\n✅ Download complete: {repo_id}")
        return True

    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="HuggingFace 모델 다운로드 (속도 제한 지원)")
    parser.add_argument(
        "model",
        nargs="?",
        choices=list(MODELS.keys()) + ["custom"],
        help="다운로드할 모델"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        help="커스텀 HuggingFace 레포지토리 ID"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="다운로드 디렉토리"
    )
    parser.add_argument(
        "--speed-limit",
        type=int,
        default=80,
        help="최대 다운로드 속도 (Mbps, 기본값: 80)"
    )
    parser.add_argument(
        "--no-limit",
        action="store_true",
        help="속도 제한 없이 다운로드"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="사용 가능한 모델 목록 표시"
    )

    args = parser.parse_args()

    # 모델 목록 표시
    if args.list:
        print("\n📋 Available Models:")
        print("-" * 60)
        for name, info in MODELS.items():
            print(f"  {name:15} | {info['size_gb']:3}GB | {info['description']}")
        print("-" * 60)
        return

    # 모델 인자 필수 확인
    if not args.model:
        print("❌ Model argument required. Use --list to see available models.")
        parser.print_help()
        sys.exit(1)

    # 모델 정보 가져오기
    if args.model == "custom":
        if not args.repo_id:
            print("❌ --repo-id required for custom model")
            sys.exit(1)
        repo_id = args.repo_id
        model_name = repo_id.split("/")[-1]
    else:
        repo_id = MODELS[args.model]["repo_id"]
        model_name = args.model

    # 출력 디렉토리 설정
    if args.output_dir:
        local_dir = Path(args.output_dir)
    else:
        local_dir = PROJECT_ROOT / "models" / model_name

    # 다운로드 실행
    if args.no_limit:
        success = download_with_huggingface(repo_id, local_dir)
    else:
        success = download_with_aria2(repo_id, local_dir, args.speed_limit)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
