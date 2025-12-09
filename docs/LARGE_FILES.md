# 대용량 파일 다운로드 가이드

이 프로젝트의 대용량 파일들은 Git에 포함되지 않으며, SeaweedFS에 별도로 저장되어 있습니다.

## SeaweedFS 서버 정보

- **Base URL**: `http://211.231.121.68:8888/lanco/rag_llm_chaining/`
- **총 용량**: 약 38GB

---

## 파일 목록 및 다운로드 URI

### 1. SOLAR-10.7B 모델 (20GB)

| 파일명 | 크기 | 다운로드 URI |
|--------|------|--------------|
| model-00001-of-00005.safetensors | 4.6GB | http://211.231.121.68:8888/lanco/rag_llm_chaining/models/solar-10.7b/model-00001-of-00005.safetensors |
| model-00002-of-00005.safetensors | 4.7GB | http://211.231.121.68:8888/lanco/rag_llm_chaining/models/solar-10.7b/model-00002-of-00005.safetensors |
| model-00003-of-00005.safetensors | 4.6GB | http://211.231.121.68:8888/lanco/rag_llm_chaining/models/solar-10.7b/model-00003-of-00005.safetensors |
| model-00004-of-00005.safetensors | 4.6GB | http://211.231.121.68:8888/lanco/rag_llm_chaining/models/solar-10.7b/model-00004-of-00005.safetensors |
| model-00005-of-00005.safetensors | 1.6GB | http://211.231.121.68:8888/lanco/rag_llm_chaining/models/solar-10.7b/model-00005-of-00005.safetensors |
| config.json | 685B | http://211.231.121.68:8888/lanco/rag_llm_chaining/models/solar-10.7b/config.json |
| generation_config.json | 154B | http://211.231.121.68:8888/lanco/rag_llm_chaining/models/solar-10.7b/generation_config.json |
| model.safetensors.index.json | 35KB | http://211.231.121.68:8888/lanco/rag_llm_chaining/models/solar-10.7b/model.safetensors.index.json |
| tokenizer.json | 1.7MB | http://211.231.121.68:8888/lanco/rag_llm_chaining/models/solar-10.7b/tokenizer.json |
| tokenizer.model | 482KB | http://211.231.121.68:8888/lanco/rag_llm_chaining/models/solar-10.7b/tokenizer.model |
| tokenizer_config.json | 1.4KB | http://211.231.121.68:8888/lanco/rag_llm_chaining/models/solar-10.7b/tokenizer_config.json |

### 2. Qwen2.5-14B 모델 (7.4GB, 부분 다운로드)

> ⚠️ 주의: 이 모델은 다운로드가 완료되지 않은 상태입니다 (2/8 파일만 존재)

| 파일명 | 크기 | 다운로드 URI |
|--------|------|--------------|
| model-00001-of-00008.safetensors | 3.6GB | http://211.231.121.68:8888/lanco/rag_llm_chaining/models/qwen2.5-14b/model-00001-of-00008.safetensors |
| model-00002-of-00008.safetensors | 3.7GB | http://211.231.121.68:8888/lanco/rag_llm_chaining/models/qwen2.5-14b/model-00002-of-00008.safetensors |
| config.json | 663B | http://211.231.121.68:8888/lanco/rag_llm_chaining/models/qwen2.5-14b/config.json |
| generation_config.json | 242B | http://211.231.121.68:8888/lanco/rag_llm_chaining/models/qwen2.5-14b/generation_config.json |
| merges.txt | 1.6MB | http://211.231.121.68:8888/lanco/rag_llm_chaining/models/qwen2.5-14b/merges.txt |

### 3. ChromaDB 임베딩 데이터 (8.9GB)

| 파일명 | 크기 | 다운로드 URI |
|--------|------|--------------|
| chroma.sqlite3 | 6.4GB | http://211.231.121.68:8888/lanco/rag_llm_chaining/data/embeddings/chroma/chroma.sqlite3 |
| 70b6e63d.../data_level0.bin | 2.4GB | http://211.231.121.68:8888/lanco/rag_llm_chaining/data/embeddings/chroma/70b6e63d-2c1f-407d-80ef-85ae8a6d9ee7/data_level0.bin |
| 70b6e63d.../index_metadata.pickle | 27MB | http://211.231.121.68:8888/lanco/rag_llm_chaining/data/embeddings/chroma/70b6e63d-2c1f-407d-80ef-85ae8a6d9ee7/index_metadata.pickle |
| 70b6e63d.../length.bin | 2.3MB | http://211.231.121.68:8888/lanco/rag_llm_chaining/data/embeddings/chroma/70b6e63d-2c1f-407d-80ef-85ae8a6d9ee7/length.bin |
| 70b6e63d.../link_lists.bin | 4.9MB | http://211.231.121.68:8888/lanco/rag_llm_chaining/data/embeddings/chroma/70b6e63d-2c1f-407d-80ef-85ae8a6d9ee7/link_lists.bin |
| 70b6e63d.../header.bin | 100B | http://211.231.121.68:8888/lanco/rag_llm_chaining/data/embeddings/chroma/70b6e63d-2c1f-407d-80ef-85ae8a6d9ee7/header.bin |
| 4dcab32f.../data_level0.bin | 21MB | http://211.231.121.68:8888/lanco/rag_llm_chaining/data/embeddings/chroma/4dcab32f-2e46-4cc0-8cc9-d09c76ac0d04/data_level0.bin |
| 4dcab32f.../index_metadata.pickle | 237KB | http://211.231.121.68:8888/lanco/rag_llm_chaining/data/embeddings/chroma/4dcab32f-2e46-4cc0-8cc9-d09c76ac0d04/index_metadata.pickle |
| 4dcab32f.../length.bin | 20KB | http://211.231.121.68:8888/lanco/rag_llm_chaining/data/embeddings/chroma/4dcab32f-2e46-4cc0-8cc9-d09c76ac0d04/length.bin |
| 4dcab32f.../link_lists.bin | 45KB | http://211.231.121.68:8888/lanco/rag_llm_chaining/data/embeddings/chroma/4dcab32f-2e46-4cc0-8cc9-d09c76ac0d04/link_lists.bin |
| 4dcab32f.../header.bin | 100B | http://211.231.121.68:8888/lanco/rag_llm_chaining/data/embeddings/chroma/4dcab32f-2e46-4cc0-8cc9-d09c76ac0d04/header.bin |
| 578b45a4.../data_level0.bin | 13MB | http://211.231.121.68:8888/lanco/rag_llm_chaining/data/embeddings/chroma/578b45a4-2463-4865-8743-3f62013613b8/data_level0.bin |
| 578b45a4.../index_metadata.pickle | 139KB | http://211.231.121.68:8888/lanco/rag_llm_chaining/data/embeddings/chroma/578b45a4-2463-4865-8743-3f62013613b8/index_metadata.pickle |
| 578b45a4.../length.bin | 12KB | http://211.231.121.68:8888/lanco/rag_llm_chaining/data/embeddings/chroma/578b45a4-2463-4865-8743-3f62013613b8/length.bin |
| 578b45a4.../link_lists.bin | 27KB | http://211.231.121.68:8888/lanco/rag_llm_chaining/data/embeddings/chroma/578b45a4-2463-4865-8743-3f62013613b8/link_lists.bin |
| 578b45a4.../header.bin | 100B | http://211.231.121.68:8888/lanco/rag_llm_chaining/data/embeddings/chroma/578b45a4-2463-4865-8743-3f62013613b8/header.bin |

### 4. 데이터 체크포인트 (981MB)

| 파일명 | 크기 | 다운로드 URI |
|--------|------|--------------|
| chunks.pkl | 981MB | http://211.231.121.68:8888/lanco/rag_llm_chaining/data/checkpoints/chunks.pkl |

---

## 다운로드 스크립트

### 전체 다운로드 (Bash)

```bash
#!/bin/bash
BASE_URL="http://211.231.121.68:8888/lanco/rag_llm_chaining"

# SOLAR-10.7B 모델
mkdir -p models/solar-10.7b
cd models/solar-10.7b
for i in {1..5}; do
  curl -O "${BASE_URL}/models/solar-10.7b/model-0000${i}-of-00005.safetensors"
done
curl -O "${BASE_URL}/models/solar-10.7b/config.json"
curl -O "${BASE_URL}/models/solar-10.7b/generation_config.json"
curl -O "${BASE_URL}/models/solar-10.7b/model.safetensors.index.json"
curl -O "${BASE_URL}/models/solar-10.7b/tokenizer.json"
curl -O "${BASE_URL}/models/solar-10.7b/tokenizer.model"
curl -O "${BASE_URL}/models/solar-10.7b/tokenizer_config.json"
cd ../..

# ChromaDB 임베딩
mkdir -p data/embeddings/chroma/70b6e63d-2c1f-407d-80ef-85ae8a6d9ee7
mkdir -p data/embeddings/chroma/4dcab32f-2e46-4cc0-8cc9-d09c76ac0d04
mkdir -p data/embeddings/chroma/578b45a4-2463-4865-8743-3f62013613b8
curl -o data/embeddings/chroma/chroma.sqlite3 "${BASE_URL}/data/embeddings/chroma/chroma.sqlite3"

for uuid in 70b6e63d-2c1f-407d-80ef-85ae8a6d9ee7 4dcab32f-2e46-4cc0-8cc9-d09c76ac0d04 578b45a4-2463-4865-8743-3f62013613b8; do
  for file in data_level0.bin index_metadata.pickle length.bin link_lists.bin header.bin; do
    curl -o "data/embeddings/chroma/${uuid}/${file}" "${BASE_URL}/data/embeddings/chroma/${uuid}/${file}"
  done
done

# 체크포인트
mkdir -p data/checkpoints
curl -o data/checkpoints/chunks.pkl "${BASE_URL}/data/checkpoints/chunks.pkl"
```

### Python 다운로드 스크립트

```python
import os
import requests
from pathlib import Path
from tqdm import tqdm

BASE_URL = "http://211.231.121.68:8888/lanco/rag_llm_chaining"

def download_file(url: str, dest: str):
    """파일 다운로드 with 진행률 표시"""
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))

    with open(dest, 'wb') as f, tqdm(
        desc=Path(dest).name,
        total=total,
        unit='iB',
        unit_scale=True
    ) as pbar:
        for data in response.iter_content(chunk_size=1024*1024):
            size = f.write(data)
            pbar.update(size)

# 사용 예시
download_file(
    f"{BASE_URL}/models/solar-10.7b/model-00001-of-00005.safetensors",
    "models/solar-10.7b/model-00001-of-00005.safetensors"
)
```

---

## 용량 요약

| 카테고리 | 용량 |
|----------|------|
| SOLAR-10.7B | 20GB |
| Qwen2.5-14B (부분) | 7.4GB |
| ChromaDB 임베딩 | 8.9GB |
| 체크포인트 | 981MB |
| **총합** | **~38GB** |

---

## 주의사항

1. **네트워크 환경**: 대용량 파일이므로 안정적인 네트워크 환경에서 다운로드하세요
2. **디스크 공간**: 최소 50GB 이상의 여유 공간을 확보하세요
3. **Qwen2.5-14B**: 부분 다운로드 상태입니다. 완전한 모델이 필요하면 HuggingFace에서 직접 다운로드하세요
4. **SeaweedFS 접근**: 내부 네트워크에서만 접근 가능할 수 있습니다

---

*마지막 업데이트: 2025-12-09*
