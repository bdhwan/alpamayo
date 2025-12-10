# Alpamayo R1 추론 클라이언트

Alpamayo R1 추론 API 서버와 상호작용하기 위한 Python 클라이언트입니다. 이 클라이언트는 이미지에 대한 추론을 실행하고 체인 오브 사고(chain-of-thought) 추론과 함께 궤적 예측을 가져오는 쉬운 인터페이스를 제공합니다.

## 설치

필요한 의존성이 설치되어 있는지 확인하세요:

```bash
pip install requests
```

또는 프로젝트 루트에서 모든 요구사항을 설치하세요:

```bash
pip install -r requirements.txt
```

## 빠른 시작

### Python API 사용

```python
from client.inference_client import AlpamayoClient

# 클라이언트 초기화 (기본값: http://localhost:8001)
client = AlpamayoClient(base_url="http://localhost:8001")

# 이미지 파일 경로 리스트로 추론 실행
result = client.inference(
    image_paths=["image1.jpg", "image2.jpg", "image3.jpg", "image4.jpg"],
    clip_id="030c760c-ae38-49aa-9ad8-f5650a545d26",
    t0_us=5100000,
)

# 결과 접근
print("Chain of Thought:", result["choices"][0]["chain_of_thought"])
print("Trajectory:", result["choices"][0]["trajectory"])
```

### 명령줄 사용

```bash
# 가장 간단한 사용법: 이미지 폴더와 서버 주소만 지정
python client/inference_client.py \
  --folder /path/to/images \
  --api-url "http://localhost:8001"

# 원격 서버 사용 예제
python client/inference_client.py \
  --folder /data/camera/images \
  --api-url "http://192.168.1.100:8001"

# clip_id를 사용한 기본 사용법 (권장)
python client/inference_client.py image1.jpg image2.jpg image3.jpg image4.jpg \
  --clip-id "030c760c-ae38-49aa-9ad8-f5650a545d26" \
  --t0-us 5100000

# 이미지 폴더와 호스트 URL 사용
python client/inference_client.py \
  --folder /path/to/images \
  --api-url "http://localhost:8001" \
  --clip-id "030c760c-ae38-49aa-9ad8-f5650a545d26" \
  --t0-us 5100000

# 원격 호스트와 이미지 개수 지정
python client/inference_client.py \
  --folder /path/to/images \
  --num-images 4 \
  --api-url "http://192.168.1.100:8001" \
  --clip-id "030c760c-ae38-49aa-9ad8-f5650a545d26" \
  --t0-us 5100000 \
  --output result.json \
  --pretty

# JSON 파일로 저장하며 예쁘게 출력
python client/inference_client.py image1.jpg image2.jpg image3.jpg image4.jpg \
  --clip-id "030c760c-ae38-49aa-9ad8-f5650a545d26" \
  --t0-us 5100000 \
  --output result.json \
  --pretty

# 커스텀 API URL 사용
python client/inference_client.py image1.jpg image2.jpg image3.jpg image4.jpg \
  --api-url "http://localhost:8001" \
  --clip-id "030c760c-ae38-49aa-9ad8-f5650a545d26" \
  --t0-us 5100000
```

**참고:** `clip_id`나 `trajectory_history`를 제공하지 않으면 zero history(모두 0인 궤적)가 사용됩니다. 이는 테스트 목적으로는 가능하지만, 실제 사용 시에는 정확도가 낮을 수 있습니다.

## API 참조

### AlpamayoClient 클래스

#### `__init__(base_url="http://localhost:8001", timeout=300)`

클라이언트를 초기화합니다.

**매개변수:**
- `base_url` (str): API 서버의 기본 URL (기본값: "http://localhost:8001")
- `timeout` (int): 요청 타임아웃(초) (기본값: 300)

**예제:**
```python
client = AlpamayoClient(base_url="http://localhost:8001", timeout=300)
```

#### `inference(image_paths, clip_id=None, t0_us=None, trajectory_history=None, ...)`

이미지에 대한 추론을 실행합니다.

**매개변수:**
- `image_paths` (list[str | Path]): **필수.** 이미지 파일 경로 리스트
- `clip_id` (str, 선택): 데이터셋에서 궤적 이력을 로드할 클립 ID
- `t0_us` (int, 선택): 마이크로초 단위 타임스탬프 (clip_id와 함께 사용)
- `trajectory_history` (dict, 선택): 'ego_history_xyz'와 'ego_history_rot' 키를 가진 궤적 이력 데이터 딕셔너리
- `temperature` (float): 샘플링 온도 (기본값: 0.6)
- `top_p` (float): Top-p 샘플링 매개변수 (기본값: 0.98)
- `num_traj_samples` (int): 궤적 샘플 수 (기본값: 1)
- `max_generation_length` (int): 최대 생성 길이 (기본값: 256)
- `model` (str): 모델 이름 (기본값: "alpamayo-r1")

**반환값:** `dict[str, Any]` - API 응답 딕셔너리

**예제:**
```python
result = client.inference(
    image_paths=["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"],
    clip_id="030c760c-ae38-49aa-9ad8-f5650a545d26",
    t0_us=5100000,
    temperature=0.6,
    top_p=0.98,
    num_traj_samples=1,
)
```

#### `health_check()`

API 서버가 정상인지 확인합니다.

**반환값:** `dict[str, Any]` - 헬스 체크 응답

**예제:**
```python
health = client.health_check()
print(health)  # {"status": "healthy"}
```

#### `get_recent_images(folder_path, n=4, sort_by="mtime")`

폴더에서 가장 최근에 생성/수정된 n개의 이미지 파일을 가져옵니다.

**매개변수:**
- `folder_path` (str | Path): 이미지가 포함된 폴더 경로
- `n` (int): 반환할 최근 이미지 수 (기본값: 4)
- `sort_by` (str): 'mtime' (수정 시간) 또는 'ctime' (생성 시간)으로 정렬 (기본값: 'mtime')

**반환값:** `list[Path]` - 가장 최근 순으로 정렬된 n개의 최근 이미지 파일에 대한 Path 객체 리스트

**예외:**
- `FileNotFoundError`: 폴더가 존재하지 않는 경우
- `ValueError`: n이 1보다 작거나 sort_by가 유효하지 않은 경우

**예제:**
```python
# 4개의 가장 최근 이미지 가져오기 (기본값)
recent_images = client.get_recent_images("/path/to/images")

# 생성 시간으로 정렬하여 6개의 가장 최근 이미지 가져오기
recent_images = client.get_recent_images("/path/to/images", n=6, sort_by="ctime")
```

#### `encode_image(image_path)`

이미지 파일을 base64 데이터 URL로 인코딩합니다.

**매개변수:**
- `image_path` (str | Path): 이미지 파일 경로

**반환값:** `str` - Base64로 인코딩된 데이터 URL 문자열

**지원 형식:** JPG, JPEG, PNG, GIF, BMP, WEBP

## 응답 형식

API는 다음 구조의 딕셔너리를 반환합니다:

```python
{
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "created": 1234567890,
    "model": "alpamayo-r1",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "결합된 텍스트 출력..."
            },
            "finish_reason": "stop",
            "chain_of_thought": "상세한 추론 텍스트...",
            "meta_action": "동작 설명...",
            "answer": "답변 텍스트...",
            "trajectory": {
                "xyz": [[[[0.0, 0.0, 0.0], ...]]],
                "rotation": [[[[[[1.0, 0.0, 0.0], ...]]]]]
            }
        }
    ],
    "usage": {
        "prompt_tokens": 1000,
        "completion_tokens": 500,
        "total_tokens": 1500
    }
}
```

## 명령줄 인수

| 인수 | 설명 | 기본값 |
|------|------|--------|
| `images` | 이미지 파일 경로 (위치 인수, 여러 개 허용) | 필수* |
| `--folder` | 이미지가 포함된 폴더 경로 (이미지 경로 제공 대안) | None |
| `--num-images` | --folder가 지정된 경우 사용할 최근 이미지 수 | `4` |
| `--sort-by` | 이미지를 'mtime' (수정) 또는 'ctime' (생성)으로 정렬 | `mtime` |
| `--api-url` | API 서버의 기본 URL | `http://localhost:8001` |
| `--clip-id` | 데이터셋에서 궤적 이력을 로드할 클립 ID | None |
| `--t0-us` | 마이크로초 단위 타임스탬프 (--clip-id와 함께 사용) | None |
| `--trajectory-history` | 궤적 이력을 포함한 JSON 파일 경로 | None |
| `--temperature` | 샘플링 온도 | `0.6` |
| `--top-p` | Top-p 샘플링 매개변수 | `0.98` |
| `--num-traj-samples` | 궤적 샘플 수 | `1` |
| `--max-gen-length` | 최대 생성 길이 | `256` |
| `--output`, `-o` | 응답 JSON을 저장할 출력 파일 | None |
| `--pretty` | JSON 응답을 예쁘게 출력 | False |

\* 이미지 파일 경로를 제공하거나 `--folder` 옵션을 사용하세요

## 예제

### 예제 1: 가장 간단한 사용법 (이미지 폴더와 서버 주소만)

```bash
# 로컬 서버 사용
python client/inference_client.py \
  --folder /path/to/images \
  --api-url "http://localhost:8001"

# 원격 서버 사용
python client/inference_client.py \
  --folder /data/camera/images \
  --api-url "http://192.168.1.100:8001"

# 결과를 파일로 저장
python client/inference_client.py \
  --folder /path/to/images \
  --api-url "http://localhost:8001" \
  --output result.json
```

**참고:** `clip_id`나 `trajectory_history`를 제공하지 않으면 zero history(모두 0인 궤적)가 사용됩니다. 이는 테스트 목적으로는 가능하지만, 실제 사용 시에는 정확도가 낮을 수 있습니다.

### 예제 2: 기본 Python 사용

```python
from client.inference_client import AlpamayoClient

client = AlpamayoClient()

# 서버 상태 확인
health = client.health_check()
print(f"서버 상태: {health}")

# 추론 실행
result = client.inference(
    image_paths=["camera1.jpg", "camera2.jpg", "camera3.jpg", "camera4.jpg"],
    clip_id="030c760c-ae38-49aa-9ad8-f5650a545d26",
    t0_us=5100000,
)

# 결과 추출
choice = result["choices"][0]
print("Chain of Thought:", choice["chain_of_thought"])
print("Trajectory:", choice["trajectory"])
```

### 예제 3: 궤적 이력 직접 사용

```python
from client.inference_client import AlpamayoClient

client = AlpamayoClient()

trajectory_history = {
    "ego_history_xyz": [[[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]]],
    "ego_history_rot": [[[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]]]
}

result = client.inference(
    image_paths=["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"],
    trajectory_history=trajectory_history,
)
```

### 예제 4: 폴더와 호스트 URL 사용

```bash
# 커스텀 호스트 URL과 함께 폴더 경로 사용
python client/inference_client.py \
  --folder /path/to/image/folder \
  --api-url "http://192.168.1.100:8001" \
  --num-images 4 \
  --clip-id "030c760c-ae38-49aa-9ad8-f5650a545d26" \
  --t0-us 5100000 \
  --output results.json \
  --pretty

# 원격 서버와 폴더 사용
python client/inference_client.py \
  --folder /data/camera/images \
  --api-url "http://api.example.com:8001" \
  --num-images 6 \
  --sort-by ctime \
  --clip-id "030c760c-ae38-49aa-9ad8-f5650a545d26" \
  --t0-us 5100000
```

### 예제 5: 여러 옵션을 사용한 명령줄

```bash
python client/inference_client.py \
  image1.jpg image2.jpg image3.jpg image4.jpg \
  --clip-id "030c760c-ae38-49aa-9ad8-f5650a545d26" \
  --t0-us 5100000 \
  --temperature 0.7 \
  --top-p 0.95 \
  --num-traj-samples 3 \
  --max-gen-length 512 \
  --output results.json \
  --pretty
```

### 예제 6: 배치 처리

```python
from client.inference_client import AlpamayoClient
from pathlib import Path

client = AlpamayoClient()

# 여러 이미지 세트 처리
image_sets = [
    ["set1_img1.jpg", "set1_img2.jpg", "set1_img3.jpg", "set1_img4.jpg"],
    ["set2_img1.jpg", "set2_img2.jpg", "set2_img3.jpg", "set2_img4.jpg"],
    ["set3_img1.jpg", "set3_img2.jpg", "set3_img3.jpg", "set3_img4.jpg"],
]

clip_id = "030c760c-ae38-49aa-9ad8-f5650a545d26"
t0_us = 5100000

results = []
for image_set in image_sets:
    result = client.inference(
        image_paths=image_set,
        clip_id=clip_id,
        t0_us=t0_us,
    )
    results.append(result)
    print(f"{len(results)}개의 이미지 세트 처리 완료")
```

## 오류 처리

클라이언트는 일반적인 오류에 대해 예외를 발생시킵니다:

- `FileNotFoundError`: 이미지 파일을 찾을 수 없음
- `ValueError`: 잘못된 입력 (예: 빈 이미지 리스트)
- `RuntimeError`: API 요청 실패 또는 서버 오류

**오류 처리 예제:**

```python
from client.inference_client import AlpamayoClient

client = AlpamayoClient()

try:
    result = client.inference(
        image_paths=["image1.jpg", "image2.jpg"],
        clip_id="invalid-clip-id",
    )
except FileNotFoundError as e:
    print(f"이미지 파일을 찾을 수 없습니다: {e}")
except RuntimeError as e:
    print(f"API 오류: {e}")
except Exception as e:
    print(f"예상치 못한 오류: {e}")
```

## 참고사항

- **이미지 형식**: JPG, JPEG, PNG, GIF, BMP, WEBP 형식 지원
- **여러 이미지**: 클라이언트는 여러 이미지 경로를 허용하고 모두 인코딩합니다
- **궤적 이력**: `clip_id` (선택적으로 `t0_us`)를 제공하거나 `trajectory_history`를 직접 제공하세요
- **기본 포트**: API 서버는 기본적으로 포트 8001을 사용합니다
- **타임아웃**: 추론 요청에 대한 기본 요청 타임아웃은 300초(5분)입니다
- **이미지만 전달**: `clip_id`나 `trajectory_history` 없이 이미지만 전달할 수 있지만, 이 경우 zero history가 사용되어 예측 정확도가 낮을 수 있습니다

## 문제 해결

### 연결 오류

연결 오류가 발생하면 다음을 확인하세요:
1. API 서버가 실행 중인지: `curl http://localhost:8001/health`
2. 올바른 포트가 지정되었는지 (기본값: 8001)
3. 방화벽/네트워크 설정이 연결을 허용하는지

### 이미지 인코딩 오류

- 이미지 파일이 존재하고 읽을 수 있는지 확인
- 파일 권한 확인
- 이미지 형식이 지원되는지 확인

### API 오류

- 서버 로그에서 상세한 오류 메시지 확인
- 데이터셋 로딩을 사용하는 경우 `clip_id`가 유효한지 확인
- 직접 제공하는 경우 궤적 이력 형식이 올바른지 확인

## 라이선스

SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
