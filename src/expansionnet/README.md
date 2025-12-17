# ExpansionNetV2 for Korean Marine Vision Image Captioning

한국 해양 영상 데이터를 위한 멀티모달 이미지 캡셔닝 모델

## 📋 개요

ExpansionNetV2 기반의 멀티모달 아키텍처로, 해양 환경 영상에서 객체와 환경 정보를 통합하여 한국어 캡션을 생성합니다.

## 🏗️ 모델 구조

### 전체 아키텍처

```
입력 이미지 (224×224)
    ├─→ ViT Encoder        → 전역 장면 특징 (글로벌 컨텍스트)
    ├─→ DINO Encoder       → 객체별 시각 특징 (bbox crop)
    ├─→ Metadata Embedding → 객체 클래스, 환경 정보
    └─→ Transformer Decoder → 한국어 캡션 생성
```

### 1. **입력 데이터**

#### 이미지
- 크기: 224×224 RGB
- 정규화: ImageNet 표준

#### 메타데이터 (JSON)
```json
{
  "environment": {
    "season": 1-2,    // 계절 (1:여름, 2:겨울)
    "night": 1-2,     // 주야 (1:낮, 2:밤)
    "weather": 1-7,   // 날씨 (1-7단계)
    "wave": 1-7       // 파도 (1-7단계)
  },
  "annotations": [
    {
      "class": 1-4,        // 객체 클래스
      "sub_class": 1-42,   // 서브클래스
      "bounding_box": [w, h, x, y]  // bbox (픽셀 → 정규화)
    }
  ],
  "caption": "여름 맑은 주간 하늘에..."
}
```

### 2. **인코더 구조**

#### ViT (Vision Transformer)
- **모델**: `google/vit-base-patch16-224`
- **역할**: 전체 장면의 글로벌 특징 추출
- **출력**: (B, 1, 768) - CLS token 사용
- **특징**: 장면 전체의 맥락 이해 (하늘, 바다, 배경)

#### DINO v2 (Object Encoder)
- **모델**: `dinov2_vits14` (Facebook Research)
- **역할**: 각 객체의 시각적 특징 추출
- **입력**: bbox로 crop된 객체 영역 (224×224)
- **출력**: (B, N_obj, 384) → project → (B, N_obj, 768)
- **특징**: 
  - Frozen weights (학습 안 함)
  - bbox 정규화 좌표로 crop
  - 객체별 독립적인 특징 추출

#### Metadata Embeddings
- **객체 클래스**: Embedding(4, 768) - 드론, 항공기, 선박 등
- **서브클래스**: Embedding(42, 768) - 세부 분류
- **Bounding Box**: Linear(4, 768) - 정규화된 [w, h, x, y]
- **환경 정보**: Linear(4, 768) - [season, night, weather, wave]

#### Expansion Layer
```python
Input (768) → Linear(768 → 3072) → ReLU → Linear(3072 → 768)
```
- 시각적 특징의 표현력 증가

### 3. **디코더 구조**

#### Transformer Decoder
- **레이어 수**: 6층
- **어텐션 헤드**: 8개
- **임베딩 차원**: 768
- **구성**:
  ```
  Self-Attention (Causal Mask)
      ↓
  Cross-Attention (Encoder Memory)
      ↓
  Feed-Forward Network
  ```

#### 메모리 구성
```python
encoder_memory = [
    ViT 특징 (B, 1, 768),
    DINO 객체 특징 (B, N_obj, 768),
    환경 임베딩 (B, 1, 768)
]
# 총 (B, N_obj+2, 768)
```

#### Causal Mask
```python
# Upper triangular mask (future positions masked)
[[0,   -inf, -inf, ...],
 [0,    0,   -inf, ...],
 [0,    0,    0,   ...]]
```

### 4. **출력**

- **Vocabulary**: klue/roberta-base (한국어 토크나이저)
- **Vocab Size**: ~32,000
- **Max Length**: 64 tokens
- **Output**: (B, seq_len, vocab_size) → Argmax → 한국어 캡션

## 📊 데이터 처리

### Bbox 정규화
```python
# 원본: 픽셀 좌표
bbox_pixel = [w=44, h=22, x=1458, y=521]

# 정규화: [0, 1] 범위
w_norm = w / image_width
h_norm = h / image_height
x_norm = x / image_width
y_norm = y / image_height

# DINO crop 시 다시 픽셀로 변환
x1 = (x_norm - w_norm/2) * 224
y1 = (y_norm - h_norm/2) * 224
x2 = (x_norm + w_norm/2) * 224
y2 = (y_norm + h_norm/2) * 224
```

### Label 인덱싱
```python
# 모든 클래스: 1-based → 0-based
class_id = json["class"] - 1        # 1~4 → 0~3
subclass_id = json["sub_class"] - 1 # 1~42 → 0~41
season = json["season"] - 1         # 1~2 → 0~1
weather = json["weather"] - 1       # 1~7 → 0~6
```

## 🎯 학습

### 학습 설정
```python
# Optimizer
AdamW(lr=1e-4)

# Loss
CrossEntropyLoss(ignore_index=pad_token_id)

# 예측: logits[:, :-1] (마지막 제외)
# 타겟: ids[:, 1:]     (첫 CLS 제외)
```

### Early Stopping
- **Max Epochs**: 5
- **Patience**: 2 (2 epoch 개선 없으면 중단)
- **LR Scheduler**: ReduceLROnPlateau (factor=0.5, patience=1)
- **Best Model**: 가장 낮은 loss 모델 저장

### 데이터셋
- **Train**: 11,871 images
- **Test**: 3,055 images
- **Batch Size**: 4
- **Max Objects**: 가변 (배치별 최대값으로 padding)

## 🚀 사용법

### 학습
```bash
# 전체 학습 (early stopping)
python train_earlystop.py

# Quick 학습 (100 샘플, 3 epochs)
python quick_train.py 100 3
```

### 평가
```bash
# 전체 테스트셋
python test.py

# 일부 샘플만 (빠른 검증)
python test.py 10
```

### 추론 (Autoregressive Generation)
```python
# 시작: [CLS]
ids = [[CLS_token_id]]

# 반복 생성
for step in range(max_len):
    logits = model(image, objects, env, ids, tgt_mask=None)
    next_token = logits[0, -1].argmax()
    ids = torch.cat([ids, [[next_token]]], dim=1)
    
    if next_token == SEP_token_id:
        break

caption = tokenizer.decode(ids, skip_special_tokens=True)
```

## 📁 파일 구조

```
src/expansionnet/
├── model.py              # 모델 정의
│   ├── ObjectFeatureEncoder  # DINO + metadata
│   ├── EnvEncoder            # 환경 임베딩
│   ├── ExpansionLayer        # Feature expansion
│   └── ExpansionNetV2_Multimodal  # 메인 모델
├── build_data.py         # 데이터셋 로더
├── train_earlystop.py    # 학습 (early stopping)
├── quick_train.py        # 빠른 학습 (소규모)
├── test.py               # 평가 스크립트
└── README.md             # 문서

outputs/expansionnet/
├── expnetv2_earlystop.pth  # Best 모델 (early stopping)
├── expnetv2_quick.pth      # Quick 모델
└── expnetv2_multimodal.pth # Full 모델 (10 epochs)
```

## 🔍 주요 특징

### ✅ 개선 사항
1. **Bbox 정규화**: 픽셀 좌표 → [0,1] 정규화로 안정적인 학습
2. **DINO 통합**: bbox crop으로 실제 객체 시각 특징 사용
3. **Causal Mask**: 올바른 autoregressive generation
4. **Early Stopping**: 과적합 방지
5. **Multimodal Fusion**: ViT + DINO + Metadata 통합

### ⚠️ 주의사항
1. **DINO는 Frozen**: 사전학습 가중치만 사용 (fine-tuning 안 함)
2. **Bbox 필수**: 객체 좌표가 없으면 DINO 사용 불가
3. **GPU 권장**: 212M 파라미터 → CPU로는 매우 느림
4. **Padding**: 객체 수가 배치마다 다름 → zero padding 필요

## 📈 성능

### Quick Train (100 샘플, 3 epochs)
- Epoch 0: Loss 4.76
- Epoch 1: Loss 1.42
- Epoch 2: Loss 0.95
- **결과**: GT와 유사한 캡션 생성

### Full Train (과적합 문제)
- 10 epochs → 객체 누락, 환경 정보 오류
- **해결**: Early stopping으로 3-5 epoch에서 중단

## 🎓 참고

### 사용 모델
- **ViT**: [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)
- **DINO v2**: [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)
- **Tokenizer**: [klue/roberta-base](https://huggingface.co/klue/roberta-base)

### 이론적 배경
- ExpansionNet v2: 시각적 특징 확장을 통한 캡셔닝
- Self-Attention: 시퀀스 내 의존성 학습
- Cross-Attention: 이미지-텍스트 정렬
- Causal Masking: 미래 토큰 참조 방지

## 🔧 최적화 팁

1. **배치 크기**: GPU 메모리에 따라 2-8 조정
2. **Learning Rate**: 1e-4가 안정적, 필요시 1e-5로 감소
3. **Max Length**: 캡션 길이에 따라 64-128 조정
4. **Early Stopping Patience**: 2-3이 적절
5. **Epoch 수**: 5-7이면 충분, 10 이상은 과적합 위험
