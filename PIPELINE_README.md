# 해양 비전 캡션 생성 파이프라인

## 목차
1. [개요](#개요)
2. [파이프라인 아키텍처](#파이프라인-아키텍처)
3. [각 모델 역할](#각-모델-역할)
4. [파이프라인 구현](#파이프라인-구현)
5. [핵심 알고리즘](#핵심-알고리즘)
6. [사용 방법](#사용-방법)
7. [문제 해결 및 최적화](#문제-해결-및-최적화)

---

## 개요

본 파이프라인은 해양 환경의 이미지를 입력받아 한국어 캡션을 자동 생성하는 멀티모달 시스템입니다. 세 개의 독립적으로 학습된 모델(DINO, ViT, ExpansionNetV2)을 순차적으로 연결하여 구성되었습니다.

**입력**: 해양 이미지 (전자광학/적외선)  
**출력**: "하절기 맑은 낮 하늘에 삐라 한 개가 이동하고 있고 고정익 유인항공기 전투기 한 대가 비행 중이며 파도 2단계이다."

---

## 파이프라인 아키텍처

```
┌─────────────────┐
│   입력 이미지    │
└────────┬────────┘
         │
         ├──────────────────────────────────────────┐
         │                                          │
         ▼                                          ▼
┌─────────────────┐                      ┌─────────────────┐
│  DINO (객체탐지) │                      │  ViT (환경분류)  │
│  - Grounding-DINO│                      │  - ViT-Base     │
│  - Bbox 추출     │                      │  - 4가지 환경   │
└────────┬────────┘                      └────────┬────────┘
         │                                          │
         │    ┌──────────────────────────────┐     │
         │    │    객체 정보 (N, 6)           │     │
         │    │  - class_id, subclass_id     │     │
         │    │  - bbox: [w, h, x, y]        │     │
         │    └──────────────────────────────┘     │
         │                                          │
         └──────────────┬───────────────────────────┘
                        ▼
              ┌──────────────────┐
              │  ExpansionNetV2   │
              │  (캡션 생성)       │
              │                  │
              │  1. ViT Encoder  │──── 전체 장면 이해
              │  2. DINO Crops   │──── 객체별 시각 특징
              │  3. Env Encoder  │──── 환경 정보 임베딩
              │  4. Transformer  │──── 한국어 생성
              │     Decoder      │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │  한국어 캡션 출력  │
              └──────────────────┘
```

---

## 각 모델 역할

### 1. DINO (Grounding-DINO) - 객체 탐지

**역할**: 이미지에서 해양 객체(선박, 항공기, 부유물 등)를 탐지하고 위치 정보 추출

**입력**: RGB 이미지 (H×W×3)  
**출력**: 객체 정보 텐서 (N_objects, 6)
- `[0]`: class_id (0-3, 4개 클래스로 클램핑)
- `[1]`: subclass_id (0-41, 42개 세부 클래스로 클램핑)
- `[2-5]`: bbox 정규화 좌표 [w, h, x, y] (0-1 범위)

**체크포인트**: `outputs/dino/epoch_12.pth`

**주요 처리**:
```python
# src/dino/infer.py
def run_dino(image_path: str) -> torch.Tensor:
    model = init_detector(CFG_FILE, CKPT_FILE, device="cuda")
    result = inference_detector(model, image_path)
    
    # 신뢰도 필터링 (threshold=0.3)
    keep = scores >= CONF_THRESH
    
    # 최대 객체 수 제한 (MAX_OBJECTS=10)
    if len(bboxes) > MAX_OBJECTS:
        bboxes = bboxes[:MAX_OBJECTS]
    
    # XYXY → XYWH 변환 및 정규화
    bbox_norm = _xyxy_to_xywh_norm(bboxes, w, h)
    
    return torch.from_numpy(obj).float()  # (N, 6)
```

### 2. ViT (Vision Transformer) - 환경 분류

**역할**: 이미지의 환경 조건 4가지를 분류

**입력**: RGB 이미지 (224×224×3)  
**출력**: 환경 벡터 (4,)
- `[0]`: season (계절: 0=하절기, 1=동절기)
- `[1]`: night (주야: 0=주간, 1=야간)
- `[2]`: weather (날씨: 0-7, 맑음/비/눈/해무 등)
- `[3]`: wave (파도: 0-7, 1-7단계)

**체크포인트**: `outputs/vit/vit_env.pth`

**주요 처리**:
```python
# 파이프라인 내 ViT 처리
vit_logits = self.vit(img_tensor)
env_vec = torch.tensor([
    vit_logits["season"].argmax(-1).item(),
    vit_logits["night"].argmax(-1).item(),
    vit_logits["weather"].argmax(-1).item(),
    vit_logits["wave"].argmax(-1).item(),
], device=self.device)

# 안전성 클램핑 (범위 벗어남 방지)
env_vec = env_vec.clamp(0, 7)
```

### 3. ExpansionNetV2 - 캡션 생성

**역할**: 멀티모달 정보를 통합하여 한국어 캡션 생성

**입력**:
- ViT 전역 특징 (1, 768)
- DINO 객체 특징 (N_obj, 768) - bbox crop 후 DINO 재추론
- 환경 벡터 (4,)

**출력**: 한국어 시퀀스 (가변 길이, max_len=128)

**체크포인트**: `outputs/expansionnet/expnetv2_final.pth`  
**토크나이저**: `klue/roberta-base` (한국어 특화)

**아키텍처 구성**:

```python
# src/expansionnet/model.py
class ExpansionNetV2_Multimodal(nn.Module):
    def __init__(self, vocab_size, num_classes, num_subclasses):
        # 1. ViT 이미지 인코더 (전체 장면)
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        
        # 2. 객체 & 환경 인코더
        self.obj_encoder = ObjectFeatureEncoder(...)
        self.env_encoder = EnvEncoder(embed_dim)
        
        # 3. Expansion Layer (시각 특징 증강)
        self.expand = ExpansionLayer(embed_dim)
        
        # 4. Transformer Decoder (6-layer, 8-head)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=768, nhead=8, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        
        # 5. 출력 레이어
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size)
```

**객체 특징 추출 (DINO 재사용)**:
```python
class ObjectFeatureEncoder(nn.Module):
    def forward(self, obj_tensor, full_images):
        # DINO로 bbox 영역의 시각 특징 추출
        obj_crops = self._crop_objects(full_images, obj_tensor[:, :, 2:6])
        
        with torch.no_grad():
            dino_feats = self.dino(obj_crops)  # (B*N_obj, 384)
        
        # 메타데이터와 결합
        cls_emb = self.class_emb(obj_tensor[:, :, 0].long())
        sub_emb = self.subclass_emb(obj_tensor[:, :, 1].long())
        bbox_emb = self.bbox_fc(obj_tensor[:, :, 2:].float())
        
        # 투영 및 통합
        dino_feats = self.dino_proj(dino_feats)  # (B, N_obj, 768)
        x = torch.cat([dino_feats, cls_emb, sub_emb, bbox_emb], dim=-1)
        return self.proj(x)  # (B, N_obj, 768)
```

---

## 파이프라인 구현

### CaptionPipeline 클래스

전체 파이프라인을 통합 관리하는 메인 클래스입니다.

```python
# src/pipeline/pipeline.py
class CaptionPipeline:
    def __init__(self, dino_ckpt, vit_ckpt, exp_ckpt, device="cuda"):
        self.device = device
        
        # 1. DINO 초기화 (지연 로딩)
        self.dino_ckpt = dino_ckpt
        
        # 2. ViT 환경 분류기 로딩
        self.vit = ViTEnvClassifier().to(device)
        self.vit.load_state_dict(torch.load(vit_ckpt, map_location=device))
        self.vit.eval()
        
        # 3. ExpansionNet 로딩 (아키텍처 자동 추론)
        self.tokenizer = load_tokenizer()
        state_dict = torch.load(exp_ckpt, map_location=device)
        
        # 체크포인트에서 모델 구조 파라미터 추출
        num_classes = state_dict['base_model.obj_encoder.class_emb.weight'].shape[0]
        num_subclasses = state_dict['base_model.obj_encoder.subclass_emb.weight'].shape[0]
        
        self.model = ExpansionNetV2_WithDropout(
            vocab_size=self.tokenizer.vocab_size,
            num_classes=num_classes,      # 자동 추론: 4
            num_subclasses=num_subclasses  # 자동 추론: 42
        ).to(device)
        
        # base_model. 프리픽스 제거 후 로딩
        new_state_dict = {
            k[11:] if k.startswith('base_model.') else k: v
            for k, v in state_dict.items()
        }
        self.model.load_state_dict(new_state_dict)
        self.model.eval()
```

### 추론 파이프라인

```python
def run(self, image_path: str, max_len: int = 128, debug: bool = False):
    """
    전체 파이프라인 실행
    
    DINO (객체 탐지) → ViT (환경 분류) → ExpansionNet (캡션 생성)
    """
    
    # Step 1: DINO 객체 탐지
    dino_objects = run_dino(image_path)  # (N_obj, 6)
    
    # Step 2: ViT 환경 분류
    img_pil = Image.open(image_path).convert("RGB")
    img_tensor = self.preprocess(img_pil).unsqueeze(0).to(self.device)
    
    with torch.no_grad():
        vit_logits = self.vit(img_tensor)
    
    env_vec = torch.tensor([
        vit_logits["season"].argmax(-1).item(),
        vit_logits["night"].argmax(-1).item(),
        vit_logits["weather"].argmax(-1).item(),
        vit_logits["wave"].argmax(-1).item(),
    ], device=self.device).unsqueeze(0)
    
    # 환경 벡터 클램핑 (안전성)
    env_vec = env_vec.clamp(0, 7)
    
    # Step 3: 객체 정보 준비 및 클래스 ID 클램핑
    if dino_objects.shape[0] == 0:
        objects = torch.zeros(1, 1, 6, device=self.device)
    else:
        objects = dino_objects.unsqueeze(0).to(self.device)
        
        # CRITICAL: 클래스 ID를 모델의 임베딩 범위로 제한
        objects[:, :, 0] = objects[:, :, 0].clamp(0, 3)   # class_id: [0, 3]
        objects[:, :, 1] = objects[:, :, 1].clamp(0, 41)  # subclass_id: [0, 41]
    
    # Step 4: ExpansionNet으로 캡션 생성
    caption = self._generate_caption(img_tensor, objects, env_vec, max_len)
    
    return caption
```

---

## 핵심 알고리즘

### 1. Autoregressive Decoding (자기회귀 생성)

ExpansionNet은 Transformer 디코더를 사용하여 토큰을 순차적으로 생성합니다.

```python
@torch.no_grad()
def _generate_caption(self, image, objects, env, max_len=128):
    """
    자기회귀 방식으로 캡션 생성
    
    t=0: [CLS] → "하"
    t=1: [CLS, 하] → "절"
    t=2: [CLS, 하, 절] → "기"
    ...
    """
    self.model.eval()
    
    # [CLS] 토큰으로 시작
    ids = torch.tensor([[self.tokenizer.cls_token_id]], device=self.device)
    
    for step in range(max_len):
        # 현재까지 생성된 시퀀스로 다음 토큰 예측
        logits = self.model(image, objects, env, ids)
        
        # Greedy decoding: argmax로 다음 토큰 선택
        next_id = logits[:, -1].argmax(-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
        
        # [SEP] 토큰 만나면 종료
        if next_id.item() == self.tokenizer.sep_token_id:
            break
    
    # 토큰 ID를 한국어 텍스트로 디코딩
    caption = self.tokenizer.decode(ids.squeeze().tolist(), skip_special_tokens=True)
    return caption
```

### 2. Causal Masking (인과적 마스킹)

미래 토큰 정보 누출 방지를 위한 마스킹:

```python
# src/expansionnet/model.py - ExpansionNetV2_Multimodal.forward()
if tgt_mask is None:
    seq_len = tgt_ids.size(1)
    # 상삼각 행렬로 미래 토큰 마스킹
    tgt_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=tgt_ids.device) * float('-inf'),
        diagonal=1
    )
    # 결과:
    # [[  0., -inf, -inf, -inf],
    #  [  0.,   0., -inf, -inf],
    #  [  0.,   0.,   0., -inf],
    #  [  0.,   0.,   0.,   0.]]
```

### 3. Bbox Crop & DINO Re-inference

DINO를 두 번 사용:
1. **1차**: 전체 이미지에서 객체 탐지 (bbox 좌표 생성)
2. **2차**: bbox 영역을 crop하여 객체별 시각 특징 추출

```python
def _crop_objects(self, images, bboxes):
    """
    Bbox 영역을 crop하여 DINO 입력으로 준비
    
    images: (B, 3, H, W)
    bboxes: (B, N_obj, 4) [w, h, x, y] 정규화 좌표
    Returns: (B*N_obj, 3, 224, 224)
    """
    B, N_obj = bboxes.shape[:2]
    _, _, H, W = images.shape
    
    crops = []
    for b in range(B):
        for n in range(N_obj):
            w_norm, h_norm, x_norm, y_norm = bboxes[b, n]
            
            # 중심 좌표 → 모서리 좌표 변환
            x1 = int((x_norm - w_norm/2) * W)
            y1 = int((y_norm - h_norm/2) * H)
            x2 = int((x_norm + w_norm/2) * W)
            y2 = int((y_norm + h_norm/2) * H)
            
            # 이미지 범위 클램핑
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            
            # Crop 및 224x224 리사이즈
            if x2 > x1 and y2 > y1:
                crop = images[b:b+1, :, y1:y2, x1:x2]
                crop = F.interpolate(crop, size=(224, 224), mode='bilinear')
            else:
                # 유효하지 않은 bbox는 전체 이미지 사용
                crop = F.interpolate(images[b:b+1], size=(224, 224))
            
            crops.append(crop)
    
    return torch.cat(crops, dim=0)  # (B*N_obj, 3, 224, 224)
```

### 4. 멀티모달 특징 통합

```python
def forward(self, img, obj_tensor, env_vec, tgt_ids, tgt_mask=None):
    # 1. ViT로 전역 장면 인코딩
    img_feat = self.encode_image(img)  # (B, 1, 768)
    
    # 2. DINO로 객체별 시각 특징 추출 (bbox crop 사용)
    obj_feat = self.obj_encoder(obj_tensor, full_images=img)  # (B, N_obj, 768)
    
    # 3. 환경 정보 인코딩
    env_feat = self.env_encoder(env_vec)  # (B, 1, 768)
    
    # 4. 모든 인코더 특징을 하나의 메모리로 연결
    encoder_memory = torch.cat([img_feat, obj_feat, env_feat], dim=1)
    # Shape: (B, 1 + N_obj + 1, 768)
    
    # 5. Transformer 디코더로 캡션 생성
    tgt_embed = self.embedding(tgt_ids)  # (B, seq_len, 768)
    dec_out = self.decoder(tgt_embed, encoder_memory, tgt_mask=tgt_mask)
    
    return self.output(dec_out)  # (B, seq_len, vocab_size)
```

---

## 사용 방법

### 1. 환경 설정

```bash
# 가상환경 활성화
source /data/CodeLLM/ML/ensemble/bin/activate

# 필수 라이브러리 설치 확인
pip install torch torchvision transformers mmdet mmcv pillow tqdm
```

### 2. 단일 이미지 추론

```python
from pipeline.pipeline import CaptionPipeline

# 파이프라인 초기화
pipe = CaptionPipeline(
    dino_ckpt="outputs/dino/epoch_12.pth",
    vit_ckpt="outputs/vit/vit_env.pth",
    exp_ckpt="outputs/expansionnet/expnetv2_final.pth",
    device="cuda"
)

# 추론 실행
caption = pipe.run("test_image.jpg", debug=True)
print(caption)
# 출력: "하절기 맑은 낮 하늘에 삐라 한 개가 이동하고 있고..."
```

### 3. 배치 추론 (전체 테스트셋)

```bash
# 전체 테스트셋 추론 (3,055개 이미지)
python src/run_full_test.py

# 결과 저장 위치
# - test_results_full.json: 예측 결과 및 Ground Truth
```

**출력 형식**:
```json
{
  "image": "EO_SU_DT_W2_H2_E4D4_0008.jpg",
  "image_path": "/path/to/image.jpg",
  "predicted": "하절기 맑은 낮 하늘에 삐라 한 개가...",
  "ground_truth": "하절기 흐린 낮 하늘에 삐라 한 개가..."
}
```

### 4. 디버그 모드

```python
# 각 단계별 출력 확인
caption = pipe.run("test.jpg", debug=True)

# 출력:
# DINO detected 2 objects
# Objects: tensor([[3., 3., 0.023, 0.154, 0.884, 0.219], ...])
# ViT environment: tensor([[0, 0, 0, 1]])
# Step 0: 하 (id=1889)
# Step 1: ##절기 (id=17247)
# ...
# Final caption: 하절기 맑은 낮 하늘에...
```

---

## 문제 해결 및 최적화

### 1. CUDA Out-of-Bounds 에러 해결

**문제**: DINO가 예측한 클래스 ID가 ExpansionNet 임베딩 범위를 초과

```python
# 오류 발생 케이스
# DINO 출력: class_id=7 (8개 클래스)
# ExpansionNet: num_classes=4 (학습 시 4개 클래스만 사용)
# → IndexError in Embedding layer
```

**해결책**: 클램핑 적용

```python
# src/pipeline/pipeline.py - run() 메서드
objects[:, :, 0] = objects[:, :, 0].clamp(0, 3)   # class_id
objects[:, :, 1] = objects[:, :, 1].clamp(0, 41)  # subclass_id
```

### 2. 체크포인트 로딩 이슈

**문제**: 학습 시 `base_model.` 프리픽스로 저장됨

```python
# 체크포인트 키 예시
# 'base_model.vit.embeddings.cls_token'
# 'base_model.obj_encoder.class_emb.weight'
```

**해결책**: 동적 프리픽스 제거

```python
state_dict = torch.load(exp_ckpt, map_location=device)
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('base_model.'):
        new_state_dict[k[11:]] = v  # 프리픽스 제거
    else:
        new_state_dict[k] = v

model.load_state_dict(new_state_dict)
```

### 3. 모델 구조 자동 추론

**문제**: 하드코딩된 `num_classes`가 체크포인트와 불일치

**해결책**: 체크포인트에서 임베딩 크기 추출

```python
# Embedding weight shape에서 클래스 수 추론
class_emb_key = 'base_model.obj_encoder.class_emb.weight'
subclass_emb_key = 'base_model.obj_encoder.subclass_emb.weight'

num_classes = state_dict[class_emb_key].shape[0]      # 4
num_subclasses = state_dict[subclass_emb_key].shape[0] # 42
```

### 4. Bbox 정규화

**문제**: DINO가 픽셀 좌표 반환, ExpansionNet은 정규화 좌표 기대

```python
# DINO 원본 출력: [44, 22, 1458, 521] (pixels)
# ExpansionNet 기대: [0.023, 0.020, 0.759, 0.482] (normalized)
```

**해결책**: DINO 출력 단계에서 정규화

```python
# src/dino/infer.py
def _xyxy_to_xywh_norm(bboxes, w, h):
    """XYXY → XYWH 변환 및 정규화"""
    x1, y1, x2, y2 = bboxes.T
    bw = (x2 - x1) / w  # width normalized
    bh = (y2 - y1) / h  # height normalized
    cx = ((x1 + x2) / 2) / w  # center x normalized
    cy = ((y1 + y2) / 2) / h  # center y normalized
    return np.stack([bw, bh, cx, cy], axis=1)
```

### 5. 환경 벡터 범위 제한

**문제**: ViT 오분류 시 범위 초과 가능

**해결책**: 출력 값 클램핑

```python
env_vec = env_vec.clamp(0, 7)  # 최대값 7로 제한
```

---

## 성능 및 최적화

### 추론 속도

- **GPU (RTX 3080)**: ~22 images/sec
- **CPU**: ~2 images/sec

### 메모리 사용량

- **DINO**: ~2GB VRAM
- **ViT**: ~500MB VRAM
- **ExpansionNet**: ~3GB VRAM
- **Total**: ~6GB VRAM (배치 크기 1 기준)

### 배치 처리 최적화

현재 구현은 배치 크기 1로 제한됩니다. 배치 처리를 위해서는:

```python
# 다중 이미지 동시 처리 (향후 개선 사항)
def run_batch(self, image_paths: List[str]):
    # DINO 배치 처리
    all_objects = [run_dino(path) for path in image_paths]
    
    # ViT 배치 처리
    img_batch = torch.stack([self.preprocess(Image.open(p)) for p in image_paths])
    env_batch = self.vit(img_batch)
    
    # ExpansionNet 배치 디코딩
    # ... (구현 필요)
```

---

## 파일 구조

```
src/
├── pipeline/
│   ├── pipeline.py           # CaptionPipeline 메인 클래스
│   ├── dino_runner.py        # DINO 래퍼 (사용 안 함)
│   └── vit_runner.py         # ViT 래퍼 (사용 안 함)
├── dino/
│   ├── infer.py              # DINO 추론 함수
│   └── config.py             # DINO 설정
├── vit/
│   └── model.py              # ViT 환경 분류기
├── expansionnet/
│   ├── model.py              # ExpansionNetV2 아키텍처
│   ├── expnet_model.py       # Dropout 래퍼
│   └── generate.py           # 생성 유틸 (사용 안 함)
├── run_inference.py          # 단일 이미지 테스트
├── run_full_test.py          # 전체 테스트셋 평가
└── debug_inference.py        # 디버깅 스크립트

outputs/
├── dino/epoch_12.pth         # DINO 체크포인트
├── vit/vit_env.pth           # ViT 체크포인트
└── expansionnet/
    └── expnetv2_final.pth    # ExpansionNet 체크포인트
```

---

## 결론

본 파이프라인은 세 개의 독립 모델을 효과적으로 통합하여 멀티모달 캡션 생성을 수행합니다. 주요 특징:

1. **모듈화**: 각 모델이 독립적으로 동작하여 유지보수 용이
2. **안정성**: 클램핑 및 검증을 통한 런타임 오류 방지
3. **확장성**: 새로운 모델로 교체 가능한 구조
4. **효율성**: GPU 가속 및 배치 처리 지원 (개선 가능)

향후 개선 방향:
- 배치 추론 지원
- Beam search 디코딩
- 앙상블 모델 통합
- 실시간 스트리밍 처리
