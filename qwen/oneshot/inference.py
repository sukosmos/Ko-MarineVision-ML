import os
import jsonlines
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig
)

# Hugging Face 캐시를 /data 파티션으로 이동
os.environ["HF_HOME"] = "/data/CodeLLM/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "/data/CodeLLM/huggingface_cache"

PROMPT_PATH = "prompt.txt"
TEST_DIR = "../../dataset/test/image"
OUTPUT_PATH = "./results/qwen_results.jsonl"
MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"
EXAMPLE_IMG_PATH = "example.jpg"


def load_prompt():
    with open(PROMPT_PATH, "r") as f:
        return f.read().strip()


# 서브폴더 포함 전체 이미지 로드
def get_all_images(root):
    img_list = []
    for path, dirs, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                img_list.append(os.path.join(path, f))
    return img_list


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 4bit quantization 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_threshold=6.0
    )

    # Qwen3-VL은 Vision2Seq → ImageTextToText 인터페이스로 변경됨
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
    )

    # example 이미지 1회 로드
    example_image = Image.open(EXAMPLE_IMG_PATH).convert("RGB")

    # 전체 테스트 이미지 로드 (서브폴더 포함)
    img_files = get_all_images(TEST_DIR)
    print(f"[INFO] 총 테스트 이미지 수: {len(img_files)}")

    with jsonlines.open(OUTPUT_PATH, "w") as writer:
        for img_path in tqdm(img_files):
            img_name = os.path.basename(img_path)
            image = Image.open(img_path).convert("RGB")

            # One-shot: 대화 형식으로 예시를 먼저 보여주고, 새 이미지 처리
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "해양 군사 이미지 캡션을 생성하세요. 예시를 참고하세요."},
                        {"type": "image", "image": example_image},
                    ]
                },
                {
                    "role": "assistant", 
                    "content": [{"type": "text", "text": "여름 날씨가 맑아 보이는 밤 하늘에 유인항공기 회전익 한 대가 비행 중이고 오물폭탄 한 개가 떠 있고 북 어선 한 척이 가고 있으며 파도 높이는 황천 7급이다."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "다음 이미지에 대해서도 같은 형식으로 캡션을 생성하세요. 계절, 주야, 날씨, 파도 단계, 객체를 모두 포함하세요."},
                        {"type": "image", "image": image},
                    ]
                }
            ]
            
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=prompt,
                images=[example_image, image],
                return_tensors="pt"
            ).to(device)

            try:
                output = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False
                )
                # 입력 토큰 길이만큼 제외하고 새로 생성된 부분만 디코딩
                generated_ids = output[0][inputs.input_ids.shape[1]:]
                caption = processor.decode(generated_ids, skip_special_tokens=True).strip()
                
                # assistant 태그 제거
                if caption.startswith("assistant\n"):
                    caption = caption[len("assistant\n"):].strip()

            except Exception as e:
                caption = f"[ERROR] {str(e)}"

            result = {
                "image": img_name,
                "caption_pred": caption
            }
            writer.write(result)
            
            # 매 이미지마다 로그 출력 및 파일 flush
            if (len(img_files) > 0):
                progress = (img_files.index(img_path) + 1)
                print(f"[LOG] {progress}/{len(img_files)} - {img_name}: {caption[:50]}...")
            
            # 버퍼 즉시 쓰기
            writer._fp.flush()

    print(f"[INFO] 결과 저장 완료 → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
