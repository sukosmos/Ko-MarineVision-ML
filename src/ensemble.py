# src/ensemble.py
import torch
from PIL import Image
from torchvision import transforms

# ---- DINO ----
from dino.infer import run_dino

# ---- ViT ----
from vit.model import ViTEnvClassifier

# ---- ExpansionNet ----
from expansionnet.model import ExpansionNetV2_Multimodal, load_tokenizer


# -------------------------------------------------
# DINO → Training class mapping
# (반드시 학습과 맞춰야 함)
# -------------------------------------------------
# 학습 시: num_classes = 4
# 예시 매핑 (필요시 조정)
DINO_TO_TRAIN_CLASS = {
    0: 0,  # 어선
    1: 1,  # 군함
    2: 2,  # 상선
    3: 3,  # 항공기/기타
}


class FinalEnsemble:
    def __init__(
        self,
        device="cuda",
        vit_ckpt=None,
        exp_ckpt="../outputs/expansionnet/expnetv2_final.pth",
    ):
        self.device = device

        # ------------------
        # Image transform
        # ------------------
        self.img_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        # ------------------
        # Tokenizer
        # ------------------
        self.tokenizer = load_tokenizer()

        # ------------------
        # ViT (env classifier)
        # ------------------
        self.vit = ViTEnvClassifier().to(device)
        if vit_ckpt is not None:
            self.vit.load_state_dict(
                torch.load(vit_ckpt, map_location=device)
            )
        else:
            print("[WARN] ViT checkpoint not provided. "
                  "Env prediction may be noisy.")
        self.vit.eval()

        # ------------------
        # ExpansionNet (⚠ 학습 파라미터와 동일)
        # ------------------
        self.caption_model = ExpansionNetV2_Multimodal(
            vocab_size=self.tokenizer.vocab_size,
            num_classes=4,        # 🔴 학습과 동일
            num_subclasses=42     # 🔴 학습과 동일
        ).to(device)

        state = torch.load(exp_ckpt, map_location=device)
        missing, unexpected = self.caption_model.load_state_dict(
            state, strict=False
        )
        if missing or unexpected:
            print("[ExpansionNet] missing keys:", missing)
            print("[ExpansionNet] unexpected keys:", unexpected)

        self.caption_model.eval()

    # -------------------------------------------------
    # 1) ViT → env adapter (학습 규약)
    # -------------------------------------------------
    @torch.no_grad()
    def run_vit(self, image: Image.Image) -> torch.Tensor:
        """
        return: env tensor (4,)
        [season(0/1), night(0/1), weather(0~6), wave(0~6)]
        """
        img = self.img_tf(image).unsqueeze(0).to(self.device)
        out = self.vit(img)

        season = out["season"].argmax(-1).item()
        night = out["night"].argmax(-1).item()
        weather = out["weather"].argmax(-1).item()
        wave = out["wave"].argmax(-1).item()

        env = torch.tensor(
            [season, night, weather, wave],
            dtype=torch.float32,
            device=self.device
        )
        return env

    # -------------------------------------------------
    # 2) DINO → object adapter (학습 규약)
    # -------------------------------------------------
    def adapt_objects(self, dino_objs: torch.Tensor) -> torch.Tensor:
        """
        dino_objs: (N, 6)
        [dino_class, dino_subclass, w, h, x, y]

        return: (1, N, 6)
        [train_class, train_subclass, w, h, x, y]
        """
        if dino_objs.numel() == 0:
            # 학습 시 zero padding과 동일
            return torch.zeros(1, 1, 6, device=self.device)

        adapted = []
        for obj in dino_objs:
            dino_cls = int(obj[0].item())
            train_cls = DINO_TO_TRAIN_CLASS.get(dino_cls, 0)

            adapted.append([
                train_cls,   # class
                0,           # subclass (미사용 → 0 고정)
                obj[2].item(),  # w
                obj[3].item(),  # h
                obj[4].item(),  # x
                obj[5].item(),  # y
            ])

        adapted = torch.tensor(
            adapted,
            dtype=torch.float32,
            device=self.device
        )
        return adapted.unsqueeze(0)  # (1, N, 6)

    # -------------------------------------------------
    # 3) Caption generation (학습 test와 동일)
    # -------------------------------------------------
    @torch.no_grad()
    def generate_caption(
        self,
        img_tensor: torch.Tensor,
        env: torch.Tensor,
        objects: torch.Tensor,
        max_len: int = 128,
    ) -> str:
        ids = torch.tensor(
            [[self.tokenizer.cls_token_id]],
            device=self.device
        )

        for _ in range(max_len):
            logits = self.caption_model(
                img_tensor,
                objects,              # (1, N, 6)
                env.unsqueeze(0),      # (1, 4)
                ids
            )
            next_id = logits[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)

            if next_id.item() == self.tokenizer.sep_token_id:
                break

        return self.tokenizer.decode(
            ids.squeeze().tolist(),
            skip_special_tokens=True
        )

    # -------------------------------------------------
    # 4) Full pipeline
    # -------------------------------------------------
    @torch.no_grad()
    def predict(self, image_path: str) -> dict:
        image = Image.open(image_path).convert("RGB")

        # (1) DINO
        dino_objs = run_dino(image_path).to(self.device)
        objects = self.adapt_objects(dino_objs)

        # (2) ViT → env
        env = self.run_vit(image)

        # (3) Image tensor
        img_tensor = self.img_tf(image).unsqueeze(0).to(self.device)

        # (4) Caption
        caption = self.generate_caption(
            img_tensor,
            env,
            objects
        )

        return {
            "env": {
                "season": int(env[0]),
                "night": int(env[1]),
                "weather": int(env[2]),
                "wave": int(env[3]),
            },
            "objects": objects.squeeze(0).cpu().tolist(),
            "caption": caption
        }
