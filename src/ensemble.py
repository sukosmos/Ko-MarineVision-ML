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


class FinalEnsemble:
    def __init__(
        self,
        device="cuda",
        vit_ckpt=None,
        exp_ckpt="outputs/expansionnet/expnetv2_final.pth",
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
        # ViT (Env Classifier)
        # ------------------
        self.vit = ViTEnvClassifier().to(device)
        if vit_ckpt is not None:
            self.vit.load_state_dict(
                torch.load(vit_ckpt, map_location=device)
            )
        else:
            print("[WARN] ViT checkpoint not provided. "
                  "Using randomly initialized ViT.")
        self.vit.eval()

        # ------------------
        # ExpansionNet
        # ------------------
        self.caption_model = ExpansionNetV2_Multimodal(
            vocab_size=self.tokenizer.vocab_size,
            num_classes=9,
            num_subclasses=9
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
    # ViT env inference
    # -------------------------------------------------
    @torch.no_grad()
    def run_vit(self, image: Image.Image) -> torch.Tensor:
        """
        image: PIL Image
        return: env tensor (4,)
        """
        img = self.img_tf(image).unsqueeze(0).to(self.device)

        out = self.vit(img)

        env = torch.tensor([
            out["season"].argmax(-1).item(),
            out["night"].argmax(-1).item(),
            out["weather"].argmax(-1).item(),
            out["wave"].argmax(-1).item(),
        ], device=self.device)

        return env  # (4,)

    # -------------------------------------------------
    # Caption generation (greedy decoding)
    # -------------------------------------------------
    @torch.no_grad()
    def generate_caption(
        self,
        img_tensor: torch.Tensor,
        env: torch.Tensor,
        objects: torch.Tensor,
        max_len: int = 128,
    ) -> str:
        """
        img_tensor: (1, 3, 224, 224)
        env: (4,)
        objects: (1, N, 6)
        """
        ids = torch.tensor(
            [[self.tokenizer.cls_token_id]],
            device=self.device
        )

        for _ in range(max_len):
            logits = self.caption_model(
                img_tensor,
                objects,                 # (1, N, 6)
                env.unsqueeze(0),         # (1, 4)
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
    # Full pipeline
    # -------------------------------------------------
    @torch.no_grad()
    def predict(self, image_path: str) -> dict:
        image = Image.open(image_path).convert("RGB")

        # ------------------
        # DINO
        # ------------------
        objects = run_dino(image_path)  # (N, 6)
        if objects.numel() == 0:
            objects = torch.zeros(1, 6)
        objects = objects.unsqueeze(0).to(self.device)  # (1, N, 6)

        # ------------------
        # ViT env
        # ------------------
        env = self.run_vit(image)  # (4,)

        # ------------------
        # Image tensor
        # ------------------
        img_tensor = self.img_tf(image).unsqueeze(0).to(self.device)

        # ------------------
        # Caption
        # ------------------
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
