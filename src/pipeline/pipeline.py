import torch
from PIL import Image
from torchvision import transforms

from dino.infer import run_dino
from vit.model import ViTEnvClassifier

from expansionnet.expnet_model import ExpansionNetV2_WithDropout
from expansionnet.model import load_tokenizer
from expansionnet.generate import generate_caption


class CaptionPipeline:
    def __init__(
        self,
        dino_ckpt,
        vit_ckpt,
        exp_ckpt,
        device="cuda"
    ):
        self.device = device

        # ------------------
        # DINO
        # ------------------
        self.dino_ckpt = dino_ckpt
        # ⚠️ run_dino 내부에서 ckpt 사용 중이면 여기선 저장만

        # ------------------
        # ViT
        # ------------------
        self.vit = ViTEnvClassifier().to(device)
        self.vit.load_state_dict(
            torch.load(vit_ckpt, map_location=device)
        )
        self.vit.eval()

        # ------------------
        # ExpansionNet
        # ------------------
        self.tokenizer = load_tokenizer()
        
        # Load checkpoint first to determine architecture
        state_dict = torch.load(exp_ckpt, map_location=device)
        
        # Infer num_classes and num_subclasses from checkpoint
        # Check class_emb.weight shape
        class_emb_key = None
        subclass_emb_key = None
        for k in state_dict.keys():
            if 'class_emb.weight' in k and 'subclass' not in k:
                class_emb_key = k
            if 'subclass_emb.weight' in k:
                subclass_emb_key = k
        
        if class_emb_key and subclass_emb_key:
            num_classes = state_dict[class_emb_key].shape[0]
            num_subclasses = state_dict[subclass_emb_key].shape[0]
        else:
            # Fallback to defaults
            num_classes = 4
            num_subclasses = 42
        
        self.model = ExpansionNetV2_WithDropout(
            vocab_size=self.tokenizer.vocab_size,
            num_classes=num_classes,
            num_subclasses=num_subclasses,
            dropout=0.15
        ).to(device)
        
        # Remove 'base_model.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('base_model.'):
                new_state_dict[k[11:]] = v  # Remove 'base_model.' (11 chars)
            else:
                new_state_dict[k] = v
        
        self.model.load_state_dict(new_state_dict)
        self.model.eval()

        # Image preprocessing for ViT
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def run(self, image_path: str, max_len: int = 128, debug: bool = False):
        """
        Run full pipeline: DINO -> ViT -> ExpansionNet
        
        Args:
            image_path: Path to input image
            max_len: Maximum caption length
            debug: Print debug information
        
        Returns:
            Generated caption (str)
        """
        # 1. Run DINO detection
        dino_objects = run_dino(image_path)  # (N_obj, 6)
        
        if debug:
            print(f"DINO detected {dino_objects.shape[0]} objects")
            print(f"Objects: {dino_objects}")
        
        # 2. Load and preprocess image for ViT
        img_pil = Image.open(image_path).convert("RGB")
        img_tensor = self.preprocess(img_pil).unsqueeze(0).to(self.device)  # (1, 3, 224, 224)
        
        # 3. Run ViT environment classification
        with torch.no_grad():
            vit_logits = self.vit(img_tensor)
        
        # Extract environment predictions
        env_vec = torch.tensor([
            vit_logits["season"].argmax(-1).item(),
            vit_logits["night"].argmax(-1).item(),
            vit_logits["weather"].argmax(-1).item(),
            vit_logits["wave"].argmax(-1).item(),
        ], device=self.device).unsqueeze(0)  # (1, 4)
        
        if debug:
            print(f"ViT environment: {env_vec}")
        
        # 4. Prepare objects for ExpansionNet (add batch dimension and padding)
        if dino_objects.shape[0] == 0:
            # No objects detected, use dummy
            objects = torch.zeros(1, 1, 6, device=self.device)
        else:
            objects = dino_objects.unsqueeze(0).to(self.device)  # (1, N_obj, 6)
        
        # 5. Generate caption with ExpansionNet
        caption = self._generate_caption(
            img_tensor, 
            objects, 
            env_vec, 
            max_len=max_len, 
            debug=debug
        )
        
        return caption

    @torch.no_grad()
    def _generate_caption(self, image, objects, env, max_len=128, debug=False):
        """
        Generate caption using ExpansionNet decoder
        
        Args:
            image: (1, 3, 224, 224) preprocessed image
            objects: (1, N_obj, 6) object detections from DINO
            env: (1, 4) environment vector from ViT
            max_len: Maximum sequence length
            debug: Print debug info
        
        Returns:
            Generated caption string
        """
        self.model.eval()
        
        # Start with [CLS] token
        ids = torch.tensor(
            [[self.tokenizer.cls_token_id]],
            device=self.device
        )
        
        for step in range(max_len):
            # Forward pass through model
            logits = self.model(
                image,
                objects,
                env,
                ids
            )
            
            # Get next token
            next_id = logits[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
            
            if debug and step < 10:
                token = self.tokenizer.decode([next_id.item()])
                print(f"Step {step}: {token} (id={next_id.item()})")
            
            # Stop at [SEP] token
            if next_id.item() == self.tokenizer.sep_token_id:
                break
        
        caption = self.tokenizer.decode(
            ids.squeeze().tolist(),
            skip_special_tokens=True
        )
        
        if debug:
            print(f"Final caption: {caption}")
        
        return caption
