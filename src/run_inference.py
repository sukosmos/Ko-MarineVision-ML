# src/run_inference.py
import os
import glob
from pipeline.pipeline import CaptionPipeline

# Use absolute paths - find first image in dataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
test_images = glob.glob(os.path.join(BASE_DIR, "dataset/test/image/*/*.jpg"))
if not test_images:
    test_images = glob.glob(os.path.join(BASE_DIR, "dataset/train/image/*/*.jpg"))

if not test_images:
    raise FileNotFoundError("No test images found in dataset!")

test_img = test_images[0]

pipe = CaptionPipeline(
    dino_ckpt=os.path.join(BASE_DIR, "outputs/dino/epoch_12.pth"),
    vit_ckpt=os.path.join(BASE_DIR, "outputs/vit/vit_env.pth"),   
    exp_ckpt=os.path.join(BASE_DIR, "outputs/expansionnet/expnetv2_final.pth"),
    device="cpu"  # Use CPU for now
)

print(f"Running inference on: {test_img}")
out = pipe.run(test_img, debug=True)
print(f"\n=== Generated Caption ===")
print(out)
