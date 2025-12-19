# src/smoke_test.py
from ensemble import FinalEnsemble

model = FinalEnsemble(
    device="cuda",
    vit_ckpt="../outputs/vit/vit_env.pth",
    exp_ckpt="../outputs/expansionnet/expnetv2_final.pth",
)

out = model.predict("test.jpg")
print(out)
