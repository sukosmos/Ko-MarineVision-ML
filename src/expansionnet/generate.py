import torch

@torch.no_grad()
def generate_caption(
    model,
    image,
    env,
    objects,
    tokenizer,
    device="cuda",
    max_len=128,
    debug=False
):
    model.eval()

    ids = torch.tensor(
        [[tokenizer.cls_token_id]],
        device=device
    )

    for _ in range(max_len):
        logits = model(
            image,
            objects,
            env,
            ids
        )

        next_id = logits[:, -1].argmax(-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)

        if next_id.item() == tokenizer.sep_token_id:
            break

    caption = tokenizer.decode(
        ids.squeeze().tolist(),
        skip_special_tokens=True
    )

    if debug:
        print("Generated ids:", ids)

    return caption
