from __future__ import annotations


def generate_completions(
    model,
    tokenizer,
    prompts: list[str],
    device,
    max_new_tokens: int = 12,
    temperature: float = 0.8,
    top_p: float = 1.0,
) -> list[str]:
    import torch

    if not prompts:
        return []
    original_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    # Llama-family causal models are trained with a beginning-of-text token.
    # Add it to the prompt once; decoded output still contains only newly
    # generated response tokens.
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=True)
    tokenizer.padding_side = original_padding_side
    encoded = {key: value.to(device) for key, value in encoded.items()}
    do_sample = temperature > 0.0
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_width = encoded["input_ids"].shape[1]
    return [
        tokenizer.decode(output[index, prompt_width:], skip_special_tokens=True).strip()
        for index in range(len(prompts))
    ]
