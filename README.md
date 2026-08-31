# ComfyUI Krea 2 Weighted Conditioning

**Krea2 Prompt Mix** keeps a handwritten / edit instruction at strength 1.0 and scales a second (moodboard / style) prompt through K2 **self-attention**. Stock `(word:1.4)` is CLIP-era syntax; K2 reads the prompt through **Qwen3-VL**, which ignores it.

Optional **Mustyrocks K2 Edit** image grounding: the same Qwen3-VL prep (resize, 28-pixel vision grid, system prompt, scene-then-subject order). Source latents stay on `Krea2EditModelPatch`.

**Sampler CFG must be 1.0 for text-only mix.** For **Mustyrocks K2 Edit**, follow that pack’s CFG (Turbo CFG 1; Raw removals ~CFG 3). Wire **both** `MODEL` and `CONDITIONING`. Load LoRAs *before* this node so the attention patch is last.

`value_scale` (default) leaves K2Edit `ref_boost` masks in place. `k_bias` / `both` **add** key bias on top of that mask. Incompatible with **Krea2 Apply Regional** joint masks.

Category: `Krea2/conditioning`. Node: **Krea2 Prompt Mix**.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/cicalooo/ComfyUI-Krea2-WeightedConditioning.git
```

Restart ComfyUI. No extra Python packages. Needs a ComfyUI build with native Krea 2.

## Krea2 Prompt Mix

Moodboard dumps steal the subject if you concat two CLIP encodes or wrap the board in `(prompt:0.5)`. Prompt Mix encodes **one** sequence:

```
[ handwritten prompt ] [ moodboard / style ]
       strength 1.0         aux_strength (default 0.45)
```

```
STRING (subject) ──────── text_main ─┐
Moodboard.positive ────── text_aux ──┤ Prompt Mix ─┬─ MODEL ──────── sampler
CLIP (type krea2) ─────── clip ──────┤             └─ CONDITIONING ─ sampler.positive
UNET ──────────────────── model ─────┘
```

- `aux_strength` **0.35–0.55** is the usual weaken. `1.0` = equal; `0` = ignore aux.
- `apply_to` = **cond** (CFG 1 has no uncond pass).
- Do **not** `ConditioningConcat` two K2 encodes. Each encode already includes the chat template.

`debug` reports how many aux tokens were scaled. If that count is ~0, the join failed to split — check that `text_main` is really the prefix of the combined prompt.

### Optional K2Edit grounding

Connect Mustyrocks source **image** (and **image_b** for two-ref) on Prompt Mix. The node reproduces only the Qwen3-VL grounding prep from Mustyrocks. It does **not** inject source latents — keep `Krea2EditModelPatch` for that.

One grounded sequence:

```
[vision tokens] [main edit instruction @ 1.0] [aux / moodboard @ aux_strength]
```

Aux weights are applied **after** Qwen expands `<|image_pad|>` into vision tokens, so vision tokens, the main instruction, and chat-template tokens stay at 1.0.

```
Krea2 model
  -> Identity Edit LoRA
  -> Mustyrocks K2 Edit source patch
  -> Grounded Krea2 Prompt Mix model
  -> KSampler.model

source image(s) + main edit instruction + auxiliary prompt
  -> Grounded Krea2 Prompt Mix conditioning
  -> KSampler.positive
```

- Prompt Mix **replaces** Mustyrocks positive `Grounded Encode` (it runs the same grounding internally).
- For **CFG > 1**, the negative stays an **empty instruction grounded with the same image(s)** via Mustyrocks Grounded Encode (trained unconditional).
- `grounding_px` default 768 (0 = native). Empty `system_prompt` uses the Mustyrocks training default.
- Empty aux still returns grounded main conditioning (no patch). `aux_strength=1.0` returns grounded combined conditioning with no patch.

## Wiring notes

- Text-only Prompt Mix: CFG **1.0**. Identity edit with grounding: Turbo CFG 1; Raw “delete salient content” needs real CFG (Mustyrocks: Raw CFG ~3, ~20 steps) and a grounded empty negative.
- Keep an empty negative slot if your K2 graph goes grainy without one; `ConditioningZeroOut` on the mixed cond is fine for text-only CFG 1.
- **Conditioning Krea 2 Rebalance** can sit on the CONDITIONING output (tap EQ). Orthogonal to this node.
- `block_range`: `all` (default), `0-27`, or `4,8,12` if the effect compounds too hard.

## Tests

From this directory (no GPU):

```bash
python -m pytest tests -q
```

## Credits

Attention-side weighting for K2 follows the approach in [kijai/ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) (`Krea2 Prompt Weight`): V-scale for de-emphasis, k-bias for emphasis. Prompt Mix is this pack’s encode-once mix for subject + moodboard, with optional Mustyrocks K2Edit Qwen grounding.

## License

Apache License 2.0. See `LICENSE`.
