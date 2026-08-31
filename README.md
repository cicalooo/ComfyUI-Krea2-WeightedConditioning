# ComfyUI Krea 2 Weighted Conditioning

Prompt weighting that actually reaches **Krea 2**. Stock `(word:1.4)` is CLIP-era syntax; K2 reads the prompt through **Qwen3-VL**, which ignores it or shoves the whole sentence around.

This pack patches K2 **self-attention** instead: scale a token’s **value** (weaken / subtract) or add a **k-bias** (more of the image attends to it).

**Sampler CFG must be 1.0.** Wire **both** `MODEL` and `CONDITIONING`. Load LoRAs *before* these nodes so the attention patch is last.

Not compatible with **Krea2 Apply Regional** (k-bias replaces the attention mask).

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/doggeddalle/ComfyUI-Krea2-WeightedConditioning.git
```

Restart ComfyUI. No extra Python packages. Needs a ComfyUI build with native Krea 2.

## Which node

| Node | Use when |
| --- | --- |
| **Krea2 Prompt Mix** | Handwritten subject at 1.0, moodboard / style prompt quieter. **Start here.** |
| **Krea2 Conditioning Mix** | You already encoded two `CONDITIONING`s and want concat + scale. Worse than Prompt Mix (two Qwen chat templates). |
| **Krea2 Weighted Conditioning** | `(red:1.5)` / `(glasses:-1)` *inside one prompt*. `strength` only multiplies those parentheticals — it does **not** turn a whole second prompt down to 0.5. |

Category: `Krea2/conditioning`.

## Krea2 Prompt Mix

Moodboard dumps are long and steal the subject if you concat two CLIP encodes or wrap the board in `(prompt:0.5)`. Prompt Mix encodes **one** sequence:

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

## Krea2 Weighted Conditioning

```
a portrait of (red:1.5) hair, (glasses:-1), street at night
```

| Weight | Effect |
| --- | --- |
| `< 1` (including negative) | Scale the token **value**. Negative subtracts the concept. |
| `> 1` | **k-bias** so more image tokens attend to the phrase (`emphasis_mode=k_bias`) |
| omitted | 1.0 |

`strength` (default 1) multiplies that effect across patched blocks. Removal is more reliable than emphasis. Nested parens and CLIP `[word]` are not supported.

If there are no `(phrase:weight)` terms, the node encodes the text and **does not patch** — use Prompt Mix for whole-prompt scale.

## Wiring notes

- CFG **1.0** (Turbo / Raw+Turbo-LoRA). Keep an empty negative slot if your K2 graph goes grainy without one; `ConditioningZeroOut` on the mixed cond is fine.
- **Conditioning Krea 2 Rebalance** can sit on the CONDITIONING output (tap EQ). Orthogonal to these nodes.
- `block_range`: `all` (default), `0-27`, or `4,8,12` if the effect compounds too hard.

## Tests

From this directory (no GPU):

```bash
python -m pytest tests -q
```

## Credits

Attention-side weighting for K2 follows the approach in [kijai/ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) (`Krea2 Prompt Weight`): V-scale for de-emphasis, k-bias for emphasis. Prompt Mix is this pack’s own encode-once mix for subject + moodboard.

## License

MIT.
