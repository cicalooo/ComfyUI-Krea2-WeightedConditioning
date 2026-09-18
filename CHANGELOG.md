# Changelog

## Unreleased

- Fixed `aux_strength=0.0` so Prompt Mix omits aux before Qwen tokenization and exactly follows the main-only encode path instead of leaving aux context in the conditioning sequence.

## 0.3.0

- Pack now ships only **Krea2 Prompt Mix** (including optional K2Edit grounding).
- Removed **Krea2 Conditioning Mix** and **Krea2 Weighted Conditioning**.
- License is Apache License 2.0.

## 0.2.0

- **Krea2 Prompt Mix** optional K2Edit grounding: `image`, `image_b`, `grounding_px`, `system_prompt` use Mustyrocks Qwen3-VL prep (28-pixel grid, training system prompt, scene-then-subject order). Encodes `[vision] [main @ 1.0] [aux @ aux_strength]` as one sequence. Text-only mix is unchanged when no image is connected.
- Auxiliary weights are placed after vision-token expansion so vision tokens, the main edit instruction, and chat-template tokens stay at 1.0.
- Attention **k-bias** composes with Mustyrocks `ref_boost` masks instead of replacing them. `value_scale` (default) already left that mask intact.
- Empty aux and `aux_strength=1.0` still emit grounded conditioning (no model patch).

## 0.1.0

- **Krea2 Prompt Mix** — encode a handwritten prompt and a moodboard/style prompt as one Qwen sequence; scale only the aux tokens. Subject stays at 1.0.
- **Krea2 Conditioning Mix** — concat two already-encoded conditionings and scale the aux slice (prefer Prompt Mix; concat duplicates the chat template).
- **Krea2 Weighted Conditioning** — `(phrase:weight)` inside a single prompt via attention V-scale / k-bias. `strength` only multiplies those terms; it is not whole-prompt scale.
