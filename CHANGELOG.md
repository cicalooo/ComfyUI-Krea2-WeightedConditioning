# Changelog

## Unreleased

## 1.0.0

- Pack is **Krea2 Split Encode**. Node class IDs unchanged (`Krea2Encode`, `Krea2EncodeOptions`, `Krea2MultiRef`). Menu titles: Split Encode, Split Encode Options, Split Ref.
- **Krea2 Encode** — T2I mix, K2 Edit grounding, and Identity Edit pixel path (`vae` + identity). No source latent sockets.
- **Split Encode Options** — people lock, emphasis, vision strengths, focus, masks, `ref_boost_a`, guide grounding px.
- **Split Ref** — up to 5 **Qwen placement guides** (clothing/objects). They are not VAE frames and do not replace `identity`.
- With guides connected, DiT is the person only (`ref_boost` ~4). Scene is Qwen-only so a fitted scene frame does not overwrite the face.
- Identity lives on Encode: `identity` (alias `image_b`); optional `scene` (alias `image`). Named sockets win over aliases.
- If Split Ref is connected with no person on Encode, slot 1 becomes identity and the rest stay guides.
- Removed **Krea2 Prompt Mix**. Old graphs stay on the previous GitHub release (`text_main` → `prompt`, `text_aux` → `aux`).
- Grounded aux spans use vision-aware prefix matching (`aux_id_span_grounded`).
- `k_bias` / `both` above 1.0 use a steeper key ramp. `value_scale` (default) is unchanged.
- Vendored Mustyrocks identity-edit source patch (Apache-2.0) for in-node pixel-path encode.
- Example workflow: `workflows/krea2_encode_identity.json`.

## 0.3.0

- Pack now ships only **Krea2 Prompt Mix** (including optional K2Edit grounding).
- Removed **Krea2 Conditioning Mix** and **Krea2 Weighted Conditioning**.
- License is Apache License 2.0.
- Fixed `aux_strength=0.0` so Prompt Mix omits aux before Qwen tokenization and exactly follows the main-only encode path instead of leaving aux context in the conditioning sequence.

## 0.2.0

- **Krea2 Prompt Mix** optional K2Edit grounding: `image`, `image_b`, `grounding_px`, `system_prompt` use Mustyrocks Qwen3-VL prep (28-pixel grid, training system prompt, scene-then-subject order). Encodes `[vision] [main @ 1.0] [aux @ aux_strength]` as one sequence. Text-only mix is unchanged when no image is connected.
- Auxiliary weights are placed after vision-token expansion so vision tokens, the main edit instruction, and chat-template tokens stay at 1.0.
- Attention **k-bias** composes with Mustyrocks `ref_boost` masks instead of replacing them. `value_scale` (default) already left that mask intact.
- Empty aux and `aux_strength=1.0` still emit grounded conditioning (no model patch).

## 0.1.0

- **Krea2 Prompt Mix** — encode a handwritten prompt and a moodboard/style prompt as one Qwen sequence; scale only the aux tokens. Subject stays at 1.0.
- **Krea2 Conditioning Mix** — concat two already-encoded conditionings and scale the aux slice (prefer Prompt Mix; concat duplicates the chat template).
- **Krea2 Weighted Conditioning** — `(phrase:weight)` inside a single prompt via attention V-scale / k-bias. `strength` only multiplies those terms; it is not whole-prompt scale.
