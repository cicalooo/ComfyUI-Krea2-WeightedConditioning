# Reference

Details that do not belong on the README. Behavior matches the nodes, not the old Prompt Mix graph.

## Paths

Identity Edit turns on when **VAE + a person image** are connected. There is no mode switch.

**No images.** One Qwen sequence: prompt at 1.0, aux scaled. `aux_strength` 0 drops aux before tokenization. 1.0 encodes both and applies no attention patch.

**Images, no VAE.** Same sequence, plus Qwen3-VL vision tokens. No in-context latents. Debug says grounding only.

**Person + VAE, no Split Ref.** Mustyrocks pixel path. Order is fixed: scene = frame 1, person = frame 2. Swapping them drops likeness. `ref_boost` (~4) is the person. Scene boost is `ref_boost_a` on Encode Options (default 1, off). Single image (person only) is the strongest likeness this LoRA has.

**Person + VAE + Split Ref.** DiT is the person only, boost 4. Scene is Qwen-only. Guides are Qwen-only. The LoRA was trained for one ref, or scene-then-person. A third VAE frame, or a scene frame fitted onto the output while clothes are also in the tower, overwrites the face. Clothes still place because Qwen sees them.

Qwen order with guides: scene, then guides, then **person last**, so the instruction binds to the face. Empty focus text is filled in:

- Scene: layout only, do not copy any face.
- Guide: clothing or object only, do not copy any person.
- Person: this is the only person; preserve this face.

If Split Ref is connected and Encode has no person, slot 1 is identity and the rest are guides. Do not use that if `identity` or `image_b` is already connected.

## Aux

`aux_strength` scales aux tokens only. It is not CLIP `(prompt:w)` and it is not whole-prompt scale.

| Value | Effect |
|---|---|
| 0 | Aux omitted before tokenize. Main-only encode, no attention patch. |
| 0.35–0.55 | Typical moodboard weaken. Default 0.45. |
| 1 | Equal to the prompt. No patch. |
| >1 | Enforce. Needs emphasis `k_bias` or `both` on Options to ramp keys. `value_scale` only multiplies values. |

A long aux still dominates a short prompt at 0.45, because token count matters. Debug warns when aux is long during an identity edit. Clear it, or set strength to 0, to match stock Krea Edit.

Do not `ConditioningConcat` two Krea 2 encodes. Each already has the chat template.

## Sockets

| Socket | Role |
|---|---|
| `identity` | Person. VAE appearance ref. Prefer this over `image_b`. |
| `scene` | Optional layout. VAE frame 1 only when no guides are connected. |
| `image` / `image_b` | Aliases for scene / identity. Named sockets win. |
| `vae` | Qwen Image VAE. Pixel-path encode. Ignored for text-only. |
| `ref_boost` | Person likeness. 1 = off, ~4 start, >10 often over-copies. Ignored without VAE. |
| `latent` | Same empty latent as `KSampler.latent_image`. Optional. Skipped, the VAE can load mid-sample and some cards hitch every step. |
| `refs` | Split Ref output. Does not replace identity. |
| `options` | Encode Options. Disconnected = defaults. |

Guide grounding defaults to **384**. Identity and scene stay at **768**. Higher grounding holds likeness and can duplicate the subject if you go far above the trained range. Lower grounding makes edits obey harder.

## Options

| Input | Default | Notes |
|---|---|---|
| `people_count_lock` | off | Injects a phrase into the prompt at 1.0. `solo` or `match_refs`. |
| `emphasis_mode` | `value_scale` | Safest with `ref_boost` masks. `k_bias` / `both` add key bias on top of that mask. |
| `block_range` | all | `0-27` or `4,8,12` if the mix is too strong. |
| `apply_to` | cond | CFG 1 has no uncond pass. |
| `separator` | newline | How prompt and aux are joined. |
| `grounding_px` | 768 | Cap for scene and identity. 0 = native. |
| `guide_grounding_px` | 384 | Cap for Split Ref. |
| `ref_a_strength` / `ref_b_strength` | 1 | Qwen vision strength for scene / person. |
| `ref_a_focus` / `ref_b_focus` | empty | Overrides the automatic labels. |
| `ref_boost_a` | 1 | Scene VAE likeness. No effect while guides force the scene off the DiT. |
| `ref_boost_mask` / `_a` | empty | Region on the person / scene. Empty = whole image. |
| `fit_mode` | fit | Training-matched. `crop (legacy)` only for older Identity Edit weights. |
| `system_prompt` | empty | Replaces the Mustyrocks grounding system prompt. |

## Sampler

Wire both MODEL and CONDITIONING. LoRA first, this node last. Incompatible with Krea2 Apply Regional joint masks.

| Job | Model | Steps | CFG |
|---|---|---|---|
| Text mix, restage, add, recolor | Turbo | 8–12 | 1 |
| Removals | Raw | ~20 | ~3 |

Generate at **≤2MP**. Above that, source content bleeds or subjects duplicate.

At CFG > 1 the negative must be the trained unconditional: empty prompt, same images, **no VAE** on that second Encode. A second VAE tries to install another `krea2_edit` wrapper and errors. Throw away that node's MODEL.

`latent` only tells the node the output size so the VAE runs before sampling. It is not a source encode.

## What this is not

- Not a 5-ref Identity Edit. Frames above 2 were never trained. Guides exist so clothes can place without taking the person slot.
- Not KJNodes `Krea2 Prompt Weight`. That node weights phrases inside one string. This node splits prompt and aux, and owns the edit path. Same V-scale / k-bias idea. Do not stack them on the same blocks.
- Not a second Mustyrocks patch. The pixel path is vendored (Apache-2.0, [comfyui-krea2edit](https://github.com/lbouaraba/comfyui-krea2edit) 1.3.0). One wrapper per model.

## Tests

From the pack directory, no GPU:

```bash
python -m pytest tests -q
```
