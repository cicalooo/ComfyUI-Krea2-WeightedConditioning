# Changelog

## 0.1.0

- **Krea2 Prompt Mix** — encode a handwritten prompt and a moodboard/style prompt as one Qwen sequence; scale only the aux tokens. Subject stays at 1.0.
- **Krea2 Conditioning Mix** — concat two already-encoded conditionings and scale the aux slice (prefer Prompt Mix; concat duplicates the chat template).
- **Krea2 Weighted Conditioning** — `(phrase:weight)` inside a single prompt via attention V-scale / k-bias. `strength` only multiplies those terms; it is not whole-prompt scale.
