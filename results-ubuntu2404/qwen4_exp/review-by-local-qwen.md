Based on the review of `port.py` against `hf_reference_excerpt.py`, here are the mathematical differences found:

### 1. PLE Gate Activation (Missing Absolute Value)
*   **Location:** `HpuPLELayer.forward` in `port.py` (approx. line 235).
*   **Reference:** `gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()`
    *   The reference computes the magnitude of the gate, clamps it, takes the square root, and then reapplies the sign. This results in a signed square-root-like activation: $\text{sign}(g) \cdot \sqrt{|g|}$.
*   **Port:** `gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()` is **missing** in the port.
    *   The port code is: `gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()`? **Wait**, looking closely at `port.py`:
    ```python
    gate = (key * query).sum(-1, keepdim=True) / math.sqrt(self.H)
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign() # This line IS present in port.py
    ```
    *   *Correction*: The line `gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()` **is** present in `port.py`. My initial scan missed it. Let me re-read carefully.
    *   Reference: `gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()`
    *   Port: `gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()`
    *   **Result:** No difference here.

### 2. PLE Short Convolution Implementation (Dilation Logic)
*   **Location:** `HpuPLELayer.forward` in `port.py` (approx. lines 245-250).
*   **Reference:** Uses `nn.Conv1d` with `dilation=self.dil`. The reference `_short_conv` pads the input with `short_conv_state_len` zeros on the left, then applies the conv. The `nn.Conv1d` with dilation $D$ and kernel size $K$ effectively looks at indices $t, t-D, t-2D, \dots$.
*   **Port:** Implements the convolution manually:
    ```python
    w = self.conv1d.weight.view(hc_hidden, self.kernel) # w[:, j]
    out = x.new_zeros(n, L, hc_hidden)
    for j in range(self.kernel):
        off = j * self.dil
        out = out + xp[:, off:off + L] * w[:, j]
    ```
    *   The reference `nn.Conv1d` with dilation $D$ computes: $y[t] = \sum_{k=0}^{K-1} w[k] \cdot x[t - (K-1-k)D]$.
    *   The port computes: $y[t] = \sum_{j=0}^{K-1} w[j] \cdot x[t + j \cdot D - \text{offset?}]$.
    *   Let's check the indexing. `xp` is `[hist, x]`. `hist` has length `state_len = (K-1)*D`.
    *   In `nn.Conv1d`, the kernel is applied such that `w[0]` corresponds to the leftmost tap (earliest time) and `w[K-1]` to the rightmost (current time) if `padding` is handled correctly, or vice versa depending on convention. PyTorch `Conv1d` weights are `[out, in, K]`. The operation is `y[t] = sum_k w[k] * x[t - (K-1-k)]` for dilation 1. For dilation $D$, it is `x[t - (K-1-k)D]`.
    *   So `w[0]` multiplies `x[t - (K-1)D]` (oldest). `w[K-1]` multiplies `x[t]` (newest).
    *   In the port: `xp` has `hist` (oldest) then `x` (newest).
    *   `xp[:, off:off+L]` where `off = j * D`.
    *   For `j=0` (oldest weight `w[0]`), it takes `xp[0:L]`. Since `xp` starts with `hist` of length `(K-1)D`, `xp[0]` is the oldest history. This matches `x[t - (K-1)D]`.
    *   For `j=K-1` (newest weight `w[K-1]`), `off = (K-1)D`. `xp[(K-1)D : (K-1)D + L]`. This corresponds to the start of `x` (since `hist` length is `(K-1)D`). This matches `x[t]`.
    *   **Result:** The manual convolution logic appears mathematically equivalent to `nn.Conv1d` with dilation, assuming `w` is viewed correctly. `self.conv1d.weight` is `[C, 1, K]`. `view(C, K)` preserves the order. The logic holds.

### 3. N-Gram Hashing: EOS Segment Handling
*   **Location:** `NGramHasher.ids` in `port.py` (approx. lines 145-150).
*   **Reference:** `_shift_right_ignore_eos` calculates `previous_eos` using `cummax` on `eos_positions`. It determines `segment_start` and `position_in_segment`. It shifts tokens within the segment. If the source position is before the segment start (or negative), it fills with EOS.
*   **Port:**
    ```python
    eos_pos = torch.where(hist == self.eos, pos, -1)
    prev_eos = torch.cat([torch.tensor([-1], device="cpu"), torch.cummax(eos_pos, 0).values[:-1]])
    pos_in_seg = pos - (prev_eos + 1)
    # ...
    valid = (pos_in_seg >= s) & (pos - s >= 0)
    shifted.append(torch.where(valid, hist[src], torch.full_like(hist, self.eos)))
    ```
    *   The reference uses `previous_eos` (exclusive of current position) to define the segment start.
    *   Reference: `previous_eos = cat([-1], cummax(eos_positions)[:-1])`. `segment_start = previous_eos + 1`.
    *   Port: `prev_eos = cat([-1], cummax(eos_pos)[:-1])`. `pos_in_seg = pos - (prev_eos + 1)`.
    *   This logic matches.
    *   However, there is a subtle difference in **how history is handled for the first token**.
    *   Reference: `token_history = cat([previous_context, input_ids])`. `previous_context` is initialized to EOS.
    *   Port: `hist = torch.cat([context, tokens])`. `context` is passed in.
    *   In `HpuPLELayer.forward`, `ctx_cpu` is initialized to EOS for new sequences.
    *   The hashing logic itself seems identical.

### 4. PLE State Update: History Padding/Truncation
*   **Location:** `HpuPLELayer.forward` state write-back (approx. lines 260-265).
*   **Reference:** The reference relies on `past_key_values.update_conv_state` which handles padding/truncation internally based on `conv_kernel_size`.
*   **Port:**
    ```python
    gather = (lens_t.view(-1, 1) + torch.arange(self.state_len, device=x.device).view(1, -1))
    new_hist = torch.gather(xp, 1, gather.unsqueeze(-1).expand(-1, -1, hc_hidden))
    ```
    *   `xp` is `[hist, x]`. `lens_t` is the number of valid tokens in `x`.
    *   The gather indices are `lens + [0, 1, ..., state_len-1]`.
    *   This attempts to grab the last `state_len` elements from `xp` ending at `lens + state_len - 1`.
    *   Since `xp` length is `state_len + L`, and `lens <= L`, the max index is `L + state_len - 1`, which is the last element.
    *   This correctly extracts the last `state_len` valid history elements.
    *   **However**, if `lens < state_len`, the reference `update_conv_state` typically pads with zeros (or EOS for tokens). The port gathers from `xp`. If `lens` is small, `gather` indices might point into `hist` (which is valid history) or `x`.
    *   Wait, `xp` contains `hist` (previous state) and `x` (current).
    *   If `lens` is small, say 1, and `state_len` is 4. Indices: `1, 2, 3, 4`.
    *   `xp[0]` is old hist[0]. `xp[1]` is old hist[1]... `xp[4]` is `x[0]`.
    *   This seems to correctly maintain the rolling window.

### 5. **CRITICAL ISSUE: PLE Gate Sign Handling**
Let me re-examine the gate calculation in `port.py` vs `hf_reference_excerpt.py`.

**Reference:**
```python
gate = (key_normed * query_normed).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
gated_value = torch.sigmoid(gate) * value.unsqueeze(-2)
```

**Port:**
```python
gate = (key * query).sum(-1, keepdim=True) / math.sqrt(self.H)
gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
gated = (torch.sigmoid(gate) * value.unsqueeze(-2)).flatten(-2)
```

These are identical.

### 6. **CRITICAL ISSUE: RMSNorm Weight Initialization/Scaling**
**Reference:**
`Qwen4ExpTextRMSNorm` initializes `self.weight = nn.Parameter(torch.zeros(dim))`.
Forward: `output = output * (1.0 + self.weight.float())`.

**Port:**
`GroupedGemmaRMSNorm` initializes `self.weight = nn.Parameter(torch.zeros(dim))`.
Forward: `return (xf * (1.0 + self.weight.float())).to(x.dtype)`.

These are identical.

### 7. **CRITICAL ISSUE: N-Gram Embedding Lookup Device/Type**
**Reference:**
`self.ngram_embedding(ngram_ids.to(execution_device))`
`ngram_ids` are `long`.

**Port:**
`emb = self.ple_embedding.ngram_embedding(all_ids.to(torch.int32).to(hidden.device)).flatten(-2)`

The port casts IDs to `int32`. The reference uses `long` (int64).
`nn.Embedding` in PyTorch accepts `long` or `int`. If the vocab size fits in `int32`, this is safe.
`padded_vocab` is derived from `ngram_vocab_size_base` and `ngram_size`.
If `padded_vocab` > $2^{31}-1$, this would crash or wrap.
Given `ngram_vocab_size_base` is usually small (e.g., 1000s) and `ngram_size` is small, `total_vocab_size` is likely small.
However, `ShardedNGramEmbedding` uses `ids - self.start`.
If `ids` are `int32` and `start` is `int`, subtraction is fine.
This is likely not a math error unless vocab is huge.

### 8. **CRITICAL ISSUE: PLE Convolution Weight View**
**Reference:**
`self.conv1d = nn.Conv1d(...)`
`F.silu(self.conv1d(hidden_states))`

**Port:**
`w = self.conv1d.weight.view(hc_hidden, self.kernel)`
`out = out + xp[:, off:off + L] * w[:, j]`

`nn.Conv1d` weight shape is `[out_channels, in_channels/groups, kernel_size]`.
Here `groups=hc_hidden`, so `in_channels/groups = 1`.
Shape is `[hc_hidden, 1, kernel]`.
`view(hc_hidden, kernel)` flattens the middle dim.
`w[:, j]` is `[hc_hidden]`.
`xp[:, off:off+L]` is `[n, L, hc_hidden]`.
Broadcasting `w[:, j]` (shape `[hc_hidden]`) against `[n, L, hc_hidden]` works correctly (multiplies last dim).
This is correct.

### 9. **CRITICAL ISSUE: Decoder Layer Order of Operations**
**Reference:**
1. `hidden_states = hidden_states + self.ple(...)`
2. `hidden_states, hyper_input, injection_weights = self.attn_hyper_connection(hidden_states)`
3. `hidden_states = self.linear_attn(...)` or `self.self_attn(...)`
4. `injection = hidden_states.unsqueeze(-2) * injection_weights.unsqueeze(-1)`
5. `hidden_states = hyper_input + injection.flatten(-2)`
6. `hidden_states, hyper_input, injection_weights = self.mlp_hyper_connection(hidden_states)`
7. `hidden_states = self.mlp(hidden_states)`
8. `injection = hidden_states.unsqueeze(-2) * injection_weights.unsqueeze(-1)`
9. `hidden_states = hyper_input + injection.flatten(-2)`

**Port:**
1. `hidden = self.ple(hidden, input_ids)` (Note: `HpuPLELayer.forward` returns `hidden + gated + out`. The reference `PLELayer.forward` returns `output` which is added to `hidden_states` in the decoder layer. The port's `HpuPLELayer` *adds* the result to `hidden` internally?
   Let's check `HpuPLELayer.forward` return: `return (hidden + gated + out).view(orig_shape)`.
   Reference `PLELayer.forward` returns `output` (which is `gated_value + short_conv`).
   Reference Decoder: `hidden_states = hidden_states + self.ple(...)`.
   Port Decoder: `hidden = self.ple(hidden, input_ids)`.
   Since `HpuPLELayer` returns `hidden + result`, and the decoder assigns `hidden = ...`, the addition happens inside the PLE module.
   **Mathematically equivalent.**

2. `x, hyper, inj = self.attn_hyper_connection(hidden)`
3. `out = self.linear_attn(...)` or `self.self_attn(...)`
4. `hidden = GatedResidual.combine(hyper, out, inj)`
   `combine` does `hyper_input + (block_out.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)`.
   This matches steps 4-5.

5. `x, hyper, inj = self.mlp_hyper_connection(hidden)`
6. `return GatedResidual.combine(hyper, self.mlp(x), inj)`
   This matches steps 6-9.

**Order is correct.**

### 10. **CRITICAL ISSUE: Hyper-Connection Input Mixing**
**Reference:**
`input_mix_weight = F.silu(self.input_mix_weight_down(hyper_input_normed) / self.hc_count)`
`input_mix_weight = torch.sigmoid(self.input_mix_weight_up(input_mix_weight))`
`input_mix_weight = input_mix_weight.unflatten(-1, (self.hc_count, self.hidden_size))`
`mixed_input = (input_mix_weight * hyper_input_normed.unflatten(-1, (self.hc_count, self.hidden_size))).mean(dim=-2)`

**Port:**
`w = F.silu(self.input_mix_weight_down(normed) / self.hc)`
`w = torch.sigmoid(self.input_mix_weight_up(w)).unflatten(-1, (self.hc, self.H))`
`mixed = (w * normed.unflatten(-1, (self.hc, self.H))).mean(dim=-2)`

**Identical.**

### 11. **CRITICAL ISSUE: N-Gram Hashing Multipliers**
**Reference:**
`multipliers.append(2 * (_splitmix64(value) % half_bound) + 1)`
`half_bound = max(1, multiplier_max // 2)`
`multiplier_max = max_long // max(unigram_vocab_size, 1)`

**Port:**
`half = max(1, (((1 << 63) - 1) // max(unigram_vocab, 1)) // 2)`
`return [2 * (_splitmix64(...) % half) + 1 ...]`

**Identical.**

### 12. **CRITICAL ISSUE: PLE Norms**
**Reference:**
`self.norm_key = Qwen4ExpTextRMSNorm(hc_hidden_size, group_size=self.hidden_size, eps=config.rms_norm_eps)`
`self.norm_query = Qwen4ExpTextRMSNorm(hc_hidden_size, group_size=self.hidden_size, eps=config.rms_norm_eps)`
`self.norm_conv = Qwen4ExpTextRMSNorm(hc_hidden_size, group_size=self.hidden_size, eps=config.rms_norm_eps)`

**Port:**
`self.norm_key = GroupedGemmaRMSNorm(hc_hidden, self.H, eps)`
`self.norm_query = GroupedGemmaRMSNorm(hc_hidden, self.H, eps)`
`self.norm_conv = GroupedGemmaRMSNorm(hc_hidden, self.H, eps)`

**Identical.**

### 13. **CRITICAL ISSUE: PLE Gate Scaling**
**Reference:**
`gate = (key_normed * query_normed).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)`

**Port:**
`gate = (key * query).sum(-1, keepdim=True) / math.sqrt(self.H)`

**Identical.**

### 14. **CRITICAL ISSUE: PLE Value Projection**
**Reference:**
`value = self.value_proj(embeddings)`
`gated_value = torch.sigmoid(gate) * value.unsqueeze(-2)`

**Port:**
`value = self.value_proj(emb)`
`gated = (torch.sigmoid(gate) * value.unsqueeze(-2)).flatten(-2)`

**Identical.**

### 15. **CRITICAL ISSUE: PLE Key Projection**
**Reference:**
`key_normed = self.norm_key(self.key_proj(embeddings)).unflatten(-1, (self.hc_count, self.hidden_size))`

**Port:**
`key = self.norm_key(self.key_proj(emb)).unflatten(-1, (self.hc, self.H))`

**Identical.**

### 16. **CRITICAL ISSUE: PLE Query Projection**
**Reference:**
`query_normed = self.norm_query(hidden_states).unflatten(-1, (self.hc_count, self.hidden_size))`

**Port:**
`query = self.norm_query(hidden).unflatten(-1, (self.hc, self.H))`

**Identical.**

### 17. **CRITICAL ISSUE: PLE Convolution Input**
**Reference:**
`gated_value_normed = self.norm_conv(gated_value.flatten(-2))`
`output = gated_value + self._short_conv(gated_value_normed, past_key_values)`

**Port:**
`gated_normed = self.norm_conv(gated)`
`out = ... conv ...`
`return (hidden + gated + out).view(orig_shape)`

Wait.
Reference: `output = gated_value + self._short_conv(gated_value_normed, ...)`
Port: `return (hidden + gated + out).view(orig_shape)`
The port adds `gated` (which is `gated_value`) and `out` (which is `short_conv`).
This matches `gated_value + short_conv`.
And the decoder adds this to `hidden`.
So `hidden + (gated + out)`.
Reference: `hidden + (gated_value + short_conv)`.
**Identical.**

### 18. **CRITICAL ISSUE: N-Gram Embedding Flattening**
**Reference:**
`return self.ngram_embedding(ngram_ids.to(execution_device)).to(ngram_ids.device).flatten(-2)`

**Port:**
`emb = self.ple_embedding.ngram_embedding(all_ids.to(torch.int32).to(hidden.device)).flatten(-2)`

**Identical.**

### 19. **CRITICAL ISSUE: PLE State History Storage**
**Reference:**
The reference uses `past_key_values.update_conv_state` for the short conv history.
The port manually manages `conv_hist` in `kv_cache[1]`.

In `port.py`:
`conv_hist = rows[:, :self.hist_f32].contiguous().view(torch.bfloat16)[:, :self.state_len * hc_hidden]`
`conv_hist = conv_hist.reshape(n, self.state_len, hc_hidden)`

In `hf_reference_excerpt.py`:
The reference does not show the internal `Cache` implementation, but `update_conv_state` is standard.
The port stores history as `bf16`.
The reference `hidden_states` in `_short_conv` are `float32` (due to `x.float()` in RMSNorm? No, `_short_conv` takes `hidden_states` which is `gated_value_normed`. `gated_value` comes from `value_proj` (Linear) and `sigmoid`. `value_proj` output dtype depends on input. `embeddings` are from `ngram_embedding`.
If `ngram_embedding` is `bf16`, `value` is `bf16`.
`gate` is `float32` (due to `sqrt` and `sign`? No, `gate` is computed from `key` and `query`. `key` and `query` are from `norm_key`/`norm_query` which return `type_as(x)`. If `x` is `bf16`, they are `bf16`.
`gate` calculation: `sum` of `bf16` is `bf16`. `sqrt` of `bf16` is `bf16`.
`sigmoid` of `bf16` is `bf16`.
`gated_value` is `bf16`.
`norm_conv` returns `bf16`.
`_short_conv` input is `bf16`.
`conv1d` on `bf16` is `bf16`.
So storing history as `bf16` is consistent with the reference's likely dtype flow if the model runs in `bf16`.

However, the reference `RMSNorm` casts to `float32` internally but returns `type_as(x)`.
The port `GroupedGemmaRMSNorm` casts to `float32` internally and returns `to(x.dtype)`.
This is consistent.

### 20. **CRITICAL ISSUE: N-Gram Hashing `shift_right_ignore_eos` Logic**
Let's look really closely at the shift logic.

Reference:
```python
source_positions = positions - shift
gather_positions = source_positions.clamp_min(0).unsqueeze(0).expand(batch_size, -1)
shifted = token_ids.gather(dim=1, index=gather_positions)
valid = (position_in_segment >= shift) & (source_positions.unsqueeze(0) >= 0)
return torch.where(valid, shifted, token_ids.new_full((), self.eos_token_id))
```

Port:
```python
src = (pos - s).clamp_min(0)
valid = (pos_in_seg >= s) & (pos - s >= 0)
shifted.append(torch.where(valid, hist[src], torch.full_like(hist, self.eos)))
```

The reference uses `token_ids` (the current input) for the fallback `new_full`.
The port uses `hist` (the concatenated history) for the fallback `full_like`.
`hist` contains `context` + `tokens`.
`token_ids` in reference is just `input_ids`.
In the reference, `shifted` is gathered from `token_ids`.
In the port, `shifted` is gathered from `hist`.

**Difference:**
The reference gathers from `token_ids` (current step inputs).
The port gathers from `hist` (context + current step inputs).

In the reference `forward`:
`token_history = torch.cat([previous_context, input_ids], dim=-1)`
`shifted_tokens = [self._shift_right_ignore_eos(token_history, shift) ...]`

Ah, the reference `_shift_right_ignore_eos` is called on `token_history`.
So `token_ids` inside `_shift_right_ignore_eos` IS `token_history`.
So the port gathering from `hist` is correct.

### 21. **CRITICAL ISSUE: PLE `norm_conv` Input**
Reference: `gated_value_normed = self.norm_conv(gated_value.flatten(-2))`
Port: `gated_normed = self.norm_conv(gated)`

`gated` in port is `(torch.sigmoid(gate) * value.unsqueeze(-2)).flatten(-2)`.
`gated_value` in reference is `torch.sigmoid(gate) * value.unsqueeze(-2)`.
Reference flattens it *before* norm.
Port flattens it *before* norm (inside the assignment to `gated`).
**Identical.**

### 22. **CRITICAL ISSUE: PLE `norm_conv` Output Usage**
Reference: `output = gated_value + self._short_conv(gated_value_normed, ...)`
Port: `out = ... conv ...`
`return (hidden + gated + out)`

Reference adds `gated_value` (un-normed) to `short_conv` (normed input).
Port adds `gated` (un-normed) to `out` (normed input).
**Identical.**

### 23. **CRITICAL ISSUE: PLE `norm_conv` Group Size**
Reference: `Qwen4ExpTextRMSNorm(hc_hidden_size, group_size=self.hidden_size, ...)`
Port: `GroupedGemmaRMSNorm(hc_hidden, self.H, ...)`

`hc_hidden_size` is `hc_count * hidden_size`.
`group_size` is `hidden_size`.
This groups the norm into `hc_count` groups of `hidden_size`.
**Identical.**

### 24. **CRITICAL ISSUE: PLE `norm_key` / `norm_query` Group Size**
Same as above. **Identical.**

### 25. **CRITICAL ISSUE:
