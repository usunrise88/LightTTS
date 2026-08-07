# GPU-dependent optimization ideas

Findings that need a GPU box (and a running server) to implement or validate. Everything here was
identified by source inspection only — **none of it has been measured**, so treat the payoff estimates as
hypotheses to test, not results.

Target deployment this was written against: **CosyVoice3 + TensorRT, 2× Blackwell (RTX PRO 5000 + PRO 6000,
both sm_120)**.

Baseline first: `cd test && python test_zs_stream.py` and `python test_zs_speed.py` both print RTF/TTFT
tables. Capture those numbers before changing anything, and re-run after each item below in isolation.

---

## 1. Drop the two device-wide syncs per estimator call (highest expected payoff)

`light_tts/server/tts_decode/model_infer/patch_conditional_cfm.py` wraps every TensorRT estimator call in
**two `torch.cuda.synchronize()` calls**:

```python
torch.cuda.synchronize()          # before
estimator.execute_async_v3(...)
torch.cuda.synchronize()          # after — "关键修复"
```

`ConditionalCFM.solve_euler` calls `forward_estimator` once per Euler step, and `n_timesteps=10`
(`cosyvoice/cosyvoice/flow/flow.py:139,270,398`). So this is **20 full-device synchronizations per audio
chunk**, each draining the entire CUDA context and destroying any CPU/GPU overlap in the decode stage. At
`decode_token_hop_len=25` (~1 s of audio per chunk) that is 20 syncs per second of speech, per stream.

The patch exists because TRT "有概率结果错误" — intermittently wrong results. That is the signature of a
missing stream dependency, not something that genuinely requires a device-wide barrier. Things to try, in
increasing order of risk:

1. `torch.cuda.current_stream().synchronize()` instead of `torch.cuda.synchronize()` — scopes the wait to
   the one stream instead of the whole device. Very likely sufficient and a drop-in change.
2. Drop the **pre**-call sync entirely. `execute_async_v3` is enqueued on `torch.cuda.current_stream()`, so
   prior work on that same stream is already ordered before it. The pre-sync is almost certainly redundant.
3. Replace the post-call sync with a `torch.cuda.Event` recorded after `execute_async_v3` and waited on only
   where the result is actually consumed.
4. Check whether the bug reproduces at all on TRT 10.14 + Blackwell. It may have been a TRT-version bug that
   is already fixed, in which case the patch can be removed outright.

**Validation is mandatory and must be statistical**, since the bug it guards against is intermittent: run
`test_zs_speed.py` for a few hundred iterations and byte-compare output wavs against a known-good run, not
just one request. If output ever diverges, back off one step.

Note `TrtContextWrapper.acquire_estimator()` already hands out a `stream` and the code runs `with stream:` —
so the plumbing for proper stream scoping is present.

---

## 2. Build a FP16/BF16 TRT engine for CosyVoice3

`light_tts/server/tts_decode/model_infer/model_rpc.py` hardcodes:

```python
elif version == CosyVoiceVersion.VERSION_3:
    self.fp16 = False          # <-- CosyVoice3 always runs the flow decoder in fp32
```

That `fp16` flag is what reaches `convert_onnx_to_trt`, which only sets `trt.BuilderFlag.FP16` when it is
true (`cosyvoice/cosyvoice/utils/file_utils.py:63-64`) and otherwise pins every input/output tensor to
`trt.DataType.FLOAT`. So on Blackwell the flow decoder runs a **pure FP32 engine** — leaving most of the
card's throughput unused. `--data_type` does *not* affect this; it only controls the LLM stage.

Worth testing: FP16, and BF16 (`trt.BuilderFlag.BF16`, not currently wired up at all) which has FP32-range
exponents and is usually safe where FP16 overflows. There is a known reason for caution — the CosyVoice
source has a comment at `flow_matching.py:94` that fp32 `x` "cause nan in trt fp16 inference" — so validate
audio quality, not just that it runs.

Suggested approach: make the dtype a CLI flag (e.g. `--flow_data_type {float32,float16,bfloat16}`) defaulting
to today's behavior, so this is opt-in and revertible. Note the plan filename already encodes the dtype
(`flow.decoder.estimator.{fp16|fp32}.sm120.plan`), so engines won't collide.

---

## 3. Re-tune `opt_shape` for the streaming chunk size

`CosyVoice2Model.get_trt_kwargs` (`cosyvoice/cosyvoice/cli/model.py:91-97`) builds the engine with:

```python
min_shape = [(2, 80, 4), ...]
opt_shape = [(2, 80, 500), ...]     # TRT tunes tactics for THIS shape
max_shape = [(2, 80, 3000), ...]
```

TRT selects kernels/tactics for `opt_shape`. In streaming mode the actual per-call mel length is driven by
`decode_token_hop_len=25` plus `flow_pre_lookahead_len` and the token→mel upsample ratio — plausibly a few
hundred frames, not 500, and definitely not on the first chunk (which adds `prompt_token_pad`).

Instrument `patched_forward_estimator` to log `x.size(2)` for a realistic workload, histogram it, then rebuild
the engine with `opt_shape` set to the mode of that distribution. If streaming and non-streaming shapes differ
a lot, consider a second optimization profile rather than one compromise shape.

Cheap to try, no correctness risk, and pure win if the current opt shape is wrong.

---

## 4. Add a TRT timing cache

`convert_onnx_to_trt` (`cosyvoice/cosyvoice/utils/file_utils.py:53-88`) sets a 4 GB workspace and nothing
else — no timing cache. Every engine build re-runs full tactic selection, which is why first boot on a new
card takes minutes (and it happens inside the decode worker's `init_model`, while `start_submodule_processes`
blocks on the readiness pipe with no timeout, so it looks like a hang).

Add `config.set_timing_cache(...)` persisted next to the `.plan`. Also worth evaluating
`config.builder_optimization_level` (default 3; level 4-5 builds slower but can produce faster engines — a
good trade for a long-lived server).

This is quality-of-life for rebuilds plus a possible engine-quality win.

---

## 5. Batch the flow/vocoder stage

`light_tts/server/tts_decode/manager.py:52` hardcodes `self.decode_max_batch_size = 1`, and `--decode_max_batch_size`'s
own help text says only 1 is supported. Meanwhile the TRT engine's shapes are all built with batch dim 2
(CFG-doubled), so batching multiple *requests* means rebuilding with a larger batch dimension.

For a single-user voice robot this is irrelevant. For concurrent sessions it is probably the single biggest
throughput lever left, since the vocoder currently runs one request at a time regardless of load. Needs real
concurrency to evaluate — use `test_zs_speed.py`, which sweeps `num_workers = [1, 2, 4, 8]`.

---

## 6. Re-run the Triton kernel autotuner on Blackwell

`test/kernel/triton_flashdecoding/llama_triton_flashdecoding_tuning.py` autotunes the flash-decoding kernel and
persists results via `LlamaFlashDecodingStage1KernelConfig.save_config`. Any checked-in configs were tuned on
older architectures; sm_120 has different SM counts, shared-memory limits and scheduling behavior.

Run it on both cards (they differ in SM count, so ideally tune per-card) and confirm the LLM stage actually
picks up the saved configs. `--mode ["triton_flashdecoding"]` is the relevant arg.

While there: verify CUDA graph capture is actually happening (`--disable_cudagraph` off,
`--graph_max_batch_size 16`, `--graph_max_len_in_batch 8192`) and not silently falling back to eager because a
request exceeded those bounds — the help text says it "will turn into eager mode if encounters a larger value".

---

## 7. Actually use the second GPU

Today essentially everything lands on GPU 0:

- LLM stage: `torch.cuda.set_device(0)`, hardcoded — `light_tts/server/tts_llm/model_infer/mode_backend/base_backend.py:49`
- Encode stage (ONNX sessions) and the HTTP-side frontend: also device 0
- Decode stage: the only one that spreads — `gpu_id = decode_proc_index % torch.cuda.device_count()`
  (`light_tts/server/tts_decode/manager.py:56`)

and `api_start.py:121` asserts `--decode_process_num <= len(lora_info)`, so with the default single style
`decode_process_num` can only be 1. **Net effect: with one style, the second card is completely idle.**

Two options:

- **Cheap and safe:** run two independent server instances, one per card, pinned with `CUDA_VISIBLE_DEVICES`,
  on **different ports** — the port is what namespaces every shm segment
  (`LIGHTLLM_UNIQUE_SERVICE_NAME_ID`), so distinct ports are mandatory, not cosmetic. Load-balance in front.
  This also sidesteps the asymmetry between the 5000 and the 6000.
- **Proper fix:** thread a `gpu_id` through the LLM stage instead of the hardcoded `set_device(0)`, so
  per-style LLM processes can be placed per card. Larger change; needs the multi-style config to be worth it.

Related: I already fixed `torch.cuda.get_device_capability(0)` → `get_device_capability(gpu_id)` in
`tts_decode/model_infer/model_rpc.py`, which would have picked the wrong TRT plan on a machine with mixed
architectures. Both your cards are sm_120 so it was inert for you, but it is now correct.

---

## 8. Measure whether `empty_cache()` / `synchronize()` in the decode loop is worth it

`light_tts/server/tts_decode/manager.py` runs, every 1000 decode batches:

```python
torch.cuda.empty_cache()
torch.cuda.current_stream().synchronize()
```

`empty_cache()` releases cached blocks back to the driver and forces later allocations to hit `cudaMalloc`,
which is slow; combined with a sync it stalls the pipeline. The encode stage does something similar on going
idle. With a steady-state server and a fixed `max_total_token_num`, this may be pure overhead. Profile with
`torch.cuda.memory_summary()` before deciding; if fragmentation is not actually growing, gate it behind a flag
or raise the interval a lot.

---

## 9. Tune the LLM GPU time-sharing semaphore

`--gpt_paral_num` (default 50) and `--gpt_paral_step_num` (default 200) control the only semaphore that is
actually acquired: each style's LLM router holds it across forward passes and releases every
`gpt_paral_step_num` steps (`light_tts/server/tts_llm/manager.py:196-217`). With one style it barely matters;
with several it decides how the per-style LLM processes interleave on the GPU. Defaults look untuned — sweep
them under a realistic multi-style load.

Also note `--encode_paral_num` and `--decode_paral_num` allocate semaphores that are **never acquired**
(confirmed by reading `tts_encode/manager.py` and `tts_decode/manager.py`). They are no-ops today; either wire
them up or drop them so they stop looking like working knobs.

---

## 10. LLM stage dtype / quantization on Blackwell

`--data_type float16` is the default. Blackwell handles BF16 natively, and `light_tts/common/quantization/`
carries inherited LightLLM machinery (including an `ENABLE_PINGPONG_FP8_GEMM` path in `vllm_quant.py`) that is
not exercised by any documented flag. For a 0.5 B model the LLM stage is unlikely to be the bottleneck versus
the flow decoder — measure where time actually goes before investing here. Listed for completeness, lowest
priority.

---

## Suggested order

1. **#1** (remove device-wide syncs) — biggest expected win, smallest diff, but needs statistical validation
2. **#3** + **#4** (opt_shape, timing cache) — low risk, quick
3. **#2** (FP16/BF16 engine) — big potential win, needs audio-quality validation
4. **#6** (Triton autotune) — mechanical, just needs the hardware
5. **#7** (second GPU) — start with two instances behind a load balancer
6. **#5**, **#8**, **#9**, **#10** — only once you have profile data saying they matter
