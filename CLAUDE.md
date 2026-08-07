# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

LightTTS serves CosyVoice2 / CosyVoice3 TTS models. The serving stack and the LLM-stage model code are
derived from [LightLLM](https://github.com/ModelTC/lightllm) — hence the `LIGHTLLM_*` env vars and the
`lightllm` module layout throughout `light_tts/`. `cosyvoice/` is a **vendored, modified fork** of upstream
CosyVoice (see `NOTICE`) supplying the frontend, flow-matching decoder and vocoder; note the doubled path,
the importable package is `cosyvoice/cosyvoice/`.

## Setup

```bash
git submodule update --init --recursive   # third_party/Matcha-TTS; needed by the flow decoder
pip install -r requirements.txt           # root file only — cosyvoice/requirements.txt pins a conflicting stack
```

Model weights go under `pretrained_models/` (gitignored). See `README.md` for the `snapshot_download` calls.

`cosyvoice/` is **not pip-installed**; several modules do `sys.path.insert(0, <repo>/cosyvoice)` at import
time (`light_tts/server/api_start.py:30`, `api_http.py:40`, `utils/start_utils.py:17`) and
`utils/load_utils.py:11` appends `third_party/Matcha-TTS`. Importing `light_tts.server.*` modules out of
order can therefore fail to resolve `cosyvoice`.

## Running

```bash
python -m light_tts.server.api_server --model_dir ./pretrained_models/Fun-CosyVoice3-0.5B-2512 --port 8080
bash launcher.sh                          # same thing, plus nvidia-cuda-mps-control -d
```

`LIGHTLLM_DEBUG=1` makes `api_start.py` run uvicorn in-process instead of spawning a gunicorn subprocess —
use it to debug or breakpoint the HTTP layer.

Version (CosyVoice2 vs 3) is auto-detected from whether `cosyvoice2.yaml` or `cosyvoice3.yaml` exists in
`--model_dir` (`CosyVoiceVersion.from_config_file`, `light_tts/utils/load_utils.py:14`). Full arg list is in
`light_tts/server/api_cli.py`; notable defaults `--data_type float16`, `--load_trt True`, `--load_jit False`,
`--max_total_token_num 65536`, `--running_max_req_size 30`, `--zmq_mode ipc:///tmp/`.
`light_tts/server/core/objs/start_args_type.py` is IDE-hint-only and its defaults are stale — read
`api_cli.py` for real values.

Keep `--httpserver_workers` at 1: `ReqIDGenerator.__init__` resets the shm counter to 0 on construction
(`light_tts/server/req_id_generator.py:17-19`), so multiple gunicorn workers would collide on request ids.

## Tests

There is no test framework and no CI. `test/*.py` are standalone client scripts that require an
already-running server and a GPU. **They must be run from inside `test/`** — they open `test_texts.json` and
`../cosyvoice/asset/zero_shot_prompt.wav` by relative path.

```bash
cd test
python test_zero_shot.py --port 8080 --cosyvoice_version 3 --num 1   # smallest smoke check; writes test/outs/
python test_zs_speed.py                                              # non-streaming benchmark
python test_zs_stream.py                                             # streaming benchmark
python test_bistream.py --port 8080                                  # WebSocket bi-stream (defaults to 8090!)
```

`/health` is itself a full end-to-end inference through every configured style
(`light_tts/utils/health_utils.py:45`), and it runs once at FastAPI startup — a 200 means the whole pipeline
works.

## Lint

```bash
pre-commit install && pre-commit run --all-files
black --line-length=120 light_tts/ test/
```

`.pre-commit-config.yaml` excludes `^cosyvoice/` from both hooks — do not reformat the vendored tree. Two
known inconsistencies: it pins `black==21.12b0` while `requirements.txt` pins `black==25.1.0` (different
formatting), and the flake8 hook passes `--config=.flake8`, a file that does not exist in the repo.

## Architecture

### Process topology

`api_start.py:normal_start` allocates all ports, then launches every worker **in parallel** through
`process_manager.start_submodule_processes` (`light_tts/utils/start_utils.py:16`), which also injects
`cosyvoice/` into each child's `sys.path`. Each child's entry function must send exactly `"init ok"` back
over an `mp.Pipe`; any failure kills the whole set. Start method is `spawn` (`api_server.py`). Gunicorn is
launched only after every child reports ready — meaning the LLM weights, the TRT flow engine and the ONNX
sessions all contend for GPU memory simultaneously at boot.

```
gunicorn → api_http.py (FastAPI + WS)
  └─ HttpServerManager        httpserver/manager.py   ← PULL on httpserver_port
       │ PUSH → tts1_encode_ports[i % encode_process_num]
  ┌────▼ tts_encode procs     × --encode_process_num (1)
       │ PUSH → tts_llm_ports[style_idx]
  ┌────▼ tts_llm procs        × one per style in style_config.json
       │ PUSH → tts_decode_ports[style_idx % decode_process_num]
  ┌────▼ tts_decode procs     × --decode_process_num (1)
       └ PUSH → httpserver_port  (bare `None` doorbell)
```

`--zmq_mode ipc:///tmp/` rewrites the prefix to include the port so multiple instances on one box don't
collide (`api_start.py:95-99`). **ZMQ only ever carries tiny control messages** — a `GroupReqIndexes`
dataclass, a bare shm index int, a `(request_id, output_len)` tuple, or `None`. All real data lives in
shared memory.

Three `mp.Semaphore`s are threaded into the children, but only `gpt_parall_lock` (`--gpt_paral_num`) is
actually acquired: each LLM router holds it across forward passes and releases every `--gpt_paral_step_num`
steps (`tts_llm/manager.py:196-217`). That is how per-style LLM processes time-share the GPU. The encode and
decode semaphores are allocated but never taken.

### Stages

- **Encode** (`light_tts/server/tts_encode/`) — `CosyVoiceFrontEnd.frontend_zero_shot` from the vendored
  tree: `speech_tokenizer_v2/v3.onnx` for speech tokens, the mel feat extractor, and `campplus.onnx` for the
  speaker embedding. Results go into shared memory, not back over ZMQ.
- **LLM** (`light_tts/server/tts_llm/`) — the autoregressive speech-token model. `RouterManager` owns
  continuous batching, the request queue (`req_queue/`), prompt caching (`dynamic_prompt/`) and preemption
  (`pause_strategy.py`); scheduling for the next batch is overlapped with the current forward via a
  1-thread executor (`manager.py:113-116`). Sampling is CosyVoice's RAS sampling with a resample-on-early-EOS
  retry (`model_infer/mode_backend/continues_batch/post_process.py`).
- **Decode** (`light_tts/server/tts_decode/`) — flow matching + HiFiGAN via the vendored `CosyVoice2Model` /
  `CosyVoice3Model`. Batch size is forced to 1. `model_infer/patch_conditional_cfm.py` monkeypatches
  `ConditionalCFM.forward_estimator` **as an import side effect** to add `torch.cuda.synchronize()` around
  the TRT call, which otherwise returns intermittently wrong results. Patching from `light_tts/` like this
  is preferred over editing `cosyvoice/`.

Audio leaves the server as raw `int16` PCM at 24 kHz with **no WAV header** (`api_http.py:161`); clients add
it themselves. Long text is split into sentences by `text_normalize(split=True)`, and **each sentence
becomes its own request_id** whose generators are concatenated in the response.

### Shared memory is the real IPC channel

`light_tts/server/core/objs/req.py` defines `Req` as a `ctypes.Structure` holding three `c_int64 * 32768`
arrays (`prompt_ids`, `output_ids`, `text_cache`) plus an embedded audio `CircularQueue`. `ShmReqManager`
maps `--running_max_req_size` of them into one segment, so that flag is a hard cap on in-flight requests and
`generate` busy-waits when it's exhausted. Every stage mutates the same struct in place; the field comments
at `req.py:100-125` document which process owns which write, and ordering matters — there are no barriers.

Lifetime is a manual refcount plus a `can_released_mark` convention (`Req.can_release`, `req.py:347`); only
`HttpServerManager.recycle_resource_loop` frees an index. Leaking one leaks the slot for the process
lifetime, which is what the periodic "left req_id ... refcount" log is watching for. Non-stream audio uses a
separate per-request segment `{server}_shm_gen_audios_{index}` that is never unlinked.

`SharedSpeechManager` (`core/objs/shm_speech_manager.py`) is the timbre cache: keyed by md5 of the prompt
wav, with `use_marks` progressing 0 → allocated → wav written → tokens/feat/embedding written. The encoder
**polls** and re-queues a request whose speech slot isn't ready. The LRU `OrderedDict` itself is per-process;
only `use_marks` is shared.

Every shm segment name is prefixed by `LIGHTLLM_UNIQUE_SERVICE_NAME_ID`, which is set to `str(args.port)` —
multi-instance isolation depends entirely on distinct ports.

### `world_size == 1` short-circuits the RPC layer

`ModelRpcClient` holds a direct `ModelRpcServer` reference when `world_size == 1`
(`tts_llm/model_infer/model_rpc.py:80-101`), so `await client.prefill(...)` is a blocking call inside the
router's event loop and no separate model process is spawned. The `RpcShmParams`/`rpc_event` machinery and
everything in `utils/dist_utils.py` is allocated but dormant — inherited from LightLLM's multi-GPU path,
which is not wired up here.

### Start args propagation

CLI args are JSON-serialized into the `LIGHTLLM_START_ARGS` env var by `set_env_start_args` and read back by
`get_env_start_args` (`light_tts/utils/envs_utils.py`). Spawned children and gunicorn workers recover args
from the environment, not from their call arguments.

### Model code (`light_tts/models/`, `light_tts/common/`)

`models/llama/` holds the base implementation and Triton kernels, `models/qwen2/` subclasses it, and
`models/cosyvoice2/model.py` subclasses `Qwen2TpPartModel`. Weights come from **two places** merged in
`_init_weights`: HF safetensors under `<model_dir>/CosyVoice-BlankEN` plus the style's `llm.pt`. The custom
pre-layer concatenates `[wte ; llm_embedding ; speech_embedding]` into one table so text, control and speech
ids share a vocab (hence `embed_offset`); the post-layer uses `llm_decoder` rather than `lm_head`.
`models/cosyvoice3/model.py` only overrides `eos_token` / `fill_token` / `stop_token_ids` / `embed_offset`.
`common/basemodel/` is the generic infer template, mem manager, CUDA-graph capture and quantization.

### CosyVoice2 vs 3 differences

Only token offsets and dtype, but they are defined in three places that must stay in sync: `LiteConfig` and
`load_yaml` in `utils/load_utils.py`; the two model classes above; and `tts_decode/model_infer/model_rpc.py`,
where V2 runs fp16 and **V3 runs fp32** with `load_jit` force-disabled. CosyVoice3 also requires the
`"...<|endofprompt|>"` prefix in `prompt_text`.

### Streaming chunk arithmetic

`decode_token_hop_len = 25` is hardcoded in `tts_llm/manager.py:97` and `tts_decode/manager.py:47`, and must
match `CosyVoice2Model.token_hop_len` (it mirrors training's `static_chunk_size`). The chunk boundary
`decode_token_hop_len (+ prompt_token_pad on the first chunk) + flow_pre_lookahead_len + token_offset` is
computed independently in `tts_llm/manager.py:405-420` and `tts_decode/decode_req.py:20-44`, where
`prompt_token_pad` was set back in `tts_encode/manager.py:149`. All three must agree or streaming audio
desyncs.

### Multi-LLM (styles)

`style_config.json` in the model dir lists `lora_info` entries (`style_name` + `llm_path`); with no such
file, `get_config_json` (`light_tts/utils/config_utils.py`) synthesizes one default style named `cosyvoice`
pointing at `llm.pt`. **One tts_llm process is spawned per style**, and `--decode_process_num` must be
`<= len(lora_info)`. Clients pick a style with `tts_model_name`, whose API default `"default"` resolves to
the first style. (This file was renamed from `config.json` in commit `ad1c76e`.)

### Three YAML loaders — pick the right one

In `light_tts/utils/load_utils.py`:

- `load_yaml` — full `load_hyperpyyaml`, **instantiates llm/flow/hift**. Only safe in the decode worker.
- `load_yaml_frontend` — instantiates tokenizer + feat extractor, nulls the big models, substitutes
  `_FakeLLM`/`_FakeFlow` shims that duck-type the config attribute chains. Used by the HTTP and encode sides.
- `load_yaml_lite` — pure YAML scrape via `LiteYamlLoader` (custom constructors that turn `!new:`/`!ref` into
  placeholders), zero instantiation. Used by the LLM and decode managers.

Adding a config field usually means touching `LiteConfig` **and** both fake shims.

## Conventions and known dead code

- Comments and log messages are largely in Chinese; match the surrounding style.
- Don't edit `cosyvoice/` — vendored fork (`NOTICE`, original commit `bc34459`), excluded from linting.
- `light_tts/server/tokenizer.py` imports a non-existent `models.sovits_gpt.tokenizer` and will `ImportError`
  if touched; nothing imports it.
- `light_tts/server/tts_encode/model_infer/frontend.py` is a stale copy of `CosyVoiceFrontEnd`; the live one
  is `cosyvoice/cosyvoice/cli/frontend.py`.
- `RouterManager._filter_batch` / `_merge_batch` / `_remove_batch` reference an unassigned `self.model_rpcs`
  and are dead. `_can_decode` is hardcoded `True`, so the pause/preempt path is currently unreachable.
