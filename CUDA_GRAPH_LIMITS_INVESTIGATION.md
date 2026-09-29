# CUDA command-buffer limits: mmBERT investigation and design plan

> Status: investigation and design note, 2026-09-29. This document is not a
> pull request description. No implementation or default change is implied by
> this note.

## Executive summary

An mmBERT W4A16 inference run at sequence length 8192 exposed excessive CUDA
memory retention on an RTX 5070 Ti. The run had cuDNN SDPA enabled and used the
current SM 12.0 command-buffer defaults. The MLX allocator reported a
13,570.221 MiB measured peak. The same workload, with
`MLX_MAX_OPS_PER_BUFFER=20` and `MLX_MAX_MB_PER_BUFFER=100`, reported a
2,491.681 MiB measured peak. Median model time stayed effectively unchanged:
144.734 ms versus 144.682 ms.

This is not evidence that mmBERT is incompatible with CUDA. It is evidence
that one hardware-class default can retain too much transient state for one
long-sequence lazy evaluation. The relevant MLX policy is shared by all CUDA
models. mmBERT reveals the problem because its long encoder evaluation creates
many large, sequential intermediate arrays.

The current overrides are process-wide environment variables. They are a valid
deployment workaround when one process has one known workload. They are not a
complete library interface when one process can run mmBERT, Qwen, and different
Qwen phases. A setting that is useful for a long encoder pass can reduce
performance for short decode evaluations in the same process.

The candidate upstream change is an opt-in, per-evaluation override for
command-buffer limits. It must apply when lazy arrays are materialized by
`eval`, not when a model builds the lazy graph. However, the API must not be
implemented from the downstream result alone. The mechanism must first be
reproduced with public MLX operations, without `mlx_cpp_inference`, MCI, model
weights, or BT11.

The MLX-native reproducer is the primary upstream evidence. It must include a
four-cell ablation because the existing downstream result changes both limits
at once:

| Case | Maximum operations | Approximate memory limit |
| --- | ---: | ---: |
| Current SM 12.0 default | 100 | 1000 |
| Operation-only change | 20 | 1000 |
| Memory-only change | 100 | 100 |
| Current successful pair | 20 | 100 |

This experiment does not try to select a universal default. It determines
whether the mechanism is reproducible in MLX itself and which control the first
scoped API must expose. If a suitable MLX-native workload cannot reproduce the
effect, API design stops and the diagnosis returns to model-specific factors.

The mmBERT/MCI benchmark remains important, but it becomes external real-model
acceptance evidence. The BT11 service remains product-level acceptance. Neither
is a dependency of the MLX change or its test suite.

## 1. Scope and terminology

The word "graph" refers to several different objects in this problem. They
must not be treated as one object.

- A **lazy graph** is the unevaluated MLX array graph rooted at the arrays
  passed to `mx::eval`.
- An **evaluation tape** is the ordered work that `eval_impl` creates from
  those roots.
- A **CUDA submission graph** is the work accumulated by one CUDA
  `CommandEncoder` before `commit()`.
- A **cached CUDA Graph** is the instantiated CUDA object that MLX can update
  and replay.

The requested scope is one lazy evaluation. One evaluation can be split into
several CUDA submission graphs. The limit controls the split points inside that
evaluation.

This note covers command submission and transient lifetime. It does not select
a quantization format, implement an attention kernel, or change model code.
W4A16 and fused SDPA are relevant controls, but they are separate mechanisms.

### 1.1 Evidence hierarchy

The work uses four evidence layers. Each layer answers a different question.

| Layer | Location | Purpose | Role in an MLX change |
| --- | --- | --- | --- |
| Mechanism benchmark | MLX repository, public MLX operations only | Reproduce transient retention and isolate the two limits | Primary upstream evidence |
| API regression tests | MLX C++ and Python tests | Prove scope, precedence, correctness, and no policy leakage | Required merge gate |
| Real-model acceptance | `mlx_cpp_inference` mmBERT/MCI probe | Confirm that the upstream mechanism fixes the discovery workload | External acceptance evidence |
| Product acceptance | BT11 service | Measure user-visible process memory, latency, and semantic output | Downstream product gate |

The MLX change must remain understandable and testable if the downstream
repository and model files are unavailable.

## 2. Source baselines

The Linux measurements used MLX commit:

```text
973e27f82ffe68dbd626cda31ba34997045d1eb7
```

The source inspected for this document is:

```text
64ea011cb65f14d9ce2737e60db9a4ae91ed7441
```

The relevant limit calculation, input accounting, `needs_commit()` condition,
and temporary retention are unchanged between these commits. The newer source
contains other error handling, synchronization, and stream-lifetime changes.

Relevant files in the current tree are:

- [`mlx/backend/cuda/device.cpp`](mlx/backend/cuda/device.cpp)
- [`mlx/backend/cuda/device.h`](mlx/backend/cuda/device.h)
- [`mlx/backend/cuda/eval.cpp`](mlx/backend/cuda/eval.cpp)
- [`mlx/transforms.cpp`](mlx/transforms.cpp)
- [`mlx/transforms.h`](mlx/transforms.h)
- [`mlx/utils.h`](mlx/utils.h)
- [`mlx/array.h`](mlx/array.h)
- [`mlx/backend/metal/device.cpp`](mlx/backend/metal/device.cpp)

## 3. Current MLX behavior

### 3.1 Device defaults

`get_graph_limits(Device&)` in the CUDA backend selects the following pairs.
The second value is named `max_mb_per_graph_` in the implementation, but it is
only an approximate score, as described below.

| CUDA device class | Maximum operations | Approximate memory limit |
| --- | ---: | ---: |
| Fallback | 20 | 100 |
| A100, SM 8.0 | 20 | 400 |
| H100, SM 9.0 | 100 | 1000 |
| B200, SM 10.0 | 100 | 1000 |
| Consumer Blackwell, SM 12.0 | 100 | 1000 |
| DGX Spark, SM 12.1 | 20 | 25 |

An RTX 5070 Ti reports compute capability 12.0. It therefore receives the same
`100/1000` pair as H100 and B200, even though their memory capacity and normal
workloads can differ substantially.

This fact does not by itself prove that the SM 12.0 default is wrong. It shows
why a result from one device family is not enough to choose a replacement
default for every SM 12.0 GPU.

### 3.2 Submission and temporary lifetime

The current execution path is:

```text
lazy output arrays
        |
        v
eval_impl builds and walks an evaluation tape
        |
        v
gpu::eval encodes each primitive into a per-stream CommandEncoder
        |
        +-- keep primitive inputs and siblings in temporaries_
        |
        +-- needs_commit() checks operation and memory scores
                 |
                 v
             commit()
                 |
                 +-- launch the CUDA submission graph
                 +-- move temporary references to the completion worker
                 +-- release references only after stream completion
```

For each evaluated primitive, `gpu::eval` retains the primitive inputs and
siblings in `CommandEncoder::temporaries_`. A commit moves those references to
a completion callback. They are released after the submitted stream work has
finished. A larger submission therefore keeps more intermediate buffers alive
at the same time. After release, the MLX allocator can keep those allocations
in its cache, so a large first submission can also leave a large process cache.

`eval_impl` commits every open GPU stream at the end of an evaluation. Reducing
the limits adds earlier commits inside the same evaluation. It does not add a
model-layer API and it does not change the mathematical operation order inside
each primitive.

### 3.3 The `MB` limit is not a byte counter

On CUDA, `CommandEncoder::set_input_array` currently performs:

```cpp
bytes_in_graph_ += arr.data_size();
```

`array::data_size()` is explicitly measured in elements, not bytes. The CUDA
counter also adds an input every time a primitive binds it. It does not
deduplicate repeated buffer pointers. Outputs are tracked for CUDA Graph
dependencies, but their sizes are not added when they are produced.

The commit condition shifts this element score by 20 bits and compares it with
the value called `max_mb_per_graph_`. Therefore:

- the value is not measured VRAM;
- the same numeric limit represents different byte sizes for FP32, BF16, and
  other dtypes;
- repeated use of one buffer can be counted more than once;
- output allocations and allocator fragmentation are not represented directly.

Metal also uses `data_size()`, but its encoder deduplicates input buffer
pointers, and `set_output_array` calls `set_input_array` before it registers the
output. CUDA and Metal therefore do not have identical accounting semantics.
The public environment-variable documentation already describes the value as
an approximate memory limit.

The first scoped API should not claim that this value is an exact byte or VRAM
budget. Correcting the accounting is a separate design problem because such a
change can alter existing split behavior on every device.

### 3.4 The existing controls are process-wide

`MLX_MAX_OPS_PER_BUFFER` and `MLX_MAX_MB_PER_BUFFER` are read through helpers in
`mlx/utils.h`. Each helper stores its first resolved value in a function-local
static. A CUDA `CommandEncoder` then copies the pair when the encoder is
constructed.

Consequences:

- the variables must be set before the relevant encoder is constructed;
- a process cannot select one pair for an mmBERT evaluation and another pair
  for a Qwen evaluation;
- changing the environment after initialization has no effect;
- the first cached default can also be problematic for heterogeneous devices
  in one process when no explicit environment value is present.

This scope is acceptable for a dedicated service worker. It is not sufficient
as the only control in a general inference library.

### 3.5 CUDA Graphs disabled

When `MLX_USE_CUDA_GRAPHS=0`, CUDA input sizes are not added to
`bytes_in_graph_`, but kernel launches still increase `node_count_` and the
operation limit can still cause commits. This gives a useful negative control:
it can separate the general completion-batch lifetime effect from CUDA Graph
capture and replay effects.

## 4. Why long mmBERT exposes the problem

mmBERT is an encoder-only workload, but "encoder-only" is not the cause. A
long encoder forward creates large token-by-hidden-state and feed-forward
intermediates across many sequential layers. MLX constructs these arrays
lazily. If the materialization of the result encodes many primitives before a
commit, references for several layers remain live in the same completion
batch.

W4A16 reduces persistent model storage. The measured active allocation after
model load was only 218.508 MiB. It does not remove all long-sequence
activations. As a result, transient retention can dominate the memory profile
even after the weight format is improved.

cuDNN SDPA changes the implementation of attention. It does not define commit
boundaries for the remaining projections, normalization, feed-forward blocks,
heads, and other primitives in the lazy evaluation. A fused attention path and
smaller command submissions are therefore orthogonal controls.

The same mechanism can appear in other models that create large sequential
intermediates. Examples include long prefill, vision encoders, audio encoders,
and large compiled subgraphs. The correct backend abstraction is workload or
evaluation shape, not the name "mmBERT".

This source-level explanation is the working hypothesis. An MLX-native
benchmark is required to separate the shared-runtime mechanism from details of
the downstream model implementation.

## 5. Downstream discovery evidence

The measurements in this section explain how the issue was found and show
that the candidate controls have practical value. They are not, by themselves,
an upstream reproducer. The upstream mechanism must be demonstrated using MLX
alone, as specified in Section 9.

### 5.1 Linux discovery system

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5070 Ti |
| Compute capability | 12.0 |
| Physical GPU memory | 16,303 MiB reported by `nvidia-smi` |
| Driver | 595.84 |
| `nvidia-smi` CUDA compatibility level | 13.2 |
| Build toolkit | CUDA 13.0, `nvcc` 13.0.88 |
| System CUDA symlink during inspection | CUDA 13.1 |
| cuDNN | 9.25.0 |
| Host compiler | GCC/G++ 13.4.0 |
| Build tools | CMake 4.2.3, Ninja 1.13.2 |
| MLX benchmark commit | `973e27f82ffe68dbd626cda31ba34997045d1eb7` |
| Model | `mmbert-q4-w4a16.mci` |
| Probe | `mlx_cpp_inference_benchmark_mmbert_memory` |
| SDPA request | `MLX_CUDA_USE_CUDNN_SDPA=1` |

The `nvidia-smi` CUDA value is a driver compatibility value. It is not the
toolkit used to build MLX. The build used the explicit CUDA 13.0 toolchain.

### 5.2 Default SM 12.0 sequence sweep

The default `100/1000` run reported the following internal allocator values.
Times are the low-level model-run times, not end-to-end service latency.

| Sequence length | Median time (ms) | Mean time (ms) | MLX measured peak (MiB) | Cache at end (MiB) |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 17.824 | 17.921 | 1,744.945 | 1,526.437 |
| 2048 | 30.774 | 30.822 | 3,560.809 | 3,342.300 |
| 4096 | 63.869 | 63.706 | 6,897.029 | 6,678.521 |
| 8192 | 144.734 | 144.814 | 13,570.221 | 13,351.713 |

At length 8192, the active model allocation was still 218.508 MiB at the end.
Almost all of the 13,570.221 MiB total was cached allocation created while the
evaluation ran.

### 5.3 Length-8192 comparison

The central downstream observation is the comparison at length 8192:

| Metric | Default `100/1000` | Override `20/100` | Change |
| --- | ---: | ---: | ---: |
| Median model time | 144.734 ms | 144.682 ms | -0.036% |
| Mean model time | 144.814 ms | 144.816 ms | +0.002% |
| MLX measured peak | 13,570.221 MiB | 2,491.681 MiB | -81.639%, 5.446x lower |
| MLX cache after warm-up | 9,290.070 MiB | 2,407.430 MiB | -74.086% |
| MLX active allocation at end | 218.508 MiB | 218.508 MiB | unchanged |
| NVML process peak | 9,608 MiB | 2,726 MiB | -71.628%, 3.525x lower |
| NVML device peak | 9,746 MiB | 2,864 MiB | -70.614%, 3.403x lower |

The allocator and NVML columns use different definitions and sampling methods.
They must not be subtracted from each other. They independently show the same
direction and approximate scale of improvement. The tuned NVML values above
come from the raw CSV files rather than the invalid summary fields described
below.

The successful override command was equivalent to:

```bash
env \
  MLX_CUDA_USE_CUDNN_SDPA=1 \
  MLX_MAX_OPS_PER_BUFFER=20 \
  MLX_MAX_MB_PER_BUFFER=100 \
  bash <mlx_cpp_inference>/scripts/linux/benchmark_mmbert_cuda_memory.sh \
    --probe "$PROBE" \
    --model "$MODEL" \
    --output "$OUTPUT" \
    --buckets 8192 \
    --warmup-runs 2 \
    --runs 10 \
    --sample-ms 20
```

The raw evidence directories on the test host were:

```text
~/mmbert-memory-20260928-173643/w4a16-cudnn-on/
~/mmbert-memory-graph20-100-20260928/
```

### 5.4 Measurement limitations

The result is strong enough to justify isolating the command-buffer controls,
but it is not yet a formal benchmark package.

1. The default run used one warm-up, five measured runs, and 50 ms NVML
   sampling. The `20/100` run used two warm-ups, ten measured runs, and 20 ms
   sampling. Both configurations must be rerun with identical settings.
2. The experiment changed both limits. It does not show whether operation
   count, memory score, or both caused the improvement.
3. The benchmark summary parser printed `292 MiB` and `430 MiB` as the tuned
   process and device peaks. The raw CSV files show sustained values of
   `2726 MiB` and `2864 MiB`. The summary values are invalid. The pattern is
   consistent with a non-numeric maximum comparison, but the parser must be
   inspected before the cause is recorded as fact.
4. The two model artifacts still need recorded SHA-256 values in the formal
   result bundle.
5. The memory probe did not by itself establish long-input numerical
   equivalence. The formal run must compare outputs between default and scoped
   limits.
6. One RTX 5070 Ti cannot establish a new default for all consumer Blackwell
   devices, H100, or B200.

### 5.5 macOS downstream service context

The W4A16 model was also measured through the full BT11 service on Apple
Silicon. This measurement includes the resident model and service process. It
samples physical process memory every 200 ms during five formal requests. It
excludes model load and warm-up samples.

| Sequence length | Average request time (s) | Peak physical process memory (MiB) |
| ---: | ---: | ---: |
| 1024 | 2.678 | 1,239.1 |
| 2048 | 3.630 | 1,603.9 |
| 4096 | 5.844 | 1,613.9 |
| 8192 | 8.994 | 2,303.8 |

These numbers are not directly comparable with the Linux low-level probe. The
platform, allocator, timing boundary, process contents, and sample interval are
different. They are useful context: the model format itself does not require a
13 GiB persistent footprint, and the tuned Linux result returns to the same
general memory scale.

### 5.6 What the evidence does and does not prove

The current evidence supports these statements:

- The current SM 12.0 `100/1000` pair is a poor fit for this long mmBERT
  evaluation on this RTX 5070 Ti.
- Earlier commits inside the lazy evaluation sharply reduce peak and cached
  memory.
- The successful pair does not cause a measurable latency loss in this test.
- Weight quantization and fused SDPA do not replace control of cross-primitive
  transient lifetime.
- The result is consistent with excessive transient retention in the shared
  CUDA submission path and justifies building an MLX-native reproducer.

The current evidence does not support these statements:

- A workload made only from public MLX operations reproduces the same effect.
- The proposed scoped API is necessary before that MLX-native reproduction
  exists.
- Every SM 12.0 GPU should use `20/100` by default.
- Encoder-only models always need smaller submission graphs.
- Decoder-only models always benefit from larger submission graphs.
- Both values in `20/100` are necessary.
- The approximate memory score is a correct VRAM estimator.
- A custom fused SDPA implementation is required to solve this memory issue.

## 6. Architecture decision

### 6.1 Do not branch on encoder-only versus decoder-only

Model architecture is a useful validation dimension. It is not a sufficient
backend dispatch key.

- An encoder can be short or long.
- Decoder prefill can look like a large encoder pass.
- Decoder token-by-token generation has different shapes and launch overhead.
- A multimodal model can execute an encoder and a decoder in one request.
- Compiled primitives can change the relationship between model layers and
  command-buffer operation counts.

MLX core sees arrays, primitives, streams, and devices. It should not contain a
special case for `ModernBERT`, `mmBERT`, `Qwen`, or a generic model-family
label. The application can select an expert policy because it knows the model,
phase, and request shape. MLX should provide a correctly scoped mechanism.

### 6.2 Process-wide tuning is sufficient only for isolated workers

The environment variables are enough when all of the following are true:

- the process runs one model or one stable workload class;
- the limits are known before the first MLX stream or encoder is initialized;
- all evaluations in the process can accept the same trade-off.

This can describe a dedicated `cms_bt11` worker. It does not describe a general
`mlx_cpp_inference` process that may run mmBERT and Qwen, or a decoder process
that has both prefill and decode phases.

The important boundary is not "the library supports several model classes."
The boundary is whether different evaluations with different desired policies
can occur in the same process. If they can, a process environment variable is
not enough.

### 6.3 Lessons from other runtimes

Serving runtimes have more workload knowledge than MLX core, but they show the
same general principle: graph policy follows phase and shape, and users retain
an explicit control.

vLLM supports several CUDA Graph modes, including no graph, piecewise graph,
full graph, full decode only, and full plus piecewise. Its dispatcher uses a
batch descriptor with values such as token count, request count, and batch
uniformity. It treats uniform decode separately from prefill or mixed batches.
It also caps default capture sizes to control startup time and memory, and its
vision encoder support captures separate token-budget buckets.

- [vLLM CUDA Graph design](https://github.com/vllm-project/vllm/blob/main/docs/design/cuda_graphs.md)
- [vLLM compilation configuration](https://github.com/vllm-project/vllm/blob/main/vllm/config/compilation.py)
- [vLLM vision encoder CUDA Graph design](https://docs.vllm.ai/en/latest/design/cuda_graphs_multimodal/)

TensorRT-LLM documents decoder graph buckets by batch size and normally limits
them to decode-only batches because prefill shapes depend on sequence lengths.
It exposes a separate encoder graph configuration with batch-size, total-token,
and sequence-length buckets. Its documentation also warns that a captured graph
can add substantial memory.

- [TensorRT-LLM LLM API](https://nvidia.github.io/TensorRT-LLM/llm-api/reference/LLM.html)
- [TensorRT-LLM encoder-decoder design](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/models/encoder-decoder.md)

These designs should not be copied directly into MLX. vLLM and TensorRT-LLM
own serving schedulers and model-phase metadata. MLX is a general lazy tensor
runtime. The transferable lesson is to expose a local execution policy and let
a higher layer select it from workload information.

## 7. Recommended upstream path

### 7.1 Entry criterion: reproduce the mechanism in MLX

Before changing a public API or backend interface, add an opt-in benchmark in
the MLX repository that uses only public MLX C++ operations. A suitable
conceptual location is:

```text
benchmarks/cpp/cuda_command_buffer_memory.cpp
```

The benchmark must not depend on MCI, `mlx_cpp_inference`, model weights, a
tokenizer, BT11, `nvidia-smi`, or Nsight tools. Its required measurements come
from public MLX memory APIs:

- `get_active_memory()`;
- `get_cache_memory()`;
- `get_peak_memory()` and `reset_peak_memory()`;
- `clear_cache()`;
- `synchronize()`.

Optional NVML sampling can corroborate a manual run, but it must not be needed
to build, run, or interpret the benchmark.

The benchmark should contain three parameterized workload modes:

1. **Operation-bound:** more than 100 ordinary primitives over relatively
   small arrays, with an aggregate input score below the memory threshold.
   This mode is intended to respond to the operation limit but not the memory
   limit.
2. **Memory-score-bound:** fewer than 20 ordinary primitives over large arrays,
   with repeated input accounting large enough to cross the approximate memory
   threshold. This mode is intended to respond to the memory limit but not the
   operation limit.
3. **Encoder-like combined:** a long chain over arrays shaped like
   `[batch, sequence, hidden]`, with branches and residual additions. This mode
   is intended to create overlapping temporary lifetimes and can respond to
   both limits.

The primitive chain should make donation less likely to erase the lifetime
effect. For example:

```cpp
auto a = sin(x);
auto b = cos(x);
x = add(a, b);
```

Every mode must consume a small checksum so that its outputs are evaluated and
can be checked for equality across limit settings. Sizes and chain depths must
be command-line parameters. The default benchmark size should be safe on
ordinary CUDA test machines; the RTX 5070 Ti investigation can select a larger
case explicitly.

If a simple primitive chain does not reproduce the effect, the next step is a
self-contained transformer-like residual and MLP block, still implemented
only with public MLX operations. If that also fails to reproduce a meaningful
limit-dependent memory difference, work on the scoped API stops. The
investigation then returns to model-specific graph structure, compiled
primitives, or downstream evaluation boundaries.

### 7.2 Candidate API goal

Allow one call to `eval` or `async_eval` to override the command-buffer split
limits without changing process state, device defaults, or later evaluations.

The policy must cover every GPU stream touched by that evaluation. It must be
copied at evaluation entry, so asynchronous execution does not depend on the
lifetime of a caller-owned options object. This is a candidate design until
the Section 7.1 entry criterion is satisfied.

### 7.3 Preferred API shape

The preferred first surface is an explicit evaluation option. Names below are
conceptual and must be reviewed against MLX naming conventions.

```cpp
struct CommandBufferLimits {
  std::optional<int> max_ops_per_buffer;
  std::optional<int> max_mb_per_buffer;
};

struct EvalOptions {
  std::optional<CommandBufferLimits> command_buffer_limits;
};

void eval(std::vector<array> outputs, const EvalOptions& options);
void async_eval(std::vector<array> outputs, const EvalOptions& options);
```

Existing overloads remain and use default options. A C++ caller would build the
lazy result first, then pass the result and options to the materialization
call:

```cpp
auto output = model.forward(input);  // Builds lazy arrays.
mx::eval(collect_roots(output), eval_options);  // Applies limits here.
```

Putting a guard only around `model.forward(input)` would not work. MLX is lazy;
the command encoder is used when the roots are evaluated.

For Python, a keyword option on `mx.eval` and `mx.async_eval` is preferable if
the variadic binding can keep a clear signature. A stackable context manager is
an acceptable alternative, but it introduces hidden thread-local state and is
easier to place around graph construction by mistake. A persistent per-stream
setter is not recommended because one stream can execute evaluations from
different model phases.

### 7.4 Internal data flow

The implementation should pass an immutable options snapshot through the
existing evaluation path:

```text
eval / async_eval
    -> eval_impl(options)
        -> gpu::eval(array, options)
            -> CommandEncoder::needs_commit(effective_limits)
```

The `CommandEncoder` should retain its environment/device pair as the fallback.
It should not be mutated for one evaluation. This avoids policy leakage into a
later evaluation and avoids races around long-lived encoders.

Precedence should be:

```text
per-evaluation field, when present
    > process environment override, when present
    > device default
```

An omitted per-evaluation field inherits the existing effective value. This
allows a caller to override only the operation limit if the ablation shows that
the memory-score limit is not required.

### 7.5 Public naming and the approximate memory score

The existing `max_mb` name is useful for compatibility, but it must be
documented as approximate. There are two possible paths:

1. Mirror `MLX_MAX_MB_PER_BUFFER` exactly in the first API. This is the smallest
   change and preserves existing semantics.
2. First replace the current element score with deduplicated byte accounting,
   then expose an exact byte-oriented name. This is cleaner, but it changes
   existing split behavior and needs a much wider performance study.

Given the available hardware, the first path is safer. The API documentation
must state that the value mirrors the existing heuristic and is not a memory
reservation or hard VRAM limit. The accounting cleanup should be a separate
proposal.

The four-cell ablation can reduce the first API further. If `20/1000` matches
`20/100`, the first use case only requires an operation override. The memory
field need not be exposed merely because an environment variable already
exists.

### 7.6 Backend scope

The current environment variables affect both Metal and CUDA. A generic
`CommandBufferLimits` option should therefore have matching semantics on both
GPU backends. CPU evaluation ignores it.

If reviewers prefer a CUDA-only first change, the option should be explicitly
named and namespaced as CUDA-specific. It must not silently behave differently
on Metal. A backend-neutral option is architecturally cleaner, but it adds Metal
tests to the first patch. Both platforms are available for validation, and no
default behavior changes, so the added validation cost is bounded.

### 7.7 Non-goals for the first change

- Do not change the SM 12.0, H100, B200, A100, or Metal defaults.
- Do not detect encoder-only or decoder-only models in MLX core.
- Do not add mmBERT, ModernBERT, or Qwen names to backend code.
- Do not implement an automatic transient-memory controller.
- Do not change CUDA or Metal memory accounting.
- Do not implement a new fused SDPA kernel.
- Do not change allocator cache policy.
- Do not modify `mlx_cpp_inference` in the same upstream patch.
- Do not make the upstream benchmark or tests depend on a downstream
  repository, model artifact, service, NVML, or profiler.

## 8. File-level implementation plan

The benchmark comes before the implementation. Whether it remains in the
eventual upstream change as an opt-in benchmark target can be decided during
review, but the result must be reproducible from an MLX checkout without
downstream sources or private artifacts.

The exact implementation file list depends on whether the first API is
backend-neutral.

| File | Planned responsibility |
| --- | --- |
| `benchmarks/cpp/cuda_command_buffer_memory.cpp` | Provide the parameterized public-API mechanism benchmark, including workload mode, shape, chain depth, timing, MLX allocator metrics, and output checksum. |
| `benchmarks/cpp/CMakeLists.txt` | Add an optional benchmark target if the reproducer is retained in the MLX tree. |
| `mlx/transforms.h` | Define the public evaluation option and overloads while preserving existing calls. |
| `mlx/transforms.cpp` | Snapshot options at evaluation entry and pass them through `eval_impl`. |
| `mlx/backend/gpu/eval.h` | Add the internal option or effective-limit parameter to GPU evaluation. |
| `mlx/backend/cuda/eval.cpp` | Use the evaluation limits for the `needs_commit` decision. |
| `mlx/backend/cuda/device.h` | Define a small limit value type or update the `needs_commit` interface. |
| `mlx/backend/cuda/device.cpp` | Resolve per-evaluation values over the existing encoder fallback values. |
| `mlx/backend/metal/eval.cpp` | Apply the same option if the public API is backend-neutral. |
| `mlx/backend/metal/device.h` | Accept an effective limit without mutating the device default. |
| `mlx/backend/metal/device.cpp` | Resolve and check the backend-neutral limits. |
| `python/src/transforms.cpp` | Bind the option without breaking the existing variadic array/tree interface. |
| `tests/eval_tests.cpp` | Test default inheritance, explicit values, synchronous and asynchronous evaluation, and no leakage. |
| `tests/gpu_tests.cpp` | Force early command boundaries and verify correct results across them. |
| `tests/cuda_tests.cpp` | Add CUDA-specific coverage if graph accounting or CUDA-only API behavior is involved. |
| `python/tests/test_eval.py` | Test the Python surface, tree flattening, errors, and option lifetime. |
| `docs/src/python/transforms.rst` | Expose the updated evaluation API documentation. |
| `docs/src/usage/lazy_evaluation.rst` | Explain that limits apply at materialization, not lazy graph construction. |

This is a design map, not permission to edit all listed files. The smallest
reviewable implementation should be chosen after the ablation and API review.

## 9. Validation plan

### Gate 0: preserve the downstream discovery evidence

1. Fix the downstream benchmark summary parser so numeric maxima are compared
   as numbers.
2. Record the probe binary hash, model SHA-256, MLX commit, MCI commit, build
   configuration, toolkit paths, driver, and cuDNN version.
3. Rerun the downstream default and `20/100` cases with identical warm-up,
   measurement, process-lifetime, and sampling settings.
4. Use a fresh process for every case. An allocator cache from a previous large
   run must not contaminate a smaller-limit run.
5. Keep raw `nvidia-smi` CSV, process CSV, probe log, and summary together.

This gate makes the original observation auditable. It is not a prerequisite
for building the MLX-only reproducer, and it is not part of the upstream MLX
test suite.

### Gate 1: reproduce the mechanism using MLX only

Build and run the Section 7.1 benchmark from the MLX tree. Run each setting in
a fresh process because the existing environment overrides are resolved into
function-local static values. Do not reuse one process and mutate its
environment between cases.

Exercise all three workload modes:

- operation-bound;
- memory-score-bound;
- encoder-like combined.

For every run, record:

- MLX commit and build configuration;
- CUDA device name and compute capability;
- workload mode, shape, dtype, chain depth, and seed;
- effective `MLX_MAX_OPS_PER_BUFFER` and `MLX_MAX_MB_PER_BUFFER` values;
- warm-up and measured-run counts;
- minimum, median, mean, p95, and maximum evaluation time;
- active, cache, and peak MLX memory before evaluation, after warm-up, after
  each measured run, and after `clear_cache()`;
- the output checksum and its comparison with the default case.

The required result is a repeatable change in internal MLX peak or cache
memory that is larger than run-to-run noise, while the checksum remains
equivalent. No fixed MiB value belongs in a portable unit test. For the
combined RTX 5070 Ti diagnostic, the expected result is a large reduction in
the same direction as the downstream mmBERT observation, without pathological
latency growth.

If the ordinary primitive chains do not reproduce the effect, try the
self-contained transformer-like block described in Section 7.1. If neither
does, stop the scoped-API proposal. Record the negative result and investigate
compiled primitives, graph topology, or downstream materialization instead of
using the mmBERT result to claim a general MLX mechanism.

### Gate 2: isolate the two controls on the RTX 5070 Ti

Run the encoder-like combined workload with the four limit pairs from the
executive summary. Keep the shape, dtype, seed, build, run counts, and process
lifetime fixed. Start a fresh process for each cell.

The operation-bound and memory-score-bound modes are controls for the intended
separation. The operation-bound case should respond to `max_ops` without
crossing the approximate memory threshold. The memory-score-bound case should
respond to `max_mb` without reaching the operation threshold. If those controls
do not behave as designed, adjust the benchmark construction before drawing a
conclusion from the combined case.

Use `MLX_SAVE_CUDA_GRAPHS_DOT_FILE` or temporary local instrumentation to count
submissions for one measured evaluation. Do not add a permanent test-only hook
to production code. Then run a secondary diagnostic with
`MLX_USE_CUDA_GRAPHS=0` and operation limits 100 and 20. This separates the
general completion-batch lifetime effect from CUDA Graph capture and replay.
It is not a proposed production setting.

Decision rule:

- If `20/1000` is equivalent to `20/100`, an operation override is sufficient
  for the first demonstrated use case.
- If `100/100` is equivalent to `20/100`, the approximate memory override is
  sufficient for the first demonstrated use case.
- If neither single change matches the pair, the demonstrated use case needs
  both fields.
- If the four cells do not produce a stable separation, stop API design and
  audit the benchmark rather than choosing fields from the downstream result.

### Gate 3: implement and prove evaluation scope without changing defaults

Only after Gates 1 and 2 establish the mechanism and required field set should
the candidate scoped API be implemented. The upstream unit and integration
tests must cover:

- existing calls select the same device and environment defaults as before;
- one evaluation can select a smaller limit;
- the next evaluation inherits the default again;
- a partial option inherits omitted fields;
- precedence is per-evaluation field, then process environment, then device
  default;
- synchronous and asynchronous evaluations copy the option safely;
- one evaluation that uses multiple streams applies one policy to all touched
  GPU streams;
- an exception does not leak policy to a later evaluation;
- forced boundaries preserve results and the existing fence, event, donation,
  and dynamic-slice update behavior;
- CPU evaluation ignores the option cleanly;
- Metal and CUDA both honor a backend-neutral option, if that design is used.

The C++ tests are the required structural and correctness gate. Add Python
tests only if the first public change includes a Python surface. Unit tests
must not assert an absolute CUDA memory value or a performance ratio: allocator
state, hardware, drivers, and build configuration make such thresholds
non-portable. The opt-in mechanism benchmark supplies the manual performance
evidence.

Because no default changes, this phase does not require choosing a universal
hardware performance boundary. It requires API, correctness, scoping, and
regression evidence.

### Gate 4: confirm the real model externally

Use the existing `mlx_cpp_inference` probe as an external acceptance test with
both available model artifacts:

- `mmbert-q4_0.mci`, as the historical high-memory or OOM control;
- `mmbert-q4-w4a16.mci`, as the current candidate.

For lengths 1024, 2048, 4096, and 8192:

1. Run the default path and the scoped path in separate fresh processes.
2. Check all outputs for finite values.
3. Compare logits or embeddings with recorded maximum absolute and relative
   error.
4. Compare decoded labels, entities, and relations exactly where applicable.
5. Report low-level inference time and allocator or NVML memory separately
   from downstream service time and process RSS.

Add one same-process isolation sequence:

```text
mmBERT evaluation with scoped limits
Qwen evaluation with defaults
mmBERT evaluation without scoped limits
```

This is the acceptance case that a process environment variable cannot
express. It confirms that the new scope solves the downstream library need and
does not silently tune Qwen. It is not a dependency of the MLX unit tests.

### Gate 5: downstream integration and product acceptance

Only after the upstream API and tests are clear, use the option at the central
materialization boundary in `mlx_cpp_inference`.

The integration must:

- attach policy to a model or request execution decision, not to process
  startup;
- configure the actual `mx::eval` call that materializes all output roots;
- leave Qwen and other models on MLX defaults unless they have independent
  evidence;
- avoid a flag named only for mmBERT if the mechanism is general;
- keep service benchmarking in the downstream service repository.

The exact downstream file cannot be selected until the production
materialization call is traced. The model's `forward` function is not
automatically the correct integration point.

Finally, run the BT11 service measurement. Report peak process memory, request
latency, entity and relation output, and the exact MLX/MCI/model versions. This
is the product acceptance gate; it must not be presented as the upstream
mechanism test.

### Gate 6: future automatic policy

Automatic policy is a separate project. It would need data across:

- consumer Blackwell devices with different memory sizes;
- datacenter Blackwell, Hopper, and Ampere;
- short and long encoders;
- decoder prefill and token decode;
- different dtypes and quantization formats;
- compiled and non-compiled primitives;
- concurrent streams and mixed workloads.

A future policy should use runtime facts such as shape, primitive count,
estimated transient size, allocator pressure, and phase information supplied
by the application. It should not use only compute capability or an
encoder/decoder label.

## 10. Risks and open questions

1. **Synthetic representativeness:** A simple public-operation chain may not
   reproduce the downstream graph topology. A negative result reopens the
   diagnosis; it must not be hidden by proceeding directly to the API.
2. **Which threshold matters?** The current downstream experiment cannot
   attribute the improvement until the MLX-native controls and four-cell
   ablation are complete.
3. **Submission overhead:** A smaller operation limit can increase CPU launch
   work, synchronization bookkeeping, and CUDA Graph cache entries for other
   workloads.
4. **Approximate memory score:** The current element-based score is
   dtype-dependent and backend-dependent. A public name must not promise exact
   bytes.
5. **Compiled primitives:** One compiled primitive can contain more work than
   one ordinary primitive. Operation count is not a universal cost model.
6. **Asynchronous overlap:** Smaller submissions can improve reclamation but
   can also change overlap. Single-request latency is not enough to evaluate
   throughput under concurrency.
7. **Implicit evaluation:** `array::item`, printing, and other APIs can trigger
   evaluation. An explicit per-`eval` option requires the application to
   materialize the selected roots before such implicit reads.
8. **Graph cache:** More split shapes can increase cache pressure even when
   transient allocator memory decreases.
9. **Multiple devices:** The existing process-static environment resolution is
   not naturally device-local. The scoped design must use the actual encoder's
   fallback defaults for every touched device.
10. **Benchmark versus CI:** The mechanism benchmark can expose performance and
    allocator behavior, but hardware-dependent memory ratios do not make
    stable CI assertions. Correctness and scope need separate deterministic
    tests.

## 11. Current decision record

- Keep W4A16 as the current model-format direction; do not treat quantization
  as the command-submission fix.
- Do not implement a custom fused SDPA kernel for this memory symptom.
- Keep cuDNN SDPA enabled in the main CUDA comparison.
- Treat `20/100` as a validated local expert setting, not a new MLX default.
- Use the process environment setting only for dedicated single-policy workers.
- Use an MLX-native public-operation benchmark as the primary mechanism
  evidence.
- Treat the mmBERT/MCI probe as external real-model acceptance and BT11 as
  product acceptance.
- Do not implement an opt-in per-evaluation interface until the MLX-native
  benchmark reproduces the mechanism and isolates the required field set.
- If that entry criterion passes, design the interface for mixed-model or
  mixed-phase processes.
- Do not branch MLX core behavior on model family.
- Run the four-cell ablation first on the MLX-native combined workload, then
  confirm the selected behavior with mmBERT.
- Do not put an absolute CUDA memory or latency threshold in portable unit
  tests.
- Keep automatic default selection and memory-accounting cleanup out of the
  first upstream change.

## 12. Completion criteria for the first phase

The first phase is complete when all of the following are true:

- an MLX-native benchmark made only from public MLX operations reproduces a
  significant, repeatable limit-dependent retention effect;
- its operation-bound and memory-score-bound modes isolate the intended
  controls, and the four-cell combined run identifies the required field set;
- the benchmark builds and runs without downstream repositories, model files,
  services, NVML, or profiler dependencies;
- checksums match across limit settings and timing is recorded separately from
  allocator memory;
- if the MLX-native workloads do not reproduce the effect, the phase ends with
  the general diagnosis reopened and no scoped API implementation;
- the downstream parser reports the same numeric peaks as its raw CSV files,
  and the mmBERT cases are rerun with identical settings and fresh processes;
- the candidate API, if still justified, has a precise evaluation scope,
  documented precedence, and a reviewed field set;
- deterministic MLX tests prove correctness, option lifetime, default
  preservation, and no policy leakage without asserting hardware-specific
  memory values;
- the external same-process mmBERT/Qwen isolation test passes;
- BT11 product memory, latency, and semantic output are recorded separately
  from the MLX mechanism evidence;
- the design is reviewed before any source implementation begins.
