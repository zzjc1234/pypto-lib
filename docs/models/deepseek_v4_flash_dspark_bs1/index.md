# DeepSeek V4-Flash batch-one DSpark request pipeline

This deployment retains five draft tokens and six target verification rows.
It complements the [low-latency target model](../deepseek_v4_flash_lowlat/index.md)
and the [wide-batch DSpark model](../deepseek_v4_flash_dspark/index.md).

The source includes the batch-one drafter, Markov sampling, six-row target,
device request-state transitions, two-frame host policy, native build and
checkpoint-backed request benchmark. Clean-checkout execution and performance
qualification are still pending. No historical TPOT is claimed for this revision.

The target retains rank-ordered output projection, ordered HC-head split
reduction, and ascending complete-set Indexer selection. Its normalization,
routing and expert activation use compensated FP32 arithmetic at the existing
quantization boundaries. Single-query and verification entries share these
operators. Full request numerical and performance qualification remain separate
from component validation; failures remain visible in either stage.

## Component boundary

- `drafter/entry.py` exports the three-layer drafter with TP4/CP4/EP4 geometry,
  a 128-row target-history window, and five query rows. Only the four drafter
  ranks execute it; they are a subset of the eight target ranks.
- `markov/entry.py` exports the target LM-head and sequential five-step Markov
  proposal/confidence path with four vocabulary shards.
- `device_metadata.py` selects the actual absolute-position metadata bank from
  committed state. A communication epoch is not a metadata-bank index.
- `bridge_input.py` broadcasts the seed token and five proposals with an epoch
  tag before the target can consume them.
- `device_commit.py` accepts the longest greedy-matching draft prefix and emits
  the target bonus. The original control fields distinguish logical advance
  from the number of tokens still permitted by the output budget.
- `device_control.py` publishes the next state only after the real history
  commit acknowledgement. `phase_reuse.py` supplies the peer-only eight-rank
  boundary for reuse of communication slots within a two-cycle parent.

The first three request-state fields are the monotonic epoch, committed anchor,
and remaining output budget. Field three stores the last delivered token.
History has 128 live rows in a 136-row owning allocation. Target features are
three concatenated 4096-wide HC means from layers 40, 41 and 42, not the final
learned HC head.

`native_pipeline.native_cycles` composes D/M on ranks 0–3 and target verification
on all eight ranks. `native_rsag_pipeline` selects the ordered MoE
reduce-scatter/allgather composition. Its target package shares unchanged kernels
with the baseline package. The `cycle_count` constexpr selects a one- or two-cycle
parent in source; it does not require editing generated host code.

The single-query reference uses `target_single`. All three target contexts use
the same 43 layer-specific weight banks, with independently specialized sequence
geometry. None of these entries runs with random weights as a request benchmark.

The drafter and Markov packages isolate their import-time deployment arguments
and restore the caller's arguments in `finally`. Their kernel bodies and active
constants are carried forward from the batch-one implementation; unused
standalone fixtures and prefill modules are omitted. These packages have not
been substituted with the wide-batch deployment defaults.

## Streaming and termination

`stream_controller.run_request` uses two independently owned output frames and
at most two accepted parent handles. A frame is reusable only after successful
whole-parent completion, packet validation, delivery or EOS discard, and release.
It never performs a device read or frees an in-flight frame itself.

A single-cycle parent is used when fewer than seven tokens remain. A second
two-cycle parent can be accepted ahead only when at least 24 tokens remain,
so its reservation survives the predecessor's maximum twelve-token advance.
This controls preparation ahead of execution; safe execution also requires the
real runtime launch-depth-one parent retirement boundary and per-run HOST output
publication. FIFO order alone is insufficient.

The controller supports a maximum of 2048 generated tokens including the first
prefill prediction. It stops delivery at a real tokenizer EOS. An already
accepted successor is still waited on, captured and validated, then discarded;
no replacement is submitted. Exceptions remain the caller's responsibility to
persist and drain during normal Worker cleanup.

## Source generation

From the repository root, in the configured PyPTO environment:

```bash
python -m models.deepseek_v4_flash_dspark_bs1.build \
  --profile rsag --devices 0,1,2,3,4,5,6,7 \
  --cache build_output/dspark-bs1-cache \
  --report build_output/dspark-bs1-rsag-generated.json
```

The other profiles are `single`, `one`, and `two`. This command creates generated
artifacts only: it does not load weights, construct a Worker, allocate NPU memory,
or measure performance. The eight device IDs describe logical compile topology;
runtime board/chip mapping and capacity admission are separate.
Each PTOAS subprocess has a default 180-second deadline for the compensated
vector functions. `--ptoas-timeout-seconds` accepts a value from 1 to 600 and
records it in the generated report. This scoped build policy preserves the
backend's assembler command, version checks and errors, and restores its
binding on exit; it does not modify the installed toolchain.

The native W2 prefetch implementation is a tracked C++ Cube-body fragment.
`native_source` selects it before publication only after the complete generated
Cube, shared declarations, ABI and exports match the reviewed fingerprint. The
freshly generated Vector arithmetic is preserved. Both original and selected
sources remain in the artifact report, and cache hits replay the same graft
validation. The fragment and selector digest is a JIT constexpr key input, and
the fragment is included in the model source manifest. Normal native compilation
is still required; no retained binary is substituted for changed code.

## Native compilation and prepared data

Compile each generated profile without a Worker or device allocation:

```bash
python -m tools.dspark_bs1.native_package \
  --generated-report build_output/dspark-bs1-rsag-generated.json \
  --output build_output/dspark-bs1-rsag-native --jobs 32
```

Repeat for `single`, `one` and `two`. The explicit native package includes its
source and file inventory; it does not impersonate a persistent JIT READY entry.
The adapter currently uses the PyPTO prebuilt/assembly APIs and must be used with
the compatible installed PyPTO/simpler toolchain. Request execution loads actual
native bytes read-only and preserves normal Worker provenance validation.

`tools.dspark_bs1.prepare_inputs` imports the real ND target and four-rank D/M
raw-shard manifests. It reuses weight files in place, constructs only address,
slot and RoPE metadata, and initializes fresh target/D KV state. Its manifest
records the actual file paths, dtype/shape, source manifests and stat evidence.
It does not create model activations or continuation-token references.

The prepared tensor schema is explicit: `shape` contains the rank axis, `dtype`
is the actual ABI dtype, and `shards` supplies one raw or weights-only Torch
payload per rank. Raw offsets are in elements. NZ data must already have its
declared native storage order. This importer is for the existing prepared
checkpoint format; it does not claim arbitrary Hugging Face checkpoint conversion.

The request JSON contains eight distinct records with `dataset`, `question_id`,
`prompt_ids`, `prompt_tokens`, and untruncated first-turn provenance. Every request
runs to its actual tokenizer EOS or 2048 generated tokens. Checkpoint weights and
tokenized benchmark data are external inputs and are not committed as binaries.

## Complete-request benchmark

Run CPU admission before acquiring any device. Supply all four package paths:

```bash
python -m tools.dspark_bs1.benchmark --prepare \
  --package single=build_output/dspark-bs1-single-native \
  --package one=build_output/dspark-bs1-one-native \
  --package two=build_output/dspark-bs1-two-native \
  --package rsag=build_output/dspark-bs1-rsag-native \
  --tensor-manifest prepared/tensor-manifest.json \
  --output build_output/dspark-bs1-cpu
```

The `--run` command takes the same inputs plus `--cpu-ready`, `--lease-file`,
`--execution-lock`, and a fresh output directory. Run it through
`tools.dspark_bs1.owned` so its process marker predates all child creation.
Ownership covers the exact host, boot, PID/starttime and token. Resource waiting
allocates no HBM; real allocations begin only after the experiment is ready.
The launcher defaults `HCCL_NPU_SOCKET_PORT_RANGE` to `auto` to avoid a shared
NPU's fixed-port collision. An explicitly supplied value is preserved and the
effective setting is recorded in `execution.json`.

The tested 910C placement uses four complete physical boards with both chips,
eight logical ranks and symmetric SIO pairs verified from actual HAL/SMI/topology.
It does not substitute eight boards with one chip each. Capacity retains the
runtime and two-GiB margins before initialization and actual-free checks afterward.
The original two-frame depth-two/launch-depth-one adapter is scoped to construction.
Optional `--early-aic-control-map` and `--omit-optional-warmup` initialization
adapters are explicit and recorded; they do not change model math or enable SDMA
without its real workspace/runtime support.

The benchmark retains five warm parents, four serialized fixed captures, and
eight request triplets (`K0`, baseline streaming, ordered-RSAG streaming).
Every backend starts from the same request's live model-produced prefix.
The decoder clock includes native history seeding, state upload, packet delivery
and accepted-successor drain. Prefix execution, initialization, weight upload and
warmup are recorded separately. The prefill prediction is counted once in total
generation and excluded from the decode denominator.

Outputs include full token IDs and per-request JSON/CSV with prompt/decode counts,
TTFT scope, decode wall, TPOT, EOS/length reason and failures. Request-median TPOT
and token-weighted TPOT are distinct statistics. No stable P95 is inferred from
eight requests. Numerical failures remain visible and do not become qualified
speed claims; final Torch reconciliation remains a separate stage.

## Reproduction and numerical status

Performance reproduction and numerical qualification are separate. The planned
benchmark retains batch size one, concurrency one, real EOS or length termination,
initialization/upload/warmup accounting, per-request output IDs, and request-wall
TPOT. Kernel duration divided by a presumed acceptance count is not TPOT.

The merged target model's numerical checks do not establish the accuracy of
this composition. A later complete entry must retain the existing packet,
commit, history-ring, final-state and complete-output checks. Until that entry
is built and measured from a clean checkout, both full-pipeline reproduction and
numerical qualification remain pending.
