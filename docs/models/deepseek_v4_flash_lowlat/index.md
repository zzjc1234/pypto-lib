# DeepSeek V4-Flash low-latency decode

This model directory provides the batch-one decode path with eight logical
ranks. The default verification width is eight tokens. It is an experimental
decode implementation, not a `pypto-serving` deployment contract.

## Precision contract

The native HC stream uses FP32 at `hc_pre` input and `hc_post` residual/output,
including the public `pre_hc_hidden_out` output. Dense, attention and expert
local tensors also use BF16; this is a mixed-precision contract.

The independent Torch references preserve BF16 Cube K16 and FP32 Cube K4
accumulation, ascending split-K reductions and per-group projection scales. Float64
intermediates model each product group's FP32 rounding; model operands
and outputs keep their declared dtypes. This does not establish universal
bitwise equivalence or replace the existing numerical comparisons.

Standalone Q validation checks its produced head group; the other pure-Out
heads have no initialized-value guarantee.

The FP32 gate uses deferred RMS: `xg = x_mixed.float() * norm_w.float()`.
It applies `inv_rms` after projection and includes it in the dequantization
scale for INT8 `quant(xg)`, without BF16-rounding a normalized routing vector.
Gate activation quantization refines the FP32 quotient with a product-error
residual before round-to-nearest-even conversion. A power-of-two scaling keeps
the compensation intermediates bounded for large finite denominators; it does
not change the INT8 weights, FP32 dequantization scales or deferred RMS policy.
This does not assert all-domain IEEE or nonfinite-input equivalence.

The gate uses scope-local FP32 compensation primitives in `fp32_numerics.py`;
these helpers create no tasks or global-memory buffers. The gate's softplus
evaluates `log(1 + exp(-abs(x)))` before adding the
nonnegative part of the input. Its logarithm correction consumes the
unrounded two-part sum. The square-root correction applies within
`[2^-40, 2^40]` and retains the original square-root result outside that
interval. Owning Torch references round double-precision unary results to
FP32 at the specified boundaries and preserve the explicit reduction order.
Finite-case validation does not establish correctly rounded results for
every FP32 input or replace complete-forward validation.

The attention and final RMSNorm references reduce FP32 squared inputs in
FP64, then round the sum once to FP32. The mean and epsilon addition remain
FP32; the inverse square root is evaluated in FP64 and rounded to FP32.
This specifies the reference's statistical precision without changing epsilon,
activation dtypes, or the numerical comparison thresholds.

The fast HC, RMSNorm, QKV and attention paths use the device's FP32 reduction,
reciprocal, and exponential operations without the reference branch's full
product-error and transcendental compensation. Their results can differ from
the Torch reference and must be checked with the output-specific golden gates.
Finite input tests do not establish all-domain correctly rounded arithmetic or
a complete-model performance gain.

A previous HC-pre component comparison used three captured layer inputs at
six-token width on four dual-chip 910C boards. Its ordinary submit-to-result
median was 0.898 ms for the older arithmetic and 1.034 ms for the compensated
path. A separate instrumented capture measured the Sinkhorn kernel at 13.93
and 123.21 microseconds, respectively. This supports an implementation-cost
finding for that component and input scope; it is not a measurement of this
commit's complete-request TPOT.

Public stored LM-head logits are FP32. Community vLLM can select
`hf_overrides={"head_dtype": "float32"}`; this alone does not align the forward.

The observed vLLM-Ascend 0.30 reference uses BF16 HC streams across 43 layers;
its FP32 router input widens already-rounded low-precision values. Its EP8
expert quantization groups also differ from native TP8 with intermediate
shards of 256. These are reference-policy boundaries; they establish neither
a native implementation defect nor the cause of observed token differences.

A complete golden must align weights/scales, input state, HC, quantization,
routing, reduction and logit precision. Matching dtype labels or W8A8
checkpoint names, source/component CI and local type checks do not establish
complete-model numerical correctness. The existing comparison gates apply.
The current FP32 storage policy is retained. A community-precision
compatibility mode is not implemented.

## Full-forward numerical checks

The independent Torch forward remains the reference. Each owner and token row
of `pre_hc_hidden_out`, `hidden_out`, and active `logits` is checked with the
repository's `ratio_reldiff` comparator. The per-point threshold is 0.02; at
most 5% of pre-HC values and 10% of hidden or logit values may exceed it, with
a worst-point relative-difference cap of 1. These thresholds follow the
existing DeepSeek MTP forward policy. Inactive logit rows retain the previous
strict `rtol=atol=1e-3` check, and nonfinite values fail.

Sampled IDs must match sampling from the actual logits. If an ID differs from
the Torch reference, the selected and reference-token logits are also checked
at `rtol=atol=0.02`, and the mismatch is reported. This permits different IDs
when close logits change their ordering; it does not assert that generated
sequences or downstream task quality are identical. Cache and control outputs
retain their existing checks. Complete-forward device results are still
required before using these bounds as evidence of model correctness.

## Reproducible reduction order

The output projection publishes each rank's FP32 contribution into a separate
slot. Each of the two alternating lanes contains `N_RANKS * T_PAD` rows.
After the existing all-peer completion barrier, the consumer adds ranks
`0..N_RANKS-1` to a positive-zero accumulator and rounds to BF16 once.
Every producer overwrites its valid rows before publication; allocator contents
and arrival order do not contribute to the sum. Alternating lanes retain the
previous call's inputs until all peers have advanced beyond their consumers.

The HC head similarly stores each split-K projection in its own FP32 slot.
An ordered reduction adds the splits before the original gate, residual mix,
and BF16 conversion. The split-local matrix multiplication and RMS arithmetic
retain their original order.

HC-pre Sinkhorn column sums use the explicit four-row pairing
`(row0 + row1) + (row2 + row3)` in both implementations.

The fast SWA, HCA, and CSA softmax paths use the device exponential and row
reduction operations. Their reduction order and rounding can differ from the
independent Torch reference. The output-specific numerical checks cover that
difference rather than assuming bitwise equality.

These rules make the reduction order explicit. They do not establish bitwise
equivalence across different hardware, compiler revisions, or other operations
that use different arithmetic order.

## Embedding window reuse

The embedding exchange clears its local notification counters after
readback. Its current callers use a fixed notify count of one. A retained
single window must be reused only after all ranks finish the previous
invocation; overlapping reuse is outside this contract.

## Complete-set Indexer ordering

When the number of visible compressed positions is no greater than `IDX_TOPK`,
all visible positions are selected in ascending position order, including the
original index offset. This gives single-query and grouped-query consumers the
same attention accumulation order for the complete set. Score computation and
cache updates are unchanged.

For more than `IDX_TOPK` visible positions, the original score-based selection
and tie handling remain in place. The validator requires exact positional
indices for complete sets and retains the score-aware comparator for selected
subsets. The boundary between these cases must be included when validating a
new request layout.

## Retired expert experiments

The model uses the planned persistent routed-expert kernel. The shared expert
occupies its own entry in that weight bank rather than a separate FFN chain.
Its fixture weight generator remains in `utils.py`.

An earlier scatter epilogue replaced the per-slot BF16 output and local gather
with FP32 atomic accumulation into token rows. The recorded A2/A3 TP8, T8
comparison used 20 rounds after three warmups and the fastest-rank mean:
244.5 us for the retained MoE path versus 269.0 us for scatter, across three
pairs. The standalone kernels measured 129.7 versus 136.8 us. These are
historical component measurements, not request TPOT or an isolated estimate
of scatter cost: task boundaries, initialization, stores, and rounding also
changed. The mechanism's behavior at larger token or route counts remains
unproven. The retired scatter and unused persistent A/B variants are omitted
from the runnable model; re-evaluate with real consumer inputs if that scope
changes.

## Execution shapes and component ownership

`config.py` holds model configuration and quantization constants.
`decode_shapes.py` owns the shared deployment shapes and physical cache pools;
context length, parallel rank count and pool scale must be positive.
validation start-position defaults live with the host utilities.

The attention-component part supplies `rope_interleave.py` and
`o_proj_grouped.py` together with their first Indexer/compressor and standalone
attention consumers. The former prepares query-token or compression-boundary RoPE tables;
QKV metadata preparation separately handles dynamic token tiles and its swap
indices.

## CI targets and validation scope

This model currently uses A2/A3 device CI. Its runnable entries are marked
`# ci: no-sim`, which both pull-request and daily selectors honor. Simulator
qualification is deferred; this does not assert simulator incompatibility.
Other model families and their simulator coverage are unchanged.

The composed `decode_swa.py` and `decode_hca.py` entries require eight logical
TP ranks and declare `# ci: devices=8`, as does CSA. Their standalone
`decode_sparse_attn_swa.py` and `decode_sparse_attn_hca.py` entries use one
device and retain their original numerical references.

`decode_fwd.py` validates all 43 layers against an independent Torch
composition by default. It includes embedding, request-derived metadata,
2 SWA / 21 CSA / 20 HCA attention layers, 43 TP MoE layers, HC head,
final RMSNorm, vocabulary-sharded logits and sampling. All eight mutable
cache/state pools and the four public outputs are compared by the existing
harness. Source compilation and component CI do not qualify this full path.

The Torch composition reuses the child operator references. Expert gate/up/down
products accumulate INT8 operands as INT32 before applying original FP32
scales. Activation quantization uses explicit FP32 division for `127 / amax`;
the gate retains deferred RMS and its separate dequantization-scale order.
BF16 Cube consumers use K16 FP32 accumulation and their kernel's split-K
order in the CPU reference: Q/KV projections, sparse QK/PV, grouped output
projection A, compressor projections, indexer Hadamard and routing-weight
projections, and vocabulary logits. Batched rows and independent groups use
the same host helper; indexer routing weights retain four split-K partials.
Float64 evaluates each BF16 product group and each reference unary or division
before rounding its result to FP32. Model operands, outputs, INT8-to-INT32 dots
and scales retain their declared dtypes. HC-pre uses explicit balanced FP32
reductions for its square sums and four-way Sinkhorn sums; its Torch reference
spells out the stored exp/add/div boundaries instead of using a fused softmax.
The native HC-pre maps compensate their reciprocal, reciprocal-square-root and
division seeds. Exponential correction is bounded to `[-80, 80]`; the original
exponential is retained outside that interval. These bounded corrections do not
establish a general IEEE rounding guarantee for every input.
TP projection contributions include channel scales before ordered rank
reduction. Standalone grouped projections retain their own arithmetic.
These choices do not establish bitwise equivalence with device instructions;
the existing numerical comparisons remain required.

- Default execution and `--full` run the same complete-forward comparison.
  The generated bounded fixture starts at position 121 and uses one shared
  MoE bank unless `--moe-banks` is supplied. This is synthetic correctness
  coverage, not qualification of a real checkpoint or community equivalence.
- A real checkpoint comparison requires the actual weights and original FP32
  scales for all 43 layers (`--moe-banks 43`), matching inputs/state and the
  precision contract above. Shared random banks cannot stand in for them.
- `--input-data` replays complete `in/` and `out/` tensors, which take precedence
  over the locally computed Torch reference. Partial numerical snapshots fail.
- `--smoke` explicitly executes all 43 layers without numerical comparison.
  It permits generated inputs or input-only replay and reports no correctness
  verdict. It cannot be combined with saved expected outputs.
- `--compile-only` performs code generation and ABI checks without computing
  a Torch golden or executing the model.

The full TP reference checks that shared attention operands, metadata and
initial caches are replicated; mismatches are rejected explicitly. All three
attention kinds consume the actual projection shards; unused full-projection
arguments do not affect the reference. Every owner's MoE valid-row count is
consumed; counts may differ across owners and must lie in `[0, T]`. The standalone MoE reference
also computes each rank's HC/routing operands separately.

A full-model Torch comparison establishes agreement with the implemented
native precision contract for its supplied fixture. An independent community
snapshot additionally requires alignment of the community numerical policy;
matching INT8 checkpoint names or tensor dtype labels is insufficient.
Existing operator/layer validators retain their own scope. No complete-model
numerical PASS is implied by providing this reference or by host-only checks.

Use `--seed` for reproducible generated inputs and `--save-data` to save both
inputs and expected outputs. For example:

```bash
# Complete 43-layer numerical fixture; 16384 is a supported Indexer context.
python models/deepseek_v4_flash_lowlat/decode_fwd.py -p a2a3 --max-seq-len 16384
# Generate the complete program without computing a Torch reference.
python models/deepseek_v4_flash_lowlat/decode_fwd.py -p a2a3 --max-seq-len 16384 --compile-only
# Explicit execution-only smoke, without a correctness verdict.
python models/deepseek_v4_flash_lowlat/decode_fwd.py -p a2a3 --max-seq-len 16384 --smoke
# Replay a complete snapshot with matching checkpoint and execution layout.
python models/deepseek_v4_flash_lowlat/decode_fwd.py -p a2a3 --max-seq-len 16384 --moe-banks 43 --input-data <reference>/data
```

## Validation

HC pre and HC post use a dynamic token axis and bound their final tiles to the
remaining rows. Their default entries check the configured width plus 1, 6, 9,
and 17 rows. `--tokens` selects one width. HC pre saved-golden replay defaults
to the configured width unless `--tokens` is supplied.

The component part's grouped projection uses a positive static token count
divisible by 8 and at most 16. QKV projection requires
the attention head count to divide into head groups, with each local head
count divisible by its four-head RoPE tile. Invalid configurations fail
before compilation.

The output-projection L2 eviction option is diagnostic. Its large buffer is
allocated only with `--evict-l2`; ordinary correctness runs retain a small
placeholder and perform no eviction reads.

Use the operator entry points and their original Torch references:

```bash
python models/deepseek_v4_flash_lowlat/hc_head.py --help
python models/deepseek_v4_flash_lowlat/o_proj_tp.py --help
```

`o_proj_tp.py` uses the full eight-rank communication contract. Its default
correctness entry runs two distinct inputs at epochs 1 and 2 in one dispatch,
checks both outputs against the original reference, and clears signals only
after the second epoch. `--single-epoch` retains the original one-epoch entry
and saved-fixture interface. The validation handoff is not a performance
measurement. Local operators
retain deployment shapes when run on one device. Source compilation is only a
preparation step: numerical checks and performance comparisons must use the
resulting binaries, original consumers, and complete relevant request paths.

See [precision tuning](../../debug-and-tune/precision-tuning.md) and the
[compile/runtime workflow](../../run-and-validate/compile-runtime-workflow.md)
for validation and artifact handling.

### Ratio-4 verification-step boundaries

The main and indexer compressors pool every boundary with a nonnegative compressed-cache slot mapping. `kv[b, o]` contains boundary ordinal `o` within the step: row zero remains the first boundary. Unmapped ordinals and remaining rows have no output write contract. With eight tokens, starts 0, 4 and 121 produce boundaries 3/7, 7/11 and 123/127 respectively when both cache slots are admitted.

Pooling consumes the current step's projected rows directly, including speculative rows without persistent state slots, and reads earlier prefix rows through the state block table. All pools precede persistent scatter; only nonnegative state slot mappings authorize that scatter. An early advance recomputes remaining tokens from the new step inputs and the committed prefix, without promoting rejected tokens into persistent state.

Indexer query RoPE tables have one row per absolute query-token position. Compression RoPE tables are separate, indexed by boundary ordinal and aligned to each four-token window start. For a step starting at 8193, the first boundary 8195 uses compression position 8192; its query still uses its own absolute position. Standalone half-width tables are flattened `[B*S, rope_dim/2]`; the indexer additionally takes `inner_cos` and `inner_sin` for aligned compression windows. CSA derives both tables from its existing full RoPE inputs.

## Batch-one DSpark request serving


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

The compact single-query entry is `native_single.py`; request cycles use
`native_decode.py`, with the ordered collective composition in
`native_rsag_decode.py`. They share this model directory's operator files and
43 layer-specific weight banks. Each build freezes its query width in a fresh
process. Request benchmarks use actual checkpoint weights.

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
python -m models.deepseek_v4_flash_lowlat.build \
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

## Fast arithmetic delivery

This branch restores the pre-alignment hardware arithmetic in target HC,
RMS, QKV and sparse attention. It keeps W8A8 tensors, the existing public
interfaces, request scheduling, ordered collectives, buffer padding and
ownership checks. Golden functions remain enabled. The full-forward floating
outputs use the bounded checks described above; other component gates are
unchanged.

Fast arithmetic does not guarantee bitwise agreement with the refined
reference arithmetic. FP32 reductions and transcendental approximations can
change greedy token choices and EOS positions. Model-quality scores and
whole-request speed for this port must be evaluated separately; a historical
measurement is not a measurement of the current commit.

The validated reference-arithmetic source is preserved on
[the reference branch](https://github.com/zzjc1234/pypto-lib/tree/feat/dsv4-dspark-bs1-reference).
