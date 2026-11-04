# DeepSeek V4-Flash batch-one DSpark request pipeline

This deployment retains five draft tokens and six target verification rows.
It complements the [low-latency target model](../deepseek_v4_flash_lowlat/index.md)
and the [wide-batch DSpark model](../deepseek_v4_flash_dspark/index.md).

The initial source consists of the dependency closure of the batch-one drafter,
Markov sampling, device request-state transitions, and the two-frame host policy.
The complete target composition, portable checkpoint/input loader and benchmark
entry are still being integrated. This directory is not yet a runnable serving
or TPOT reproduction entry. No historical TPOT is claimed for this revision.

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
