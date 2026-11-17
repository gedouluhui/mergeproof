# Implementation evidence

2026-11-18, owner-run local checks, Windows x64 / Python 3.12.3 / NumPy 2.4.6. This is an engineering record, not an external quality certification.

- Safetensors real write/read probe: a 2x3 FP32 array was read via `safe_open(..., framework='np').get_slice`; shape, dtype and a two-dimensional tile matched the written values.
- Core RED: `PYTHONPATH=src python -m unittest discover -s tests -v` initially ran 16 tests, with 23 failing assertions including subtests, because the public verify API was absent. The log remains local under ignored `artifacts/core-red.log`.
- Core GREEN: the same 16 tests passed after implementation.
- CLI/report RED: five CLI tests ran, four failed because the report renderer and module CLI were missing; duplicate-JSON rejection already passed.
- Combined GREEN: `PYTHONPATH=src python -m unittest discover -s tests -v` ran 21 tests and passed. These include actual CLI subprocesses and saved-report roundtrips.
- `PYTHONPATH=src python -m mergeproof demo --output artifacts/demo` produced actual `good=PASS`, `bad=FAIL` reports.

Numeric contract: float64 NumPy reference, delta-relative explicit tolerances, byte-identical non-target arrays, all input FP32. Tiling reduces array allocation but is not a process memory sandbox. Input files are hashed before and after verification. No model inference, real-world checkpoint benchmark, real PEFT merger comparison, non-Windows run or external user test has been claimed.

The initial implementation agents were unavailable, so the owner implemented the repository directly.

## Independent review and first fix round

The read-only reviewer found three concrete issues; the owner reproduced all three with new failing tests: changing output tile size changed cancellation-heavy rank reductions (FAIL/PASS/FAIL/PASS), a 144-byte budget still allowed fixed MiB-sized streaming buffers, and argparse errors returned empty stdout rather than structured JSON.

Fixes: rank-ordered Neumaier-compensated float64 reduction instead of shape-dependent BLAS reduction; minimum 4096-byte budget plus budget-derived hash/comparison buffer sizes; structured argparse error handling. The cancellation oracle uses scalar `math.fsum` on FP32-representable operands, whose correct sum is 14; wrong merged value 7 must fail at block sizes 1/2/7/128 and correct 14 must pass at each. This fixed reduction costs throughput relative to BLAS; no speed claim is made.

Owner ran the complete suite after these changes: 24 tests passed. Scope of the memory control remains array/streaming-buffer estimates, not Python-object overhead or process RSS. The initial wheel was built successfully; its deprecated license metadata was changed to modern SPDX. A fresh wheel build and visual/package verification remain separate steps.

Scoped re-review independently confirmed the three original fixes, including a 4096-byte configuration issuing at most 512-byte stream reads and detecting a changed byte near the end of a large non-target. It found one new issue: compensation temporaries and previous-tile arrays could exceed the old array estimate. The owner reproduced the 1024x512/rank-1, block-512, 25,190,400-byte case as a red test; revised the estimate to four float64 panel copies plus 32 cell-sized arrays and explicitly released each tile's operands. The same budget now rejects block 512 and accepts block 128; the full suite now runs 25 tests and passes. This is a conservative estimate, not a measured process-RSS cap. The final scoped reviewer independently confirmed this correction, including the old-budget/block-512 refusal, smaller-block success and bounded live-array observations; all original and follow-up findings were addressed.

## Installed-release and visual checks

The owner rebuilt the wheel, verified that every packaged Python module matched the current source bytes, installed it into a temporary target and ran all 25 tests from outside the source checkout. Existing NumPy/safetensors dependencies were reused; this was not a fresh-OS dependency installation. An isolated source mutation disabled mismatch handling and was caught by the named `test_wrong_alpha` assertion; original source hashes remained unchanged.

Chrome inspection of the installed CLI's actual report showed the expected residual and mismatch, no root horizontal overflow at 1200 and emulated 360 CSS pixels, functioning detail expansion and no external resource timing entries. Full-page screenshots were inspected inline; saving an image into the project was not permitted by the browser workspace setting, so no screenshot asset is claimed. This is not a complete accessibility or cross-browser audit.

Machine-readable package fingerprints, test counts, mutation and visual observations: [verification.json](verification.json).
