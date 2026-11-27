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

## v0.2.0: certified decisions after two false-PASS reports (2026-11-19)

Post-release review reproduced two defect classes on the v0.1.0 verifier. A compensated float64 rank sum hid cancellation: with adapter column values `[2**120, 2**60, 1, -2**60, -2**120, 2]`, base 0, merged 2 and declared delta 3, the tool reported PASS at block sizes 1, 2 and 128 although the exact residual is 1. A rounded float64 subtraction hid an excess: base 1, merged `2**60`, delta `2**60`, `atol=0.5, rtol=0` reported PASS although the exact residual is 1. Both are reproduced by tests that fail on the old decision path.

The v0.2.0 decision path no longer uses the compensated sum. Every element is enclosed with outward-rounded binary64 intervals on the stored FP32 operands and the exact parsed alpha; only a decidable inequality becomes MATCH or MISMATCH. Undecided elements are re-evaluated with `Fraction` exact arithmetic in fixed row-major order under one shared `--max-exact-terms` budget (default 100000; 0 disables). Budget exhaustion is INDETERMINATE. PASS also requires at least one element certifying that the unchanged base is outside tolerance. The compensated reduction remains only as a labeled diagnostic in `max_abs_error`.

Owner-run measurements, Windows x64 / Python 3.12.3 / NumPy 2.4.6 / safetensors 0.7.0:

- Regressions: cancellation wrong merged=2 is FAIL at block 1/2/128 with exact residual 1; merged=3 is PASS; the subtraction case is FAIL with `max_abs_error` 1; `max_exact_terms=0` gives INDETERMINATE with one unresolved element and zero spent terms; an exact threshold equality (`residual == atol`) resolves through the oracle and passes.
- `PYTHONPATH=src python -m unittest discover -s tests -v` ran 34 tests and passed, including the optional PEFT test. The same 34 tests passed from a wheel installed into an external temporary target outside the checkout; the 0.2.0 wheel SHA-256 is recorded in [verification-0.2.0.json](verification-0.2.0.json), and every packaged module byte-matches the source tree.
- `python scripts/check_peft_integration.py` (required gate) passed with PEFT 0.19.1 and torch 2.12.0+cpu at ranks 1/4/8: the genuine merge PASS, a doubled alpha FAIL and the unchanged base FAIL; maximum true updates were 0.0123, 0.0244 and 0.0551 with no unresolved elements. Only this CPU PEFT path is measured.
- Five isolated mutations in a temporary copy were each caught by a named test: ignoring an exact mismatch, dropping the unresolved count, restoring the rounded subtraction, and removing each direction of outward rounding. Worktree source hashes were unchanged.
- Browser review of an actual CLI report at 1200 and emulated 360 CSS pixels showed no page-level horizontal overflow (360/360), an internally scrolling table (646 px content in 332 px), no external resource entries, and an embedded JSON block that reparses with the new certification policy. Screenshot files could not be written because the browser tool's workspace roots exclude the worktree; the inline full-page captures were inspected instead.

This remains a conservative, slower reference: intervals widen with cancellation, the exact oracle is bounded per call, and `--max-working-bytes` is an estimate, not a process-RSS cap. No claim is made for other PEFT versions, other merge backends, FP16/BF16 or GPU execution.

Task 4 integration review (2026-11-27) approved the interval/exact decision path after rerunning all 34 tests from source, an externally installed wheel, and the extracted sdist, plus the PEFT gate in all three contexts. Review also found that the initial sdist omitted `tests/peft_case.py` and `scripts/check_peft_integration.py`; `MANIFEST.in` now includes both, and the rebuilt sdist passed the same 34 tests and PEFT gate. The reviewed wheel SHA-256 is `fa474afca1fa6822f48b7ee13a01a7ec7fa12192b50caba54d050651a0191f00`; all eight packaged Python modules byte-match source. Isolated builds initially failed because the configured package mirror could not supply setuptools, so final builds used `python -m build --no-isolation` with the already-installed backend.
