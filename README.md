# MergeProof

**Did your LoRA reach the right weights? Check the artifacts, not just the merge command.**

MergeProof compares a local base checkpoint, a standard Linear LoRA adapter and a merged checkpoint against an explicit tensor mapping. It checks every declared target and verifies that non-target tensor bytes did not change.

[中文快速说明](README.zh-CN.md)

## Try a real failure in one minute

From a checkout, in your Python environment:

```sh
python -m pip install .
mergeproof demo --output demo
```

Open `demo/bad.html`: the report identifies an adapter update accidentally applied at **twice** its declared scale. `demo/good.html` is the passing control. These results are computed from the generated safetensors files, not recorded answers. No GPU, model download, account or API key is needed. Choose a fresh output directory.

## Check your files

```sh
mergeproof verify --base base.safetensors --adapter adapter.safetensors \
  --merged merged.safetensors --manifest manifest.json \
  --json result.json --html result.html
```

The manifest is deliberately explicit:

```json
{
  "schema_version": 1,
  "variant": "standard_linear",
  "targets": [
    {"base": "model.layers.0.q_proj.weight", "a": "q.lora_A.weight", "b": "q.lora_B.weight", "alpha": 16}
  ]
}
```

Use the actual tensor keys in **your** files. Rank is derived from A; B must have compatible dimensions. All adapter keys must be used exactly once. Missing targets, extra adapter keys and ambiguous mappings are errors. There is no automatic PEFT-config conversion in this version.

```python
from mergeproof import verify
report = verify("base.safetensors", "adapter.safetensors", "merged.safetensors", "manifest.json")
assert report["status"] == "PASS", report["findings"]
```

## What the result means

The intended equation is evaluated on the **stored FP32 operands with exact real semantics**. A PASS requires a proof for every declared element under the tolerances you supply:

```
abs((merged - base) - (alpha/r) * (B @ A)) <= atol + rtol * abs((alpha/r) * (B @ A))
```

Each element is first enclosed with outward-rounded float64 intervals (`nextafter`); cancellation or a rounded subtraction cannot silently pass. Elements whose interval straddles the threshold are re-evaluated by an exact rational (`Fraction`) comparison in fixed row-major order under one shared budget, `--max-exact-terms` (default 100000 multiply-add terms; `0` disables the fallback). Budget exhaustion is `INDETERMINATE`, never `PASS`.

Default `atol=1e-6`, `rtol=1e-5`. The relative term scales with the **update**, not the base weight: a huge base must not conceal an absent update. This is a decision procedure for the declared policy, **not** a universal rounding-error bound for all merge backends. Ordinary FP32 rounding can exceed a strict update-relative policy; inspect the witness and choose a justified tolerance rather than blindly relaxing it.

The Neumaier-compensated float64 rank reduction is retained only as a labeled **diagnostic** for the residual column. It never decides a status. Per-target reports also carry `elements_exact`, `elements_certified`, `elements_unresolved`, `exact_terms_used` and `baseline_exclusion`, so coverage is auditable without trusting the headline.

| Status / exit | Meaning |
|---|---|
| PASS / 0 | Every element is certified matched and at least one certifies the unchanged base is outside tolerance. |
| FAIL / 1 | A certified mismatch, changed non-target or nonfinite value was found. |
| ERROR / 2 | Invalid/incomplete input, budget refusal or execution error. |
| INDETERMINATE / 3 | An element remained undecided after the exact budget, or no element distinguishes the merged file from the unchanged base. |
| UNSUPPORTED / 4 | Unsupported variant or dtype. |

Reports contain SHA-256 input fingerprints, per-element certification coverage, residual witnesses and the selected policy. They do not embed checkpoint arrays. A fingerprint identifies bytes; it does not authenticate their origin.

## Using it with PEFT

Build the three artifacts yourself and name their real tensor keys:

```python
from peft import get_peft_model_state_dict

base = {name: value.detach().clone() for name, value in model.state_dict().items()}
# ... wrap with get_peft_model and train ...
adapter = get_peft_model_state_dict(peft_model)              # keys for "a" and "b"
merged = peft_model.merge_and_unload(safe_merge=True).state_dict()
```

Then save each mapping as FP32 safetensors and write the manifest with `alpha` from your `LoraConfig`. The integration test (`tests/test_peft_integration.py`) and the required runner (`scripts/check_peft_integration.py`) exercise this workflow on a self-made CPU model at ranks 1, 4 and 8: the true merge passes, a doubled alpha fails and the unchanged base fails. Measured compatibility is PEFT 0.19.1 with torch 2.12.0+cpu; other versions and backends are untested.

## Deliberate limits

- v0.2: **FP32, single-file safetensors, standard nonquantized Linear LoRA only**. No sharded indexes, FP16/BF16, DoRA, rsLoRA, quantization, bias updates, extra trained modules or modified-base initialization semantics.
- This checks an explicit equation. It does not establish which base was used in training, prove model behavior/quality or detect backdoors. `Proof` is a project name, not a formal-verification claim.
- Inputs must stay unchanged during a run. Hashes are checked again at the end. This is not an adversarial concurrent-filesystem sandbox.
- `--block-size` tiles both matrix dimensions with fixed row-major traversal, so exact-fallback work does not depend on tile size. `--max-working-bytes` (minimum 4096) bounds the conservative estimate of tile and exact-buffer memory and sets streaming/hash buffer sizes. It does **not** cap process RSS, Python/library overhead, metadata or OS file cache. Header size is separately limited. The certified reference can be slower than BLAS; no throughput advantage is claimed.
- Non-target bytes must match exactly, including signed-zero representations. Unsupported and incomplete work does not silently pass.
- HTML/JSON may expose tensor names and filenames. Review before sharing.

## Why another tool?

PEFT's [`safe_merge`](https://huggingface.co/docs/peft/en/package_reference/tuners) checks for NaNs during merging. MergeProof instead independently compares three already-produced artifacts under an explicit contract, using interval certification plus a bounded exact oracle. It complements merge tools; it is not a new merge algorithm.

## Development

```sh
python -m pip install .
python -m unittest discover -s tests -v
python -m pip install ".[peft]"
python scripts/check_peft_integration.py   # required gate: missing deps exit nonzero
python -m pip install build
python -m build --wheel
```

Tests cover hand-computed matrices, incorrect scale, swapped same-shaped layers, missing/unused keys, non-target changes, nonfinite values, tile invariance, zero-update ambiguity, exact cancellation and subtraction regressions, exact-budget exhaustion, CLI exit codes, safe outputs, HTML escaping and a genuine PEFT merge at ranks 1/4/8. See [implementation evidence](docs/IMPLEMENTATION.md). Real-world user adoption and broad checkpoint benchmarks have not been measured.

Bug reports are most useful with a **tiny self-generated** safetensors reproduction and manifest. Do not upload private model weights. AI-assisted implementation; review the code and contract rather than treating authorship or test counts as a correctness guarantee.

MIT. Uses NumPy (BSD-3-Clause) and safetensors (Apache-2.0); their licenses remain applicable.
