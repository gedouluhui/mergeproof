# MergeProof

[![CI](https://github.com/gedouluhui/mergeproof/actions/workflows/ci.yml/badge.svg)](https://github.com/gedouluhui/mergeproof/actions/workflows/ci.yml)

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

For each tile, the reference computes `delta = (alpha/r) * (B @ A)` using a **fixed rank-order, Neumaier-compensated float64 reduction**, then checks. It does not delegate the rank reduction to shape-dependent BLAS kernels; changing output tile size does not change the reduction order:

```
abs((merged - base) - delta) <= atol + rtol * abs(delta)
```

Default `atol=1e-6`, `rtol=1e-5`. The relative term scales with the **update**, not the base weight: a huge base must not conceal an absent update. This is an explicit reference policy, **not** a universal rounding-error bound for all merge backends. Ordinary FP32 rounding can exceed a strict update-relative policy; inspect the witness and choose a justified tolerance rather than blindly relaxing it.

| Status / exit | Meaning |
|---|---|
| PASS / 0 | Complete input coverage agrees with the declared numerical contract. |
| FAIL / 1 | A mismatch, changed non-target or nonfinite value was found. |
| ERROR / 2 | Invalid/incomplete input, budget refusal or execution error. |
| INDETERMINATE / 3 | Even the unchanged base satisfies the update tolerance; execution of the merge is not distinguishable. |
| UNSUPPORTED / 4 | Unsupported variant or dtype. |

Reports contain SHA-256 input fingerprints, per-tensor residual witnesses, coverage and the selected policy. They do not embed checkpoint arrays. A fingerprint identifies bytes; it does not authenticate their origin.

## Deliberate limits

- v0.1: **FP32, single-file safetensors, standard nonquantized Linear LoRA only**. No sharded indexes, FP16/BF16, DoRA, rsLoRA, quantization, bias updates, extra trained modules or modified-base initialization semantics.
- This checks an explicit equation. It does not establish which base was used in training, prove model behavior/quality or detect backdoors. `Proof` is a project name, not a formal-verification claim.
- Inputs must stay unchanged during a run. Hashes are checked again at the end. This is not an adversarial concurrent-filesystem sandbox.
- `--block-size` tiles both matrix dimensions; `--max-working-bytes` (minimum 4096) bounds the conservative estimate of tile arrays and sets streaming/hash buffer sizes. It does **not** cap process RSS, Python/library overhead, metadata or OS file cache. Header size is separately limited. Fixed-order compensated reduction can be slower than BLAS; no throughput advantage is claimed.
- Non-target bytes must match exactly, including signed-zero representations. Unsupported and incomplete work does not silently pass.
- HTML/JSON may expose tensor names and filenames. Review before sharing.

## Why another tool?

PEFT's [`safe_merge`](https://huggingface.co/docs/peft/en/package_reference/tuners) checks for NaNs during merging. MergeProof instead independently compares three already-produced artifacts under an explicit contract. It complements merge tools; it is not a new merge algorithm. Real PEFT/backend interoperability beyond this contract is not yet certified.

## Development

```sh
python -m pip install .
python -m unittest discover -s tests -v
python -m pip install build
python -m build --wheel
```

Tests cover hand-computed matrices, incorrect scale, swapped same-shaped layers, missing/unused keys, non-target changes, nonfinite values, tile invariance, zero-update ambiguity, CLI exit codes, safe outputs and HTML escaping. See [implementation evidence](docs/IMPLEMENTATION.md). Real-world user adoption and broad checkpoint benchmarks have not been measured.

Bug reports are most useful with a **tiny self-generated** safetensors reproduction and manifest. Do not upload private model weights. AI-assisted implementation; review the code and contract rather than treating authorship or test counts as a correctness guarantee.

MIT. Uses NumPy (BSD-3-Clause) and safetensors (Apache-2.0); their licenses remain applicable.
