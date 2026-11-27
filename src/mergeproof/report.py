"""An offline evidence sheet; no JavaScript or remote assets."""
from html import escape
import json


def to_html(report):
    e = lambda x: escape(str(x), quote=True)
    status = e(report['status'])
    rows = ''.join('<tr><td><code>' + e(t['tensor']) + '</code></td><td>' + e(t['status']) +
                   '</td><td>' + e(t['elements_checked']) + '</td><td>' + e(t.get('elements_exact')) +
                   '</td><td>' + e(t.get('elements_unresolved')) + '</td><td>' + e(t.get('max_abs_error')) +
                   '</td><td>' + e(t.get('witness_coordinate')) + '</td></tr>' for t in report['targets'])
    findings = ''.join('<article><b>' + e(f['rule']) + '</b><code>' + e(f['tensor']) +
                       '</code><p>' + e(f['detail']) + '</p></article>' for f in report['findings'])
    raw = e(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>MergeProof · Merge evidence</title><style>
*{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:#15243b;font:16px/1.6 system-ui,sans-serif}main{max-width:1100px;margin:auto;padding:48px 24px}header{border-bottom:2px solid #254d8a;padding-bottom:24px}.brand{font:600 14px ui-monospace,monospace;letter-spacing:.13em;color:#254d8a}h1{font:normal clamp(32px,6vw,58px)/1.15 Georgia,serif;margin:20px 0}h2{font-size:22px;margin-top:40px}.equation{padding:24px;border-left:4px solid #254d8a;background:#e7edf6;font:20px ui-monospace,monospace;overflow-wrap:anywhere}.status{display:inline-block;padding:6px 14px;background:#15243b;color:white;font-weight:650}.table{overflow-x:auto}table{width:100%;border-collapse:collapse;background:#fff}th,td{text-align:left;padding:12px;border-bottom:1px solid #d5deea}th{font-size:13px;color:#254d8a}code{overflow-wrap:anywhere;font:13px/1.6 ui-monospace,monospace}article{padding:18px 20px;background:white;border-left:3px solid #a0283c;margin:12px 0}article code{display:block;margin-top:8px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;background:#e7edf6;padding:20px}summary{cursor:pointer}summary:focus-visible{outline:3px solid #254d8a}footer{margin-top:40px;color:#4c5e78;font-size:13px}a{color:#254d8a}@media(max-width:600px){main{padding:24px 14px}.equation{font-size:15px;padding:16px}}
</style><main><header><div class="brand">MERGEPROOF / ARTIFACT VERIFICATION</div><h1>Did the adapter reach<br>the right weights?</h1><span class="status">''' + status + '''</span></header>
<h2>The contract</h2><div class="equation">W<sub>merged</sub> − W<sub>base</sub> ≈ (α / r) · B A</div><p>''' + e(report['scope']) + '''</p>
<h2>Tensor evidence</h2><div class="table"><table><thead><tr><th>Tensor</th><th>Result</th><th>Elements</th><th>Exact</th><th>Unresolved</th><th>Max residual evidence</th><th>Coordinate</th></tr></thead><tbody>''' + rows + '''</tbody></table></div><p>Status is certified from outward-rounded intervals and, for undecided elements, exact rational comparisons. The residual column is the exact value where the exact fallback ran; otherwise it is a conservative enclosure or a labeled diagnostic and never decides a status.</p>
<h2>Findings</h2>''' + (findings or '<p>No findings within the declared contract.</p>') + '''<details><summary>Input fingerprints, numeric policy and complete JSON</summary><pre>''' + raw + '''</pre></details><footer>Generated locally. No checkpoint weights are embedded. This report is not a model-quality or provenance certificate.</footer></main></html>'''
