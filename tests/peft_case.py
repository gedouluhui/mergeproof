"""Build a genuine PEFT LoRA merge fixture on CPU; optional dependencies only."""
import importlib.util
import json
from pathlib import Path

OPTIONAL = ('torch', 'peft')


def missing_dependencies():
    return [name for name in OPTIONAL if importlib.util.find_spec(name) is None]


def build_case(root, rank):
    """Save base snapshot, adapter and merged safetensors plus explicit manifests."""
    import torch
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from safetensors.numpy import save_file

    root = Path(root)
    torch.manual_seed(1000 + rank)
    model = torch.nn.Sequential(torch.nn.Linear(8, 6), torch.nn.ReLU(), torch.nn.Linear(6, 4))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0., .1)
    base = {name: value.detach().clone() for name, value in model.state_dict().items()}
    wrapped = get_peft_model(model, LoraConfig(r=rank, lora_alpha=rank, target_modules=['0'],
                                               lora_dropout=0., bias='none'))
    with torch.no_grad():
        for name, parameter in wrapped.named_parameters():
            if 'lora_A' in name:
                parameter.normal_(0., .1)
            elif 'lora_B' in name:
                parameter.normal_(0., .05)
    adapter = {name: value.detach() for name, value in get_peft_model_state_dict(wrapped).items()}
    merged = wrapped.merge_and_unload(safe_merge=True)
    merged_state = {name: value.detach().clone() for name, value in merged.state_dict().items()}

    def as_numpy(state):
        return {name: value.detach().cpu().contiguous().numpy() for name, value in state.items()}

    save_file(as_numpy(base), str(root / 'base.safetensors'))
    save_file(as_numpy(adapter), str(root / 'adapter.safetensors'))
    save_file(as_numpy(merged_state), str(root / 'merged.safetensors'))
    save_file(as_numpy(base), str(root / 'unchanged.safetensors'))

    a_key = next(name for name in adapter if 'lora_A' in name)
    b_key = next(name for name in adapter if 'lora_B' in name)
    update = (merged_state['0.weight'] - base['0.weight']).abs().max().item()
    spec = {'schema_version': 1, 'variant': 'standard_linear',
            'targets': [{'base': '0.weight', 'a': a_key, 'b': b_key, 'alpha': rank}]}
    wrong = {'schema_version': 1, 'variant': 'standard_linear',
             'targets': [{'base': '0.weight', 'a': a_key, 'b': b_key, 'alpha': 2 * rank}]}
    (root / 'manifest.json').write_text(json.dumps(spec), encoding='utf-8')
    (root / 'wrong-alpha.json').write_text(json.dumps(wrong), encoding='utf-8')
    return {'base': root / 'base.safetensors', 'adapter': root / 'adapter.safetensors',
            'merged': root / 'merged.safetensors', 'unchanged': root / 'unchanged.safetensors',
            'manifest': root / 'manifest.json', 'wrong_manifest': root / 'wrong-alpha.json',
            'rank': rank, 'a': a_key, 'b': b_key, 'max_update': update}
