"""Strict experimental defaults; explicit CLI options always win."""
import json
import sys


def apply_config(parser, args, section):
    path = getattr(args, 'experiment_config', None)
    if not path:
        return args
    data = json.loads(path.read_text(encoding='utf-8'))
    values = data.get(section, {})
    forbidden = {'image','initial_mask','manifest','mask','output','feedback_graph',
                 'initial_proposals','reference_ids','candidate_json','experiment_config'}
    actions = {a.dest:a for a in parser._actions}
    supplied = {s.split('=')[0] for s in sys.argv[1:] if s.startswith('--')}
    for name, value in values.items():
        if name in forbidden or name not in actions:
            raise ValueError(f'Unsupported experimental parameter: {section}.{name}')
        action = actions[name]
        if any(flag in supplied for flag in action.option_strings):
            continue
        if isinstance(action.default, bool) and not isinstance(value,bool):
            raise ValueError(f'{name} must be a JSON boolean')
        if action.type and value is not None:
            value=action.type(value)
        if action.choices and value not in action.choices:
            raise ValueError(f'Invalid value for {name}: {value}')
        setattr(args,name,value)
    return args
