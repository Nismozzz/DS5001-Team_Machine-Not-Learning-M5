"""Resume verified groups without rerunning model selection. Run from project root."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

import pandas as pd

from modeling.common import digest, fingerprint, read_json, require, write_json
from modeling.contracts import check_daily, check_period, seal, verify_run
from modeling.pipeline import Inputs, make_split, run_group


def check_group(folder, split, level, config):
    path = folder / 'predictions' / split / f'L{level:02d}'
    if not (path / 'diagnostics.json').exists():
        return False
    try:
        coverage = read_json(path / 'coverage.json')
        frame = pd.read_parquet(path / 'daily.parquet')
        origin = int(frame.origin_d.iloc[0])
        check_daily(frame, coverage['node_ids'], origin, config['daily_quantiles'])
        check_period(pd.read_parquet(path / 'period.parquet'), coverage['node_ids'], origin,
                     config['period_quantiles'])
        read_json(path / 'metrics.json')
        read_json(path / 'baseline_events.json')
        if origin + 28 <= 1941:
            residuals = pd.read_parquet(path / 'residuals.parquet')
            require(len(residuals) == len(frame), 'Residual row count mismatch')
        return True
    except (OSError, ValueError, KeyError, IndexError):
        return False


def preserve_partial(folder, split, level):
    # Only move concrete, checked descendants of this run; never delete results.
    folder = folder.resolve()
    backup = folder.parent / (folder.name + '_interrupted') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    for kind in ['predictions', 'models']:
        source = (folder / kind / split / f'L{level:02d}').resolve()
        require(source.is_relative_to(folder) and source != folder, 'Unsafe partial-result path')
        if source.exists():
            target = backup / kind / split / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)


def resume(folder, data):
    folder = Path(folder).resolve()
    context = read_json(folder / 'run_context.json')
    config = context['config']
    lock = read_json(folder / 'selected_config.json')
    lock_hash = lock.pop('selection_hash')
    require(fingerprint(lock) == lock_hash, 'Selection lock changed')
    require(context['config_hash'] == fingerprint(config) == lock['config_hash'], 'Configuration mismatch')
    require(lock['selected'] == context['selected'], 'Selected parameters mismatch')
    source_snapshot = folder / 'source_files_at_start.json'
    for relative, expected in context.get('model_source_sha256', {}).items():
        require(digest(Path(__file__).resolve().parent / relative) == expected,
                f'Model code changed since this run: {relative}')
    if source_snapshot.exists():
        root = Path(__file__).resolve().parent
        for relative, expected in read_json(source_snapshot).items():
            if relative.endswith('.py') and not relative.startswith('tests/'):
                require(digest(root / relative) == expected, f'Model code changed: {relative}')
    inputs = Inputs(data, context['input_signature']['source_scope'],
                    len(context['series_ids']) if context['scope'] == 'smoke' else None)
    inputs.verify_inputs()
    require(inputs.signature() == context['input_signature'], 'Input version or leaf set changed')
    require(inputs.ids == context['series_ids'], 'Sample order changed')
    available = {s['split_id']: s for s in inputs.splits}
    for origin in config['calibration_origins']:
        available[f'calibration_{origin}'] = make_split(origin)
    audit, started = [], time.monotonic()
    write_json(folder / 'resume_status.json', {'status': 'running', 'completed_groups': 0})
    for name in context['requested_splits']:
        split = available[name]
        for level in config['levels']:
            if check_group(folder, name, level, config):
                action = 'reused'
                print(f'Reuse {name} L{level:02d}', flush=True)
            else:
                preserve_partial(folder, name, level)
                run_group(inputs, split, level, config, context['selected'], folder, context)
                action = 'rebuilt'
            audit.append({'split': name, 'level': level, 'action': action})
            write_json(folder / 'resume_status.json', {'status': 'running', 'completed_groups': len(audit), 'groups': audit})
    write_json(folder / 'run_summary.json', {'status': 'completed', 'mode': context['mode'],
               'scope': context['scope'], 'leaf_count': len(inputs.ids),
               'resume_elapsed_seconds': round(time.monotonic() - started, 3),
               'timing_note': 'Resume time excludes previous interrupted work; per-group timings retained.'})
    write_json(folder / 'resume_status.json', {'status': 'completed', 'groups': audit})
    seal(folder)
    report = verify_run(folder)
    write_json(folder / 'verification_report.json', report)
    seal(folder)
    print(report, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('data/processed'))
    args = parser.parse_args()
    resume(args.output, args.data)
