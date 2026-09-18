"""Resolve the replay H5 files listed by an imitation-training manifest."""

import json
from pathlib import Path


def load_dataset_paths(manifest_path):
    manifest = Path(manifest_path).expanduser().resolve()
    if manifest.suffix.lower() != '.json':
        raise ValueError(f'pretrain.dataset_path must name a JSON file: {manifest}')

    with manifest.open(encoding='utf-8') as source:
        entries = json.load(source)
    if isinstance(entries, dict):
        entries = entries.get('datasets')
    if not isinstance(entries, list) or not entries:
        raise ValueError(f'{manifest} must contain a nonempty list of dataset paths')

    paths = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, str) or not entry.strip():
            raise ValueError(f'{manifest}: datasets[{index}] must be a nonempty path string')
        path = Path(entry).expanduser()
        if not path.is_absolute():
            path = manifest.parent / path
        path = path.resolve()
        if path.suffix.lower() not in ('.h5', '.hdf5') or not path.is_file():
            raise FileNotFoundError(f'{manifest}: datasets[{index}] is not an H5 file: {path}')
        paths.append(str(path))
    return paths
