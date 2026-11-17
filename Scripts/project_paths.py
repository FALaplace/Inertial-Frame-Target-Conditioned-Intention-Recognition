"""Resolve relocated project files without rewriting frozen provenance records."""
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path, project_root=PROJECT_ROOT):
    root = Path(project_root).resolve()
    path = Path(str(path).replace('\\', '/'))
    if path.is_absolute():
        try:
            relative = path.relative_to(root)
        except ValueError:
            return path
    else:
        relative = path
    # Specific moves precede the general experiments-directory rename.
    for old, current in (
        ('generated_dataset/rareclass_expansion_v1/frozen_18000_v1', 'dataset'),
        ('experiments/expansion_18000_balanced_v1/cache', 'experiment_data/cache'),
        ('experiments', 'experiment_data'),
        ('paper_figures', 'experiment_data'),
    ):
        try:
            suffix = relative.relative_to(old)
        except ValueError:
            continue
        return root / current / suffix
    return root / relative
