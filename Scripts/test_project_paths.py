"""Check relocation rules without running experiments or producing figures."""
import json
from pathlib import Path
import tempfile
import unittest

from project_paths import PROJECT_ROOT, resolve_project_path


class ProjectPathsTest(unittest.TestCase):
    def test_relative_and_absolute_frozen_paths(self):
        old = 'generated_dataset/rareclass_expansion_v1/frozen_18000_v1/labels_main_18000_v1.json'
        expected = PROJECT_ROOT / 'dataset/labels_main_18000_v1.json'
        for path in (old, old.replace('/', '\\'), PROJECT_ROOT / old):
            self.assertEqual(resolve_project_path(path), expected)

    def test_cache_move_precedes_general_rename(self):
        path = 'experiments/expansion_18000_balanced_v1/cache/train_validation_inputs.pkl'
        self.assertEqual(resolve_project_path(path), PROJECT_ROOT / 'experiment_data/cache/train_validation_inputs.pkl')

    def test_existing_current_paths_and_external_paths(self):
        current = PROJECT_ROOT / 'experiment_data/phase3_scratch_18000_gpu_v1/evaluation/summary.json'
        self.assertEqual(resolve_project_path(current), current)
        with tempfile.TemporaryDirectory() as temporary:
            external = Path(temporary) / 'experiments/results.json'
            self.assertEqual(resolve_project_path(external), external)

    def test_exact_directory_boundaries_and_missing_history(self):
        for path in ('experiments_backup/results.json', 'generated_dataset/sgp4_large_maneuver_1min/x_samples_frozen_v5_clean.json'):
            self.assertEqual(resolve_project_path(path), PROJECT_ROOT / path)

    def test_custom_project_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            old = root / 'experiments/phase3_scratch_18000_gpu_v1/evaluation/summary.json'
            self.assertEqual(resolve_project_path(old, root), root / 'experiment_data/phase3_scratch_18000_gpu_v1/evaluation/summary.json')

    def test_frozen_output_keys_follow_dataset_move(self):
        manifest = json.loads((PROJECT_ROOT / 'dataset/freeze_manifest.json').read_bytes())
        outputs = {resolve_project_path(name): value for name, value in manifest['outputs'].items()}
        for name in ('dataset/x_samples_frozen_18000_v1.json', 'dataset/sample_provenance_frozen.json',
                     'experiment_data/splits/actor_isolated_13000_2500_2500_v1__labels_18000_v1.json'):
            path = PROJECT_ROOT / name
            self.assertEqual(path.stat().st_size, outputs[path]['bytes'])

    def test_analysis_outputs_do_not_target_frozen_figures(self):
        self.assertEqual(resolve_project_path('paper_figures/matching_score_legacy_20261019'),
                         PROJECT_ROOT / 'experiment_data/matching_score_legacy_20261019')

    def test_frozen_manifests_resolve_without_changes(self):
        for phase in (2, 3):
            path = PROJECT_ROOT / f'experiment_data/phase{phase}_scratch_18000_gpu_v1/training_protocol.json'
            before = path.read_bytes()
            protocol = json.loads(before)
            recorded = [protocol['dataset_path'], protocol['split_manifest'], protocol['raw_input_cache']['path']]
            recorded.extend(item['path'] for item in protocol['normalization_files'].values())
            recorded.extend(protocol.get('reference_phase2', {}).get('artifacts_sha256', {}))
            for name in recorded:
                with self.subTest(path=name):
                    self.assertTrue(resolve_project_path(name).is_file())
            self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
