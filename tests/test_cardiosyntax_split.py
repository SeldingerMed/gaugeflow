import json
import os
import subprocess
import sys
from pathlib import Path


DATASET_PATH = Path(__file__).resolve().parents[1] / "experiments_v2" / "cardiosyntax_angle_adv" / "dataset.py"


def split_map_for_hash_seed(seed: int, study_uids: list[str]) -> dict[str, str]:
    # PYTHONHASHSEED is only applied at interpreter startup, so this regression
    # test must launch isolated Python processes rather than calling _split_of()
    # directly in the already-running pytest process.
    code = f"""
import importlib.util
import json
spec = importlib.util.spec_from_file_location('cardiosyntax_dataset', {str(DATASET_PATH)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
study_uids = {study_uids!r}
print(json.dumps({{uid: ('train' if module._split_of(uid, 'train') else 'test') for uid in study_uids}}, sort_keys=True))
"""
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)
    output = subprocess.check_output(
        [sys.executable, "-c", code],
        env=env,
        text=True,
        timeout=30,
    )
    return json.loads(output)


def test_cardiosyntax_split_is_stable_across_python_hash_seeds():
    study_uids = [
        "1.2.840.113619.2.55.3.604688432.1234.1",
        "CARDIOSYNTAX-STUDY-0007",
        "another-study-with-repeatable-assignment",
        "small-cohort-edge-case",
    ]

    seed_0_split = split_map_for_hash_seed(0, study_uids)
    seed_1_split = split_map_for_hash_seed(1, study_uids)
    seed_random_split = split_map_for_hash_seed(987654, study_uids)

    assert seed_1_split == seed_0_split
    assert seed_random_split == seed_0_split


def test_cardiosyntax_split_keeps_roughly_70_30_study_level_balance():
    study_uids = [f"CARDIOSYNTAX-STUDY-{index:04d}" for index in range(1000)]

    split_map = split_map_for_hash_seed(0, study_uids)
    train_count = sum(1 for split in split_map.values() if split == "train")
    test_count = sum(1 for split in split_map.values() if split == "test")

    assert 650 <= train_count <= 750
    assert 250 <= test_count <= 350
    assert train_count + test_count == len(study_uids)
