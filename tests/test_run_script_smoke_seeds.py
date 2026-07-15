import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_fake_python(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            r'''
            #!/usr/bin/env python3
            import json
            import os
            import sys
            from pathlib import Path

            log_path = Path(os.environ["FAKE_PY_LOG"])
            args = sys.argv[1:]
            with log_path.open("a") as log:
                log.write(json.dumps(args) + "\n")

            if args and args[0] == "-":
                # Emulate the run.sh inline aggregation snippets well enough for smoke tests.
                _script = sys.stdin.read()
                src = Path(args[1])
                arm = args[2]
                seed = int(args[3])
                dst = Path(args[4])
                with dst.open("a") as out:
                    for line in src.read_text().splitlines():
                        record = json.loads(line)
                        cluster = record.get("case_id") or record.get("study_id") or record.get("patient_id")
                        metric = record.get("dice", record.get("retrieval_top1", record.get("ed_es_mae", 0.0)))
                        leakage = record.get("seq_separability", record.get("angle_r2", record.get("vendor_separability")))
                        out.write(json.dumps({
                            "arm": arm,
                            "seed": seed,
                            "cluster": cluster,
                            "metric": metric,
                            "leakage": leakage,
                        }) + "\n")
                raise SystemExit(0)

            if args and args[0].endswith("analyze.py"):
                if "--out" in args:
                    Path(args[args.index("--out") + 1]).write_text(json.dumps({"ok": True}) + "\n")
                raise SystemExit(0)

            output_dir = None
            seed = None
            for index, arg in enumerate(args):
                if arg == "--override":
                    for override in args[index + 1:]:
                        if override.startswith("seed="):
                            seed = int(override.split("=", 1)[1])
                        elif override.startswith("output_dir="):
                            output_dir = Path(override.split("=", 1)[1])
                    break

            if output_dir is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                if "brats_seq_gauge" in str(output_dir):
                    metrics_name = "per_case_metrics.jsonl"
                    record = {"case_id": f"case-{seed}", "dice": 0.5, "seq_separability": 0.1}
                elif "cardiosyntax_angle_adv" in str(output_dir):
                    metrics_name = "per_study_metrics.jsonl"
                    record = {"study_id": f"study-{seed}", "retrieval_top1": 0.5, "angle_r2": 0.1}
                else:
                    metrics_name = "per_patient_metrics.jsonl"
                    record = {"patient_id": f"patient-{seed}", "ed_es_mae": 0.5, "vendor_separability": 0.1}
                (output_dir / metrics_name).write_text(json.dumps(record) + "\n")
            '''
        ).lstrip()
    )
    path.chmod(0o755)


def _smoke_trainer_seeds(script_dir: str, tmp_path: Path) -> list[int]:
    fake_py = tmp_path / "fake_python.py"
    log_path = tmp_path / f"{script_dir}.jsonl"
    _write_fake_python(fake_py)

    workdir = tmp_path / "experiments_v2" / script_dir
    workdir.mkdir(parents=True)
    shutil.copy(REPO_ROOT / "experiments_v2" / script_dir / "run.sh", workdir / "run.sh")

    env = os.environ.copy()
    env.update({
        "SMOKE": "1",
        "PY": str(fake_py),
        "FAKE_PY_LOG": str(log_path),
    })
    subprocess.run(
        ["bash", "run.sh"],
        cwd=workdir,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    seeds = []
    for line in log_path.read_text().splitlines():
        args = json.loads(line)
        if not args or not args[0].endswith("gaugeflow_lite.py"):
            continue
        override_index = args.index("--override")
        seed_arg = next(arg for arg in args[override_index + 1:] if arg.startswith("seed="))
        seeds.append(int(seed_arg.split("=", 1)[1]))
    return seeds


def test_smoke_defaults_to_two_seeds_for_brats_run_script(tmp_path: Path) -> None:
    assert _smoke_trainer_seeds("brats_seq_gauge", tmp_path) == [0, 0, 0, 1, 1, 1]


def test_smoke_defaults_to_two_seeds_for_cardiosyntax_run_script(tmp_path: Path) -> None:
    assert _smoke_trainer_seeds("cardiosyntax_angle_adv", tmp_path) == [0, 0, 0, 1, 1, 1]
