"""Regression tests for the fresh-machine scripts (setup.sh / reset.sh) and README consistency."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _bash_ok(script: Path) -> str:
    assert script.exists(), script
    subprocess.run(["bash", "-n", str(script)], check=True)  # syntax only, nothing runs
    text = script.read_text()
    assert "set -euo pipefail" in text
    return text


def test_setup_script_parses_and_covers_every_dependency():
    text = _bash_ok(ROOT / "setup.sh")
    assert "requirements.txt" in text and "pytest" in text
    assert "Qwen/Qwen2.5-1.5B-Instruct" in text  # pre-downloads the model
    assert "xcode-select --install" in text  # fresh-Mac hint when python3 is missing


def test_reset_script_removes_everything_setup_creates(tmp_path):
    text = _bash_ok(ROOT / "reset.sh")
    for target in (".venv", "qwen-abliterated", "qwen-abliterated-healed", "dpo_pairs.jsonl",
                   "__pycache__", ".pytest_cache"):
        assert target in text, target
    assert "--full" in text and "models--Qwen--Qwen2.5-1.5B-Instruct" in text

    # Run it for real against a scratch copy of the repo layout (default mode, no HF cache).
    copy = tmp_path / "repo"
    copy.mkdir()
    (copy / "reset.sh").write_text(text)
    for d in (".venv", ".pytest_cache", "qwen-abliterated", "qwen-abliterated-healed",
              "tests/__pycache__"):
        (copy / d).mkdir(parents=True)
        (copy / d / "x").write_text("x")
    (copy / "dpo_pairs.jsonl").write_text("{}")
    (copy / "scratch.txt").write_text("keep me")
    subprocess.run(["bash", str(copy / "reset.sh")], check=True, capture_output=True)
    for gone in (".venv", ".pytest_cache", "qwen-abliterated", "qwen-abliterated-healed",
                 "dpo_pairs.jsonl", "tests/__pycache__"):
        assert not (copy / gone).exists(), gone
    assert (copy / "scratch.txt").read_text() == "keep me"  # user files survive a reset


def test_readme_matches_repo():
    readme = (ROOT / "README.md").read_text()
    for cmd in ("bash setup.sh", "python abliterate.py", "python chat.py"):
        assert cmd in readme, f"README quick start must keep: {cmd}"
