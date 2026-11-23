from pathlib import Path


def test_project_root_contains_core_files():
    project_root = Path(__file__).resolve().parents[1]

    assert (project_root / "pyproject.toml").exists()
    assert (project_root / "rag_core.py").exists()
    assert (project_root / "agent.py").exists()