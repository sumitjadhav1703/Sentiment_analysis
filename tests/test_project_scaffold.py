from pathlib import Path


def test_required_project_files_exist():
    repo_root = Path(__file__).resolve().parents[1]
    required_paths = [
        repo_root / "requirements.txt",
        repo_root / ".gitignore",
        repo_root / "artifacts/.gitkeep",
    ]

    missing = [str(path.relative_to(repo_root)) for path in required_paths if not path.exists()]

    assert missing == []
