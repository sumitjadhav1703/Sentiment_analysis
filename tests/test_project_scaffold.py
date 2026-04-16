from pathlib import Path


def test_required_project_files_exist():
    required_paths = [
        Path("requirements.txt"),
        Path(".gitignore"),
        Path("artifacts/.gitkeep"),
    ]

    missing = [str(path) for path in required_paths if not path.exists()]

    assert missing == []
