from pathlib import Path

from pidcast.config import PROJECT_ROOT, get_project_root


def test_project_root_exists():
    """Test that project root is correctly resolved and exists."""
    assert PROJECT_ROOT.exists()
    assert (PROJECT_ROOT / "pyproject.toml").exists()


def test_get_project_root():
    """Test get_project_root function."""
    root = get_project_root()
    assert isinstance(root, Path)
    assert root.exists()
    assert (root / "src").exists()
