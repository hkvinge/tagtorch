from importlib.resources import files
from pathlib import Path

def project_root() -> Path:
    pkg_dir = Path(str(files("tagtorch"))).resolve()
    # adjust based on your layout:
    # if using src layout: repo_root / "src" / "your_package_name"
    for parent in [pkg_dir, *pkg_dir.parents]:
        if (parent / "pyproject.toml").exists() or (parent / "setup.cfg").exists():
            return parent
    return pkg_dir  # fallback