from pathlib import Path

import pytest

from benchmark_toolbox.environments.patches import apply_patches

ENV_YML = "\n".join(
    [
        "dependencies:",
        "  - python=3.8",
        "  - pytorch=1.1.0=py3.6_cuda9.0.176_cudnn7.5.1_0",
        "  - cudatoolkit=9.0",
        "  - numpy",
    ]
)

DEPIN_PATCH = [
    {
        "file": "environment.yml",
        "drop": ["cudatoolkit=9.0"],
        "relax": ["pytorch=1.1.0"],
    }
]


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "environment.yml").write_text(ENV_YML, encoding="utf-8")
    return tmp_path


def _read(repo: Path, name: str = "environment.yml") -> str:
    return (repo / name).read_text(encoding="utf-8")


def test_depin_drop_and_relax(repo: Path) -> None:
    apply_patches(DEPIN_PATCH, repo)

    text = _read(repo)
    assert "cudatoolkit" not in text
    assert "cuda9.0.176" not in text
    assert "pytorch=1.1.0" in text
    assert "numpy" in text


def test_depin_is_idempotent(repo: Path) -> None:
    apply_patches(DEPIN_PATCH, repo)
    once = _read(repo)
    apply_patches(DEPIN_PATCH, repo)
    assert _read(repo) == once


def test_prepend_is_idempotent(tmp_path: Path) -> None:
    main = tmp_path / "main.py"
    main.write_text("print('x')\n", encoding="utf-8")
    patch = [{"file": "main.py", "prepend": "import cpu_patch\n"}]

    apply_patches(patch, tmp_path)
    apply_patches(patch, tmp_path)

    assert main.read_text(encoding="utf-8").count("import cpu_patch") == 1


def test_insert_after_lands_under_anchor(repo: Path) -> None:
    patch = [
        {
            "file": "environment.yml",
            "after": "dependencies:",
            "lines": ["  - setuptools<58", "  - wheel<0.38"],
        }
    ]
    apply_patches(patch, repo)

    lines = _read(repo).splitlines()
    anchor = lines.index("dependencies:")
    assert lines[anchor + 1] == "  - setuptools<58"
    assert lines[anchor + 2] == "  - wheel<0.38"


def test_insert_after_is_idempotent(repo: Path) -> None:
    patch = [
        {
            "file": "environment.yml",
            "after": "dependencies:",
            "lines": ["  - setuptools<58"],
        }
    ]
    apply_patches(patch, repo)
    once = _read(repo)
    apply_patches(patch, repo)

    assert _read(repo) == once
    assert once.count("setuptools<58") == 1
