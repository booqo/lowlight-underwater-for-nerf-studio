#!/usr/bin/env python3
"""Check that a wheel contains runtime sources, configs, and license notices."""

import ast
from pathlib import PurePosixPath
import sys
import zipfile


def check_wheel(filename):
    with zipfile.ZipFile(filename) as wheel:
        names = set(wheel.namelist())
        required = {
            "lowlight_underwater/losses.py",
            "lowlight_underwater/cudalight/csrc/ext.cpp",
            "lowlight_underwater/cudalight/csrc/forward.cu",
            "lowlight_underwater/cudalight/csrc/backward.cu",
            "lowlight_underwater/cudalight/csrc/bindings.cu",
            "lowlight_underwater/cudalight/csrc/third_party/glm/glm/glm.hpp",
            "lowlight_underwater/cudalight/csrc/third_party/glm/glm/detail/setup.hpp",
            "lowlight_underwater/cudalight/csrc/third_party/glm/copying.txt",
            "lowlight_underwater/configs/D3_seathru.yaml",
            "lowlight_underwater/eval_configs/example.yaml",
            "lowlight_underwater/render_configs/example.yaml",
        }
        missing = required - names
        assert not missing, f"Missing runtime files: {sorted(missing)}"
        for suffix in ("/LICENSE", "/Apache-2.0.txt", "/THIRD_PARTY_NOTICES.md"):
            assert any(n.endswith(suffix) for n in names), f"Missing notice: {suffix}"
        forbidden = {"experiments", "viewer_configs", "test_code", "test_code2", "build", "outputs", "output", "__pycache__", ".idea", ".vscode"}
        for name in names:
            path = PurePosixPath(name)
            assert not forbidden.intersection(path.parts), f"Unexpected artifact: {name}"
            assert path.name not in {"3lowlight_underwater_model.py", "test2.py", "run_viewer.py"}, name
            assert path.suffix not in {".so", ".pyc", ".ckpt", ".pth"}, name
            if name.endswith(".py"):
                tree = ast.parse(wheel.read(name), filename=name)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.level and node.module:
                        parent = path.parent
                        for _ in range(node.level - 1):
                            parent = parent.parent
                        target = str(parent.joinpath(*node.module.split(".")))
                        assert target + ".py" in names or target + "/__init__.py" in names, (
                            f"Unshipped relative import in {name}: {node.module}"
                        )
        print(f"Verified runtime source, config, and license contents: {filename}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/check_wheel.py dist/*.whl")
    for filename in sys.argv[1:]:
        check_wheel(filename)
