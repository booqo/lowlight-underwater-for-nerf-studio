"""Checks for the missing-source and package-import issues seen in fresh clones."""

import ast
from pathlib import Path
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "lowlight_underwater"


class SourceTests(unittest.TestCase):
    def test_model_losses_are_shipped(self):
        tree = ast.parse((PACKAGE / "lowlight_underwater_model.py").read_text())
        losses = ast.parse((PACKAGE / "losses.py").read_text())
        definitions = {n.name for n in losses.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
        imports = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module == "losses"]
        self.assertTrue(imports)
        for node in imports:
            for alias in node.names:
                self.assertIn(alias.name, definitions)
        self.assertNotIn("test_code", ast.unparse(tree))

    def test_cuda_imports_are_package_relative(self):
        for directory in ("rasterizerlight", "utils"):
            for path in (PACKAGE / directory).glob("*.py"):
                tree = ast.parse(path.read_text())
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        self.assertFalse(any(a.name == "cudalight" for a in node.names), str(path))

    def test_training_configs_have_required_fields(self):
        for path in (PACKAGE / "configs").glob("*.yaml"):
            with self.subTest(config=path.name):
                config = yaml.safe_load(path.read_text())
                required = {"model_name", "output_dir", "vis", "visualization_system",
                            "downscale_factor", "colmap_path", "images_path"}
                self.assertTrue(required <= config.keys())
                self.assertGreaterEqual(config["downscale_factor"], 1)

    def test_public_repository_scope(self):
        self.assertFalse((ROOT / "experiments").exists())
        self.assertFalse((PACKAGE / "viewer_configs").exists())
        expected = {
            "1_my_data_lowlight", "2_my_medium_light", "3_my_data_light",
            "4_Curasao", "5_IUI3-RedSea", "6_JapaneseGradens-RedSea", "7_Panama",
            "D2_seathru", "D3_seathru", "D4_seathru", "D5_seathru",
        }
        actual = {p.stem for p in (PACKAGE / "configs").glob("*.yaml")}
        self.assertTrue(expected <= actual)
        for name in actual:
            self.assertNotRegex(name, r"_D[0-9]+$|_turbL[0-9]+$|^llnerf|^oursea$")
        for path in PACKAGE.rglob("*.yaml"):
            self.assertNotRegex(path.read_text(), r"20[0-9]{2}-[0-9]{2}-[0-9]{2}_[0-9]{6}")

    def test_evaluation_and_render_configs(self):
        for directory in ("eval_configs", "render_configs"):
            paths = list((PACKAGE / directory).glob("*.yaml"))
            self.assertEqual([p.name for p in paths], ["example.yaml"])
            for path in paths:
                with self.subTest(config=str(path.relative_to(PACKAGE))):
                    config = yaml.safe_load(path.read_text())
                    self.assertIn("render_load_config", config)
                    self.assertIn("render_output_path", config)
                    if directory == "render_configs":
                        self.assertIn(config["render_split"], ("train", "test", "train+test", "val"))


if __name__ == "__main__":
    unittest.main()
