from pathlib import Path
import unittest


class RepositorySkeletonTest(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]

    def test_expected_directories_exist(self) -> None:
        expected_directories = [
            "src",
            "scripts",
            "scripts/totalcapture_test",
            "data",
            "outputs",
            "tests",
            "docs",
            "third-party",
        ]
        for relative_path in expected_directories:
            with self.subTest(path=relative_path):
                self.assertTrue((self.repo_root / relative_path).is_dir(), msg=relative_path)

    def test_expected_files_exist(self) -> None:
        expected_files = [
            "README.md",
            ".gitignore",
            "src/totalcapture_test.py",
            "scripts/totalcapture_test/prepare_sample.py",
            "scripts/totalcapture_test/synthesize_imu.py",
            "scripts/totalcapture_test/plot_imu_comparison.py",
            "docs/pipeline_overview.md",
            "docs/repo_conventions.md",
            "docs/totalcapture_test_workflow.md",
            "third-party/README.md",
        ]
        for relative_path in expected_files:
            with self.subTest(path=relative_path):
                self.assertTrue((self.repo_root / relative_path).is_file(), msg=relative_path)

    def test_readme_mentions_core_workflow(self) -> None:
        readme_path = self.repo_root / "README.md"
        self.assertTrue(readme_path.is_file(), msg="README.md")
        readme_text = readme_path.read_text(encoding="utf-8")
        self.assertIn("synthetic IMU", readme_text)
        self.assertIn("data/interim", readme_text)
        self.assertIn("scripts/", readme_text)


if __name__ == "__main__":
    unittest.main()
