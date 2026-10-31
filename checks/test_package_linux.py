import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import package_linux
from licenses import license_files



class PackageNoticesTest(unittest.TestCase):
    def test_referenced_debian_licenses_are_required_and_bundled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs, licenses = root / "doc", root / "common-licenses"
            (docs / "libc6").mkdir(parents=True)
            licenses.mkdir()
            copyright_file = docs / "libc6/copyright"
            copyright_file.write_text(
                "See `/usr/share/common-licenses/LGPL-2.1' and /usr/share/common-licenses/GPL.\n"
            )
            for name in ("LGPL-2.1", "GPL"):
                (licenses / name).write_text("test license text\n")
            (root / "usr/lib").mkdir(parents=True)
            (root / "usr/lib/libc.so.6").write_text("")
            (root / "lib").symlink_to("usr/lib")
            library = root / "lib/libc.so.6"
            self.assertNotEqual(library.resolve(), library)

            def dpkg_query(command, **options):
                self.assertEqual(options["env"]["LC_ALL"], "C")
                if command[2] == str(library):
                    return SimpleNamespace(returncode=0, stdout=(
                        f"diversion by libc6 from: {library}\n"
                        f"diversion by libc6 to: {library}.usr-is-merged\n"
                        f"local diversion from: {library}\n"
                        f"local diversion to: {library}.distrib\n"
                    ))
                return SimpleNamespace(returncode=0, stdout=f"libc6:arm64: {command[2]}\n")

            with patch.object(package_linux, "DEBIAN_DOCS", docs), \
                    patch.object(package_linux, "DEBIAN_LICENSES", licenses), \
                    patch.object(package_linux.subprocess, "run", side_effect=dpkg_query):
                paths = {library}
                self.assertEqual(package_linux.distro_notices(paths), {
                    "libc6_arm64.txt": copyright_file,
                    "common-licenses/LGPL-2.1": licenses / "LGPL-2.1",
                    "common-licenses/GPL": licenses / "GPL",
                })
                (licenses / "LGPL-2.1").unlink()
                with self.assertRaisesRegex(RuntimeError, "Missing referenced Debian license"):
                    package_linux.distro_notices(paths)


class InventoryTest(unittest.TestCase):
    def test_uncovered_package_fails_instead_of_omitting_it(self):
        with tempfile.TemporaryDirectory() as directory:
            package = {
                "manifest_path": str(Path(directory) / "Cargo.toml"),
                "name": "uncovered", "version": "1.0.0", "license_file": None,
            }
            with self.assertRaisesRegex(ValueError, "Missing license text for uncovered 1.0.0"):
                license_files(package)


class LauncherTest(unittest.TestCase):
    def test_launcher_runs_the_bundle_it_sits_in(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "first"
            (bundle / "lib").mkdir(parents=True)
            loader = bundle / "lib/ld-linux-x86-64.so.2"
            loader.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
            loader.chmod(0o755)
            launcher = bundle / "rg"
            launcher.write_text(package_linux.launcher_script("rg", loader.name))
            launcher.chmod(0o755)
            moved = bundle.rename(Path(directory) / "moved bundle")
            output = subprocess.run(
                [str(moved / "rg"), "--version", "a b"], cwd="/", capture_output=True, text=True, check=True,
            ).stdout.splitlines()
            self.assertEqual(output, ["--library-path", f"{moved}/lib", f"{moved}/bin/rg", "--version", "a b"])

    def test_runner_launcher_names_the_bundled_grok(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            (bundle / "lib").mkdir()
            loader = bundle / "lib/ld-linux-x86-64.so.2"
            loader.write_text('#!/bin/sh\nprintf "%s\\n" "$REMANIE_GROK"\n')
            loader.chmod(0o755)
            launcher = bundle / "remanie-train-harness"
            launcher.write_text(package_linux.launcher_script(
                "remanie-train-harness", loader.name, package_linux.EXPORTS["remanie-train-harness"],
            ))
            launcher.chmod(0o755)
            output = subprocess.run([str(launcher)], cwd="/", capture_output=True, text=True, check=True).stdout
            self.assertEqual(output, f"{bundle}/grok\n")


class PinTest(unittest.TestCase):
    def test_notice_names_the_commit_the_build_pins(self):
        root = Path(__file__).resolve().parents[1]
        commit = json.loads((root / "upstream.json").read_text())["commit"]
        self.assertIn(commit, (root / "NOTICE").read_text())


@unittest.skipUnless(os.environ.get("REMANIE_BUNDLE"), "set REMANIE_BUNDLE to a package_linux.py output")
class PackagedBundleTest(unittest.TestCase):
    def test_bundle_carries_what_the_harbor_agent_reads(self):
        root = Path(os.environ["REMANIE_BUNDLE"]).resolve()
        manifest = json.loads((root / "manifest.json").read_text())
        self.assertIn("contract.json", manifest["files"])
        self.assertEqual(json.loads((root / "contract.json").read_text())["exit_codes"], {"gradable": 0, "infrastructure": 70})
        for name in ("remanie-train-harness", "grok", "rg"):
            self.assertTrue(os.access(root / name, os.X_OK), name)
        subprocess.run([str(root / "rg"), "--version"], cwd="/", check=True, capture_output=True)
        commit = json.loads((Path(__file__).resolve().parents[1] / "upstream.json").read_text())["commit"]
        with tempfile.TemporaryDirectory() as home:
            version = subprocess.run(
                [str(root / "grok"), "--version"], cwd="/", check=True, capture_output=True, text=True,
                env={"PATH": os.environ["PATH"], "HOME": home, "GROK_HOME": home},
            ).stdout
        self.assertIn(f"({commit[:12]})", version)
        self.assertIn("licenses/grok-build/LICENSE", manifest["files"])


if __name__ == "__main__":
    unittest.main()
