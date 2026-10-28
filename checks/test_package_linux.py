import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import package_linux

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harbor"))
from remanie_agent import Bundle


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
            owner = SimpleNamespace(
                returncode=0,
                stdout="diversion by libc6 from: /lib/libc.so.6\n"
                "diversion by libc6 to: /lib/libc.so.6.usr-is-merged\n"
                "libc6:arm64: /lib/libc.so.6\n",
            )
            with patch.object(package_linux, "DEBIAN_DOCS", docs), \
                    patch.object(package_linux, "DEBIAN_LICENSES", licenses), \
                    patch.object(package_linux.subprocess, "run", return_value=owner):
                paths = {Path("/lib/libc.so.6")}
                self.assertEqual(package_linux.distro_notices(paths), {
                    "libc6_arm64.txt": copyright_file,
                    "common-licenses/LGPL-2.1": licenses / "LGPL-2.1",
                    "common-licenses/GPL": licenses / "GPL",
                })
                (licenses / "LGPL-2.1").unlink()
                with self.assertRaisesRegex(RuntimeError, "Missing referenced Debian license"):
                    package_linux.distro_notices(paths)


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


@unittest.skipUnless(os.environ.get("REMANIE_BUNDLE"), "set REMANIE_BUNDLE to a package_linux.py output")
class PackagedBundleTest(unittest.TestCase):
    def test_bundle_carries_the_contract_the_adapter_reads(self):
        root = Path(os.environ["REMANIE_BUNDLE"])
        manifest = json.loads((root / "manifest.json").read_text())
        self.assertIn("contract.json", manifest["files"])
        self.assertEqual(Bundle.load(root).architecture, manifest["architecture"])
        subprocess.run([str(root / "rg"), "--version"], cwd="/", check=True, capture_output=True)


if __name__ == "__main__":
    unittest.main()
