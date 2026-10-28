import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import package_linux


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
            owner = SimpleNamespace(returncode=0, stdout="libc6:arm64: /lib/libc.so.6\n")
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


if __name__ == "__main__":
    unittest.main()
