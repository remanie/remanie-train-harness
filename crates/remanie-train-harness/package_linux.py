import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path


def package(binary: Path, destination: Path) -> None:
    if sys.platform != "linux":
        raise RuntimeError("Run packaging on the same Linux architecture as the build")
    rg = shutil.which("rg")
    if not binary.is_file() or not rg:
        raise RuntimeError("Provide the built harness and install ripgrep")
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "bin").mkdir()
    (destination / "lib").mkdir()
    manifest = {"architecture": platform.machine(), "files": {}}
    for name, source in (("remanie-train-harness", binary), ("rg", Path(rg))):
        target = destination / "bin" / name
        shutil.copy2(source, target)
        subprocess.run(["strip", "--strip-debug", str(target)], check=True)
        dependencies = subprocess.check_output(["ldd", str(source)], text=True)
        if "not found" in dependencies:
            raise RuntimeError(f"Missing runtime library for {name}")
        paths = [Path(value) for value in re.findall(r"(?:=>\s+)?(/[^\s()]+)", dependencies)]
        loaders = [path for path in paths if path.name.startswith("ld-linux")]
        if len(loaders) != 1:
            raise RuntimeError(f"Expected one glibc loader for {name}")
        for path in paths:
            library = destination / "lib" / path.name
            if library.exists() and library.read_bytes() != path.read_bytes():
                raise RuntimeError(f"Conflicting library: {path.name}")
            shutil.copy2(path, library)
        launcher = destination / name
        launcher.write_text(
            "#!/bin/sh\n"
            f"exec /opt/remanie-train-harness/lib/{loaders[0].name} "
            "--library-path /opt/remanie-train-harness/lib "
            f'/opt/remanie-train-harness/bin/{name} "$@"\n'
        )
        launcher.chmod(0o755)
    notices = destination / "licenses"
    notices.mkdir()
    for name in ("libc6", "libgcc-s1", "ripgrep"):
        copyright_file = Path("/usr/share/doc") / name / "copyright"
        if copyright_file.is_file():
            shutil.copyfile(copyright_file, notices / f"{name}.txt")
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            manifest["files"][str(path.relative_to(destination))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: package_linux.py BUILT_BINARY NEW_RUNTIME_DIRECTORY")
    package(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
