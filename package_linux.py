import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

DEBIAN_DOCS = Path("/usr/share/doc")
DEBIAN_LICENSES = Path("/usr/share/common-licenses")


def distro_notices(paths):
    notices = {}
    for path in sorted(paths):
        owners = []
        for candidate in dict.fromkeys((path, path.resolve())):
            result = subprocess.run(
                ["dpkg-query", "-S", str(candidate)], capture_output=True, text=True,
            )
            if result.returncode == 0:
                owners = [owner for line in result.stdout.splitlines()
                          for owner in line.split(": ", 1)[0].split(", ")]
                if owners:
                    break
        if not owners:
            raise RuntimeError(f"No Debian package owns redistributed file: {path}")
        for owner in owners:
            source = DEBIAN_DOCS / owner.split(":", 1)[0] / "copyright"
            if not source.is_file():
                raise RuntimeError(f"Missing required copyright for {owner}: {source}")
            notices[owner.replace(":", "_") + ".txt"] = source
            for name in re.findall(r"/usr/share/common-licenses/([A-Za-z0-9.+-]+)", source.read_text()):
                name = name.rstrip(".")
                license_file = DEBIAN_LICENSES / name
                if not license_file.is_file():
                    raise RuntimeError(f"Missing referenced Debian license: {license_file}")
                notices[f"common-licenses/{name}"] = license_file
    return notices


def launcher_script(name: str, loader: str) -> str:
    return (
        "#!/bin/sh\n"
        "here=${0%/*}\n"
        f'exec "$here/lib/{loader}" --library-path "$here/lib" "$here/bin/{name}" "$@"\n'
    )


def package(binary: Path, destination: Path) -> None:
    if sys.platform != "linux" or not shutil.which("dpkg-query"):
        raise RuntimeError("Run packaging on Debian or Ubuntu on the same architecture as the build")
    rg = shutil.which("rg")
    if not binary.is_file() or not rg:
        raise RuntimeError("Provide the built runner and install ripgrep")
    executables = []
    redistributed = {Path(rg)}
    for name, source in (("remanie-train-harness", binary), ("rg", Path(rg))):
        dependencies = subprocess.check_output(["ldd", str(source)], text=True)
        if "not found" in dependencies:
            raise RuntimeError(f"Missing runtime library for {name}")
        paths = [Path(value) for value in re.findall(r"(?:=>\s+)?(/[^\s()]+)", dependencies)]
        loaders = [path for path in paths if path.name.startswith("ld-linux")]
        if len(loaders) != 1:
            raise RuntimeError(f"Expected one glibc loader for {name}")
        executables.append((name, source, paths, loaders[0]))
        redistributed.update(paths)
    license_sources = distro_notices(redistributed)
    for name in ("LICENSE", "NOTICE", "THIRD-PARTY-NOTICES"):
        source = Path(__file__).resolve().parent / name
        if not source.is_file():
            raise RuntimeError(f"Missing required project notice: {source}")
        license_sources[name] = source
    contract = Path(__file__).resolve().parent / "assets/contract.json"
    if not contract.is_file():
        raise RuntimeError(f"Missing runner contract: {contract}")
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "bin").mkdir()
    (destination / "lib").mkdir()
    manifest = {"architecture": platform.machine(), "files": {}}
    for name, source, paths, loader in executables:
        target = destination / "bin" / name
        shutil.copy2(source, target)
        subprocess.run(["strip", "--strip-debug", str(target)], check=True)
        for path in paths:
            library = destination / "lib" / path.name
            if library.exists() and library.read_bytes() != path.read_bytes():
                raise RuntimeError(f"Conflicting library: {path.name}")
            shutil.copy2(path, library)
        launcher = destination / name
        launcher.write_text(launcher_script(name, loader.name))
        launcher.chmod(0o755)
    shutil.copyfile(contract, destination / "contract.json")
    notices = destination / "licenses"
    notices.mkdir()
    for name, source in sorted(license_sources.items()):
        target = notices / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            manifest["files"][str(path.relative_to(destination))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: package_linux.py BUILT_BINARY NEW_RUNTIME_DIRECTORY")
    package(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
