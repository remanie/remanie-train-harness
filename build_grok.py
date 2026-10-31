"""Build Grok Build's xai-grok-pager at the commit upstream.json pins, with upstream's Cargo.lock, protoc and ripgrep."""

import argparse
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PIN = json.loads((ROOT / "upstream.json").read_text())
NOTICES = ("LICENSE", "THIRD-PARTY-NOTICES", "crates/codegen/xai-grok-tools/THIRD_PARTY_NOTICES.md", "third_party/NOTICE")


def git(source, *arguments):
    return subprocess.run(["git", "-C", str(source), *arguments], check=True, capture_output=True, text=True).stdout.strip()


def checkout(source):
    if not (source / ".git").is_dir():
        source.mkdir(parents=True, exist_ok=True)
        git(source, "init", "-q")
        git(source, "remote", "add", "origin", PIN["repository"])
    if subprocess.run(["git", "-C", str(source), "rev-parse", "-q", "--verify", "HEAD"], capture_output=True, text=True).stdout.strip() != PIN["commit"]:
        git(source, "fetch", "-q", "--depth", "1", "origin", PIN["commit"])
        git(source, "checkout", "-q", "--detach", PIN["commit"])
    if git(source, "rev-parse", "HEAD") != PIN["commit"] or git(source, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError(f"{source} is not a clean checkout of {PIN['commit']}")


def download(url, sha256):
    with urllib.request.urlopen(url) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise RuntimeError(f"{url} does not match its pinned sha256 {sha256}")
    return data


def protoc(source, tools):
    """Fetch the protoc that upstream's DotSlash file bin/protoc pins for this host."""
    spec = json.loads((source / "bin/protoc").read_text().split("\n", 1)[1])
    entry = spec["platforms"][f"linux-{platform.machine()}"]
    if entry["hash"] != "sha256" or entry["format"] != "zip":
        raise RuntimeError(f"Unexpected DotSlash entry for protoc: {entry}")
    directory = tools / f"protoc-{entry['digest'][:12]}"
    if not (directory / entry["path"]).is_file():
        shutil.rmtree(directory, ignore_errors=True)
        with zipfile.ZipFile(io.BytesIO(download(entry["providers"][0]["url"], entry["digest"]))) as archive:
            archive.extractall(directory)
        (directory / entry["path"]).chmod(0o755)
    return directory / entry["path"]


def ripgrep(tools, arch):
    """Fetch the ripgrep release that upstream's build scripts embed for this target."""
    entry = PIN["ripgrep"][arch]
    directory = tools / f"ripgrep-{entry['sha256'][:12]}"
    if not (directory / "rg").is_file():
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True)
        with tarfile.open(fileobj=io.BytesIO(download(entry["url"], entry["sha256"]))) as archive:
            for member in archive.getmembers():
                name = Path(member.name).name
                if member.isfile() and name in ("rg", "COPYING", "LICENSE-MIT", "UNLICENSE"):
                    (directory / name).write_bytes(archive.extractfile(member).read())
        (directory / "rg").chmod(0o755)
    return directory


def build(work, target):
    source, tools = work / "src", work / "tools"
    checkout(source)
    arch = (target or platform.machine()).split("-")[0]
    rg = ripgrep(tools, arch)
    env = {
        **os.environ,
        "PROTOC": str(protoc(source, tools)),
        "GROK_TOOLS_BUNDLE_RG_PATH": str(rg / "rg"),
        "GROK_SHELL_BUNDLE_RG_PATH": str(rg / "rg"),
        "CARGO_INCREMENTAL": "0",
    }
    command = ["cargo", "build", "--release", "--locked", "-p", PIN["package"], *(["--target", target] if target else [])]
    subprocess.run(command, cwd=source, env=env, check=True)
    built = source / "target" / (target or "") / "release" / PIN["binary"]
    out = work / "out"
    shutil.rmtree(out, ignore_errors=True)
    (out / "licenses/ripgrep").mkdir(parents=True)
    shutil.copy2(built, out / "grok")
    for name in NOTICES:
        shutil.copyfile(source / name, out / "licenses" / name.replace("/", "_"))
    for notice in ("COPYING", "LICENSE-MIT", "UNLICENSE"):
        shutil.copyfile(rg / notice, out / "licenses/ripgrep" / notice)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=ROOT / "target/grok-build", help="holds the checkout, the fetched tools and out/")
    parser.add_argument("--target", help="a Rust target triple to cross-compile for")
    args = parser.parse_args()
    print(build(args.work.resolve(), args.target))
