"""Copy Grok Build files verbatim into vendor/, or check that the copies still match upstream."""

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "vendor" / "UPSTREAM.txt"
REV = "2bdd1d6a6369de0e8c68132ea4539e9abd9e14a8"
SHELL = "crates/codegen/xai-grok-shell/src/"
FILES = {
    "vendor/grok-shell/src/session/two_pass.rs": SHELL + "session/two_pass.rs",
    "vendor/grok-shell/src/session/helpers/session_compact.rs": SHELL + "session/helpers/session_compact.rs",
    "vendor/grok-shell/src/session/helpers/full_replace_compaction.rs": SHELL + "session/helpers/full_replace_compaction.rs",
    "vendor/grok-shell/src/session/helpers/prepared_compaction_history.rs": SHELL + "session/helpers/prepared_compaction_history.rs",
    "vendor/grok-shell/src/session/helpers/compaction_context.rs": SHELL + "session/helpers/compaction_context.rs",
    "vendor/grok-shell/src/session/helpers/tool_input_parsing.rs": SHELL + "session/helpers/tool_input_parsing.rs",
}


def blob_sha(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def refresh(upstream):
    tree = subprocess.run(["git", "-C", str(upstream), "ls-tree", "-r", REV, "--", *FILES.values()], capture_output=True, text=True, check=True).stdout
    shas = {line.split("\t")[1]: line.split()[2] for line in tree.splitlines()}
    lines = [f"# grok-build {REV}: vendored path, upstream path, upstream git blob sha"]
    for local, remote in FILES.items():
        data = subprocess.run(["git", "-C", str(upstream), "show", f"{REV}:{remote}"], capture_output=True, check=True).stdout
        if blob_sha(data) != shas[remote]:
            sys.exit(f"{remote}: git show disagrees with ls-tree")
        (REPO / local).parent.mkdir(parents=True, exist_ok=True)
        (REPO / local).write_bytes(data)
        lines.append(f"{local} {remote} {shas[remote]}")
    MANIFEST.write_text("\n".join(lines) + "\n")


def check():
    bad = []
    for line in MANIFEST.read_text().splitlines():
        if line.startswith("#"):
            continue
        local, remote, sha = line.split()
        if FILES.get(local) != remote:
            bad.append(f"{local}: manifest names {remote}, script names {FILES.get(local)}")
        elif blob_sha((REPO / local).read_bytes()) != sha:
            bad.append(f"{local}: differs from {remote} at {REV[:12]}")
    missing = set(FILES) - {line.split()[0] for line in MANIFEST.read_text().splitlines() if not line.startswith("#")}
    bad += [f"{local}: not in {MANIFEST.name}" for local in sorted(missing)]
    print("\n".join(bad) or f"{len(FILES)} vendored files match grok-build {REV[:12]}")
    sys.exit(1 if bad else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", type=Path, metavar="GROK_BUILD_CLONE", help="copy the files from this clone at the pinned commit and rewrite the manifest")
    args = parser.parse_args()
    if args.refresh:
        refresh(args.refresh)
    check()


if __name__ == "__main__":
    main()
