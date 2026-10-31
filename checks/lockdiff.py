"""List the crates whose locked versions in the runner's Cargo.lock differ from Grok Build's Cargo.lock; exit 1 when any does."""

import sys
import tomllib
from collections import defaultdict

def versions(path):
    out = defaultdict(set)
    for package in tomllib.load(open(path, "rb"))["package"]:
        source = package.get("source", "path")
        kind = "git" if source.startswith("git+") else "path" if source == "path" else "registry"
        out[package["name"]].add((package["version"], kind))
    return out

if len(sys.argv) != 3:
    sys.exit("usage: lockdiff.py RUNNER_CARGO_LOCK GROK_BUILD_CARGO_LOCK")
runner, upstream = versions(sys.argv[1]), versions(sys.argv[2])
shared = sorted(set(runner) & set(upstream))
differ = [(name, sorted(runner[name]), sorted(upstream[name])) for name in shared if {v for v, k in runner[name] if k != "path"} - {v for v, k in upstream[name]}]
print(f"runner packages {len(runner)}, upstream {len(upstream)}, shared names {len(shared)}, runner-only {len(set(runner) - set(upstream))}")
print("runner-only:", " ".join(sorted(set(runner) - set(upstream))) or "none")
print(f"shared with a version upstream does not lock: {len(differ)}")
for name, r, u in differ:
    print(f"  {name}: runner {r} upstream {u}")
sys.exit(1 if differ else 0)
