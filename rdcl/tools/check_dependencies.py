from __future__ import annotations

import argparse
import importlib
import shutil
import sys
from importlib import metadata
from pathlib import Path


def _version(dist_names):
    for name in dist_names:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return None


def check_import(import_name: str, dist_names: list[str] | None = None) -> bool:
    dist_names = dist_names or [import_name]
    try:
        mod = importlib.import_module(import_name)
        ver = getattr(mod, "__version__", None) or _version(dist_names)
        print(f"[OK] import {import_name:12s} version={ver} file={getattr(mod, '__file__', None)}")
        return True
    except Exception as exc:
        print(f"[FAIL] import {import_name:12s} error={exc!r}")
        return False


def check_cmd(name: str) -> bool:
    path = shutil.which(name)
    if path is None:
        print(f"[MISS] command {name}")
        return False
    print(f"[OK] command {name:10s} path={path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Check RDCL runtime dependencies.")
    parser.add_argument("--strict", action="store_true", help="Treat raw/preprocess dependencies as required.")
    args = parser.parse_args()

    print("Python:", sys.executable)
    print("Version:", sys.version.replace("\n", " "))
    print("Project:", Path.cwd())

    base = [("torch", ["torch"]), ("numpy", ["numpy"]), ("pandas", ["pandas"]), ("yaml", ["PyYAML"]), ("sklearn", ["scikit-learn"]), ("scipy", ["scipy"]), ("tqdm", ["tqdm"])]
    raw = [("rdkit", ["rdkit"]), ("Bio", ["biopython"]), ("esm", ["fair-esm", "esm"])]
    strict = [("plip", ["plip"]), ("openbabel", ["openbabel", "openbabel-wheel"])]

    ok = True
    print("\nBase dependencies:")
    for imp, dist in base:
        ok = check_import(imp, dist) and ok

    print("\nRaw inference dependencies:")
    for imp, dist in raw:
        success = check_import(imp, dist)
        ok = (success and ok) if args.strict else ok

    print("\nStrict cache-building dependencies:")
    for imp, dist in strict:
        success = check_import(imp, dist)
        ok = (success and ok) if args.strict else ok

    print("\nCommands:")
    for cmd in ["python", "pip", "torchrun", "nvidia-smi", "plip", "obabel"]:
        check_cmd(cmd)

    try:
        import torch
        print("\nTorch/CUDA:")
        print("torch:", torch.__version__)
        print("cuda:", torch.version.cuda)
        print("cuda available:", torch.cuda.is_available())
        print("device count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    except Exception as exc:
        print("Torch CUDA check failed:", repr(exc))

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
