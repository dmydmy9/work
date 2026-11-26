"""Restore and verify the contest-provided source files required for an RC03 build."""

import argparse
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath


def restore(source, sdk, profile_c):
    manifest = json.loads((source / "SOURCE_HASHES.json").read_bytes())["source"]
    upstream = {"sdk": sdk, "baselines": profile_c / "baselines"}
    restored = {}
    for name, fingerprint in manifest.items():
        relative = PurePosixPath(name)
        if relative.parts[0] not in upstream:
            continue
        target = (source / relative).resolve()
        assert target.is_relative_to(source)
        payload = (upstream[relative.parts[0]] / Path(*relative.parts[1:])).read_bytes()
        if sha256(payload).hexdigest() != fingerprint:
            raise ValueError(f"{name} differs from the contest file used for RC03")
        if target.exists() and target.read_bytes() != payload:
            raise FileExistsError(f"{target} already contains different code")
        restored[target] = payload
    for target, payload in restored.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    print(f"Restored {len(restored)} verified contest files")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sdk", type=Path, help="root of the official SDK 1.0.4 archive")
    parser.add_argument("profile_c", type=Path, help="root of the official C pack")
    args = parser.parse_args()
    restore(Path(__file__).resolve().parent, args.sdk.resolve(), args.profile_c.resolve())


if __name__ == "__main__":
    main()
