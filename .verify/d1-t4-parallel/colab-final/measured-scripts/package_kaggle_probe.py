#!/usr/bin/env python3
"""Package sources and a verified small real-token bundle as a private Kaggle job.

Creates files locally; never authenticates, uploads, or starts a runtime.
"""

import argparse
import base64
import io
import json
from pathlib import Path
import zipfile

if not __package__:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_diffusion import load_manifest, bundle_vocab, bundle_provenance


def package(bundle, output, owner, slug="d1-two-t4-bounded-probe"):
    bundle, output = Path(bundle), Path(output)
    meta = load_manifest(bundle, verify_hashes=True)
    bundle_provenance(bundle, meta)
    if (
        meta["seq_len"] != 1024
        or bundle_vocab(bundle) != 32000
        or meta["splits"]["train"]["tokens"] > 262144
        or meta["splits"]["validation"]["tokens"] > 65536
    ):
        raise ValueError(
            "use build_probe_bundle.py to create a small context-1024 / vocab-32000 subset"
        )
    if (
        not owner
        or "/" in owner
        or not slug
        or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in slug)
    ):
        raise ValueError("expected Kaggle owner and a lowercase slug")
    output.mkdir(parents=True, exist_ok=False)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for source in sorted(Path(__file__).parent.glob("*.py")):
            archive.write(source, "scripts/" + source.name)
        # Only the bundle's expected files; never capture arbitrary nearby files.
        files = {"manifest.json", "tokenizer_manifest.json", "tokenizer.json"}
        for split in meta["splits"].values():
            files.update(shard["path"] for shard in split["shards"])
        for name in sorted(files):
            path = (bundle / name).resolve()
            if not path.is_relative_to(bundle.resolve()):
                raise ValueError("bundle path escapes its directory")
            archive.write(path, "probe-bundle/" + name)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    bootstrap = (
        "from pathlib import Path\nimport io,base64,zipfile,runpy\n"
        "root=Path('/kaggle/working/d1-opt');root.mkdir(exist_ok=True)\n"
        f"zipfile.ZipFile(io.BytesIO(base64.b64decode({encoded!r}))).extractall(root/'src')\n"
        "runpy.run_path(str(root/'src/scripts/kaggle_t4_probe.py'),run_name='__main__')\n"
    )
    (output / "benchmark.py").write_text(bootstrap)
    (output / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": f"{owner}/{slug}",
                "title": slug.replace("-", " "),
                "code_file": "benchmark.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": True,
                "enable_internet": False,
                "machine_shape": "NvidiaTeslaT4",
                "dataset_sources": [],
                "competition_sources": [],
                "kernel_sources": [],
            },
            indent=2,
        )
        + "\n"
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--slug", default="d1-two-t4-bounded-probe")
    args = parser.parse_args()
    print(package(args.bundle, args.out, args.owner, args.slug))
