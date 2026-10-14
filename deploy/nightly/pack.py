"""Pack the nightly's modules into one small archive for the Nebius serverless container form.

The form uploads files one by one and caps the whole request; the five modules in the clear (37 KiB) went
over it on Oct 12 ("request entity too large"). Gzipped together they are a quarter of that and one upload.
Run from the repo root: `python deploy/nightly/pack.py`. Writes `data/kin-nightly.tgz` (gitignored); the
container's entrypoint unpacks it to /app/kin. Re-run and re-upload after changing any of the modules.
"""

import io
import pathlib
import tarfile

MODULES = ("__init__.py", "nightly.py", "consolidate.py", "llm.py", "config.py")  # not memory.py: never imported at night

root = pathlib.Path(__file__).resolve().parents[2]
out = root / "data" / "kin-nightly.tgz"
out.parent.mkdir(exist_ok=True)
with tarfile.open(out, "w:gz") as tar:
    for name in MODULES:
        path = root / "kin" / name
        info = tarfile.TarInfo(f"kin/{name}")
        data = path.read_bytes()
        info.size, info.mtime = len(data), 0  # a reproducible archive: same sources, same bytes
        tar.addfile(info, io.BytesIO(data))
plain = sum((root / "kin" / n).stat().st_size for n in MODULES)
print(f"{out} ({out.stat().st_size / 1024:.1f} KiB, from {plain / 1024:.1f} KiB of source)")
