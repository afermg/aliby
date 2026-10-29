"""Acquire one preselected official model into a private trial cache, with limits."""

import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import requests

root = Path(sys.argv[1])
catalogue = json.loads((root / "evidence/official-modelsets.json").read_text())
model = "yeast-alcatras-brightfield-sCMOS-60x-1z"
default = "yeast-alcatras-brightfield-EMCCD-60x-5z"
cache = root / "models"
cache.mkdir()
(cache / "shared").mkdir()
(cache / default).mkdir()
# Preserve the installed defaults verbatim: BABY's published identity records
# inherited tracking defaults explicitly, without substituting another CNN.
source = Path.home() / ".baby_models"
shutil.copy2(source / default / "modelset.json", cache / default / "modelset.json")
(cache / model).mkdir()
shutil.copy2(root / "evidence/official-modelsets.json", cache / "modelsets.json")
spec = catalogue["models"][model]
required = {v for k, v in spec["brain_params"].items() if k.endswith("_file")}
required.add(spec["brain_params"]["params"])
defaults = json.loads((cache / default / "modelset.json").read_text())["brain_params"]
required.add(defaults["celltrack_backup_model_file"])
required.add("modelset.json")
receipts = []
total = 0
for filename in sorted(required):
    assert Path(filename).name == filename
    section = model if filename in spec["files"] else "shared"
    assert section == model or filename in catalogue["shared"]
    url = f"https://julianpietsch.github.io/baby-models/{section}/{filename}"
    target = cache / section / filename
    started = time.monotonic()
    with requests.get(url, timeout=(10, 60), stream=True) as response:
        response.raise_for_status()
        with target.open("xb") as handle:
            for block in response.iter_content(1024 * 1024):
                total += len(block)
                if total > 256 * 1024**2 or time.monotonic() - started > 60:
                    raise RuntimeError("Private model download budget exceeded")
                handle.write(block)
    receipts.append(
        {
            "url": url,
            "path": str(target),
            "size": target.stat().st_size,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }
    )
    (root / "evidence/model-downloads.json").write_text(
        json.dumps(receipts, indent=2) + "\n"
    )
    print(filename, target.stat().st_size, flush=True)
actual = json.loads((cache / model / "modelset.json").read_text())
assert actual["brain_params"] == spec["brain_params"]
assert actual["meta"] == spec["meta"]
print("Private model assets complete:", total, "bytes")
