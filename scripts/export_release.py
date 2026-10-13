"""Create a reproducible source-only ZIP after scanning every included file."""

import hashlib
import json
import sys
import zipfile
from pathlib import Path

from check_release import ROOT, scan


def main():
    files, findings = scan()
    if findings:
        print(json.dumps({"passed": False, "findings": findings}, ensure_ascii=False, indent=2))
        return 1
    destination = ROOT / "dist/chatlens-0.1.0-source.zip"
    destination.parent.mkdir(exist_ok=True)
    manifest = []
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            content = path.read_bytes()
            info = zipfile.ZipInfo("chatlens/" + relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content)
            manifest.append({"path": relative, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
        info = zipfile.ZipInfo("chatlens/SOURCE_MANIFEST.json", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        archive.writestr(info, json.dumps(manifest, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {
                "passed": True,
                "archive": "dist/" + destination.name,
                "files": len(files),
                "bytes": destination.stat().st_size,
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
