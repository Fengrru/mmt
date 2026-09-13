"""Generate a download list and a resumable PowerShell downloader for the pilot.

The default mirror `ddPn08/maestro-v3.0.0` stores extracted `.wav` files that
match MAESTRO's official `metadata.csv` paths, so no extension rewriting is
needed. This script only writes files; it does not download anything.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

DEFAULT_BASE = "https://hf-mirror.com/datasets/ddPn08/maestro-v3.0.0/resolve/main/"

PS_TEMPLATE = r"""$ErrorActionPreference = "Continue"
$root = "__ROOT__"
$base = "__BASE__"
$listPath = "__LIST__"
$log = "__LOG__"
$files = Get-Content -LiteralPath $listPath | Where-Object { $_.Trim() -ne "" }
$i = 0
foreach ($rel in $files) {
    $i++
    $rel = $rel.Trim()
    $dest = Join-Path $root $rel
    $dir = Split-Path $dest -Parent
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    if ((Test-Path $dest) -and ((Get-Item $dest).Length -gt 4096)) {
        Write-Output "[$i/$($files.Count)] skip $rel"
        continue
    }
    Write-Output "[$i/$($files.Count)] get  $rel"
    & curl.exe -sS -L --fail --retry 5 --retry-delay 3 -C - --connect-timeout 30 -o $dest ($base + $rel)
    if ($LASTEXITCODE -ne 0) { Write-Output "[$i/$($files.Count)] FAIL $rel" }
}
Write-Output "DONE"
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate pilot download list + script")
    ap.add_argument("--metadata", default="datasets/maestro_subset/metadata.csv")
    ap.add_argument("--out", default="datasets/maestro_subset")
    ap.add_argument("--root", default=r"C:\Users\fengrru\Desktop\maestro-v3.0.0")
    ap.add_argument("--base-url", default=DEFAULT_BASE)
    ap.add_argument("--filename-column", default="audio_filename")
    ap.add_argument(
        "--swap-ext",
        default=None,
        help="optional: rewrite extension, e.g. '.flac' to download the flac tree",
    )
    args = ap.parse_args(argv)

    meta = Path(args.metadata)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with meta.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rels = []
    for row in rows:
        name = row[args.filename_column]
        if args.swap_ext:
            name = str(Path(name).with_suffix(args.swap_ext))
        rels.append(name)

    list_path = out / "download_list.txt"
    list_path.write_text("\n".join(rels) + "\n", encoding="utf-8")

    script = (
        PS_TEMPLATE.replace("__ROOT__", args.root)
        .replace("__BASE__", args.base_url)
        .replace("__LIST__", str(list_path.resolve()))
        .replace("__LOG__", str((out / "download.log").resolve()))
    )
    script_path = out / "download_pilot.ps1"
    script_path.write_text(script, encoding="utf-8-sig")

    print(f"files={len(rels)}")
    print(f"list:   {list_path}")
    print(f"script: {script_path}")
    print(f"root:   {args.root}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    raise SystemExit(main())
