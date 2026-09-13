$ErrorActionPreference = "Continue"
$root = "C:\Users\fengrru\Desktop\maestro-v3.0.0"
$base = "https://hf-mirror.com/datasets/ddPn08/maestro-v3.0.0/resolve/main/"
$listPath = "C:\Users\fengrru\Desktop\新建文件夹 (2)\datasets\maestro_subset\download_list.txt"
$log = "C:\Users\fengrru\Desktop\新建文件夹 (2)\datasets\maestro_subset\download.log"
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
