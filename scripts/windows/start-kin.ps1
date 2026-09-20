# Starts Kin in the background on Windows: the mDNS helper (so Vector can find wire-pod) and the service.
# Logs go to data\logs. Safe to run twice: anything already running is left alone.
#
#   powershell -ExecutionPolicy Bypass -File scripts\windows\start-kin.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\windows\start-kin.ps1 -Stop

param([switch]$Stop)

$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$logs = Join-Path $root "data\logs"
New-Item -ItemType Directory -Force $logs | Out-Null

$parts = @(
    @{ Name = "mdns";    Match = "escapepod_mdns"; Args = @("-u", "scripts\escapepod_mdns.py") },
    @{ Name = "service"; Match = "kin.service";    Args = @("-u", "-m", "kin.service") }
)

foreach ($part in $parts) {
    $running = Get-CimInstance Win32_Process -Filter "Name like 'python%'" |
        Where-Object { $_.CommandLine -match [regex]::Escape($part.Match) }

    if ($Stop) {
        $running | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -Confirm:$false }
        Write-Output "$($part.Name): stopped"
        continue
    }
    if ($running) {
        Write-Output "$($part.Name): already running"
        continue
    }
    $log = Join-Path $logs "$($part.Name).log"
    Start-Process -FilePath $python -ArgumentList $part.Args -WorkingDirectory $root -WindowStyle Hidden `
        -RedirectStandardOutput $log -RedirectStandardError (Join-Path $logs "$($part.Name).err.log")
    Write-Output "$($part.Name): started, log at $log"
}
