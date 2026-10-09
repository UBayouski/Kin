<#
.SYNOPSIS
    Run everything Kin needs on the hub: wire-pod, the escapepod.local helper, and the Kin service.

.DESCRIPTION
    Meant to run in its own console, or as a scheduled task at logon, so nothing depends on a tool
    session (Kin was silently down for three days when it did). The Kin service is restarted if it exits.
    Output goes to data\logs\kin-<date>.log; the console shows the same lines.

    .\scripts\hub.ps1               start everything and keep running (Ctrl+C stops all of it)
    .\scripts\hub.ps1 -Status       what is running, and the last log lines
    .\scripts\hub.ps1 -Install      register a scheduled task that runs this at logon
    .\scripts\hub.ps1 -Uninstall    remove that task

    Settings come from .env as always. The wire-pod location and the robot's IP (for the mDNS helper's
    unicast answer) are found automatically; override with -WirePod and -VectorIp.
#>
[CmdletBinding()]
param(
    [switch]$Status,
    [switch]$Install,
    [switch]$Uninstall,
    [string]$WirePod = "C:\Program Files\wire-pod\chipper\chipper.exe",
    [string]$VectorIp = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$logDir = Join-Path $root "data\logs"
$taskName = "Kin hub"

function Say($text) { Write-Host ("{0} hub: {1}" -f (Get-Date -Format "HH:mm:ss"), $text) }

function Get-KinProcesses {
    Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "kin\.service|escapepod_mdns" }
}

function Get-VectorIp {
    if ($VectorIp) { return $VectorIp }
    $config = Join-Path $env:USERPROFILE ".anki_vector\sdk_config.ini"
    if (Test-Path $config) {
        $match = Select-String -Path $config -Pattern "^\s*ip\s*=\s*(\S+)" | Select-Object -First 1
        if ($match) { return $match.Matches[0].Groups[1].Value }
    }
    return ""
}

function Ensure-WirePod {
    if (Get-Process -Name "chipper" -ErrorAction SilentlyContinue) { Say "wire-pod is running"; return }
    if (-not (Test-Path $WirePod)) { throw "wire-pod not found at $WirePod (pass -WirePod)" }
    Say "starting wire-pod"
    Start-Process -FilePath $WirePod -ArgumentList "-d" -WorkingDirectory (Split-Path -Parent $WirePod) -WindowStyle Minimized
    foreach ($i in 1..60) {
        try { Invoke-WebRequest -UseBasicParsing "http://localhost:2427/api/get_kg_api" -TimeoutSec 2 | Out-Null; Say "wire-pod is up"; return }
        catch { Start-Sleep -Seconds 1 }
    }
    throw "wire-pod did not answer on port 2427 within a minute"
}

function Start-Mdns {
    $running = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "escapepod_mdns" }
    if ($running) { Say "mDNS helper is running"; return $null }
    $args = @("-u", (Join-Path $root "scripts\escapepod_mdns.py"))
    $ip = Get-VectorIp
    if ($ip) { $args += @("--unicast", $ip) }
    Say ("starting the escapepod.local helper" + $(if ($ip) { " (unicast to $ip)" } else { "" }))
    return Start-Process -FilePath $python -ArgumentList $args -WorkingDirectory $root -WindowStyle Hidden -PassThru
}

function Show-Status {
    "wire-pod:      " + $(if (Get-Process -Name "chipper" -ErrorAction SilentlyContinue) { "running" } else { "not running" })
    $procs = Get-KinProcesses
    "mDNS helper:   " + $(if ($procs | Where-Object { $_.CommandLine -match "escapepod_mdns" }) { "running" } else { "not running" })
    "Kin service:   " + $(if ($procs | Where-Object { $_.CommandLine -match "kin\.service" }) { "running" } else { "not running" })
    $task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    "logon task:    " + $(if ($task) { $task.State } else { "not installed (-Install)" })
    $log = Get-ChildItem $logDir -Filter "kin-*.log" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1
    if ($log) { ""; "last lines of $($log.Name):"; Get-Content $log.FullName -Tail 8 }
}

function Install-Task {
    $action = New-ScheduledTaskAction -Execute "powershell.exe" `
        -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`"" -WorkingDirectory $root
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
    $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 3 `
        -RestartInterval (New-TimeSpan -Minutes 1) -StartWhenAvailable -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries
    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
    Say "scheduled task '$taskName' installed: runs at logon, restarts itself if it fails"
    Say "start it now with: Start-ScheduledTask -TaskName '$taskName'"
}

if ($Status) { Show-Status; return }
if ($Uninstall) { Unregister-ScheduledTask -TaskName $taskName -Confirm:$false; Say "task removed"; return }
if ($Install) { Install-Task; return }

if (-not (Test-Path $python)) { throw "no virtual environment at $python (python -m venv .venv; pip install -e .)" }
if (Get-KinProcesses | Where-Object { $_.CommandLine -match "kin\.service" }) {
    throw "a Kin service is already running; stop it first (or use -Status)"
}
New-Item -ItemType Directory -Force $logDir | Out-Null
Set-Location $root

$mdns = $null
try {
    Ensure-WirePod
    $mdns = Start-Mdns
    while ($true) {
        $log = Join-Path $logDir ("kin-{0}.log" -f (Get-Date -Format "yyyy-MM-dd"))
        Say "starting the Kin service (log: $log)"
        # Windows PowerShell 5.1 wraps a native program's stderr in error records; going through cmd
        # keeps the two streams as plain text. Each line is shown here and appended to the log as UTF-8
        # (Tee-Object in 5.1 writes UTF-16, which grep and git tools read as binary).
        # The script runs with ErrorActionPreference=Stop, so a hiccup in this pipeline (the log file briefly
        # locked, say) would otherwise end the hub without a word: 'stopped' is printed before PowerShell
        # shows the error. The hub exists to keep Kin alive, so report it here and start again instead.
        try {
            & cmd.exe /c "`"$python`" -u -m kin.service 2>&1" | ForEach-Object { $_; Add-Content -Path $log -Value $_ -Encoding UTF8 }
            Say "the Kin service exited; starting it again in 15 s (Ctrl+C to stop)"
        }
        catch {
            Say ("error while running the Kin service: " + $_.Exception.Message + " (at " + $_.InvocationInfo.ScriptLineNumber + "); starting it again in 15 s")
        }
        Start-Sleep -Seconds 15
    }
}
finally {
    if ($mdns -and -not $mdns.HasExited) { Stop-Process -Id $mdns.Id -Force -ErrorAction SilentlyContinue }
    Say "stopped"
}
