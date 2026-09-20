# Registers a Windows scheduled task that starts Kin whenever you log in. Run once; no admin needed.
# It runs in your user session on purpose: the microphone and speaker are only available there.
#
#   powershell -ExecutionPolicy Bypass -File scripts\windows\install-autostart.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\windows\install-autostart.ps1 -Remove

param([switch]$Remove)

$taskName = "Kin"
if ($Remove) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Write-Output "Removed the '$taskName' task."
    return
}

$script = Join-Path $PSScriptRoot "start-kin.ps1"
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$script`""
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero)

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
Write-Output "Registered the '$taskName' task: Kin starts at every login. Start it now with start-kin.ps1."
