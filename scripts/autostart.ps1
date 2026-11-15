# Runs the service in the background (started at Windows logon by the "RAG service 8003" scheduled task).
# Waits for the database (Docker Desktop starts at logon too), starts uvicorn, and restarts it if it stops.
# Output goes to logs\service.log next to this folder.
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
$logs = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$log = Join-Path $logs 'service.log'
$python = Join-Path $root '.venv\Scripts\python.exe'

function Write-Log($text) {
    Add-Content -Path $log -Encoding Ascii -Value ("[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $text)
}

function Test-Port($port) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $client.Connect('127.0.0.1', $port)
        return $true
    } catch {
        return $false
    } finally {
        $client.Close()
    }
}

Set-Location $root
while ($true) {
    if (Test-Port 8003) {
        # Already running (for example started by hand with run.bat): check again later.
        Start-Sleep -Seconds 60
        continue
    }
    $waited = 0
    while (-not (Test-Port 5433)) {
        if ($waited % 60 -eq 0) { Write-Log 'waiting for the database on port 5433 (Docker Desktop)' }
        Start-Sleep -Seconds 5
        $waited += 5
    }
    Write-Log 'starting the service on http://127.0.0.1:8003'
    # Through cmd so the log stays plain text (PowerShell 5.1 would write UTF-16 and wrap stderr lines).
    cmd /c "`"$python`" -m uvicorn app.main:app --host 127.0.0.1 --port 8003 >> `"$log`" 2>&1"
    Write-Log "the service stopped (exit code $LASTEXITCODE); restarting in 10 seconds"
    Start-Sleep -Seconds 10
}
