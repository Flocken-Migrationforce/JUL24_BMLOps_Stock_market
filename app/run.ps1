# for PowerShell in Windows
# Version 0.6
# Fabian
# 2408011315

# Set execution policy for this session (if needed)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

# Function to run docker compose command
function Invoke-DockerCompose {
    param (
        [Parameter(Mandatory=$true, Position=0)]
        [string[]]$Arguments
    )

    # Try docker compose (newer versions)
    if (Get-Command "docker" -ErrorAction SilentlyContinue) {
        $composeArgs = @("compose") + $Arguments
        & docker $composeArgs
    }
    # Try docker-compose (older versions)
    elseif (Get-Command "docker-compose" -ErrorAction SilentlyContinue) {
        & docker-compose $Arguments
    }
    else {
        Write-Host "Neither 'docker compose' nor 'docker-compose' is available. Please install Docker Desktop or Docker Compose CLI."
        exit 1
    }
}

# Set up environment variables
$env:AIRFLOW_UID = [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$env:AIRFLOW_GID = "0"
Set-Content .env "AIRFLOW_UID=$env:AIRFLOW_UID`nAIRFLOW_GID=0"

# Install Python dependencies
pip install -r .\app\requirements.txt

# Ensure we're in the correct directory
Set-Location -Path "C:\Users\itflo\Documents\DataScientestFF\JUL24_BMLOps_Stock_market"

# Stop any running Docker containers
Invoke-DockerCompose down

# Initialize Airflow
Invoke-DockerCompose up airflow-init

# Start Docker containers
Invoke-DockerCompose up -d

# Set PYTHONPATH to include the app directory
$env:PYTHONPATH = ".\app"

# Start FastAPI application
Start-Process powershell -ArgumentList "-NoExit", "-Command", "uvicorn app.main:app --reload"

# Display logs (optional)
Invoke-DockerCompose logs -f dashboard

# Reminder to update pip
Write-Host "Remember to update pip by running: python.exe -m pip install --upgrade pip"