# for PowerShell in Windows
# Version 0.9
# Fabian
# 2408011808

# Set execution policy for this session (if needed)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

# Function to run docker compose command
function Invoke-DockerCompose {
    param (
        [Parameter(Mandatory=$true, Position=0)]
        [string]$Command,
        [string[]]$Arguments
    )

    # Try docker compose (newer versions)
    if (Get-Command "docker" -ErrorAction SilentlyContinue) {
        $composeArgs = @("compose", $Command)  # Start with the base command
        $composeArgs += $Arguments               # Append the additional arguments
        Write-Host "Running: docker compose $Command $Arguments"
        & docker $composeArgs
    }
    # Try docker-compose (older versions)
    elseif (Get-Command "docker-compose" -ErrorAction SilentlyContinue) {
        $composeArgs = @($Command) + $Arguments  # Start with the command and append arguments
        Write-Host "Running: docker-compose $Command $Arguments"
        & docker-compose $composeArgs
    }
    else {
        Write-Host "Neither 'docker compose' nor 'docker-compose' is available. Please install Docker Desktop or Docker Compose CLI."
        exit 1
    }
}

# Login to Docker
# Prompt for Docker Hub credentials
$dockerUsername = Read-Host -Prompt "Enter your Docker Hub username"
$dockerPassword = Read-Host -Prompt "Enter your Docker Hub password" -AsSecureString

# Convert the secure string to plain text (for login)
$dockerPasswordPlain = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($dockerPassword))

# Perform Docker login
Write-Host "Logging in to Docker..."
$loginResult = docker login --username $dockerUsername --password $dockerPasswordPlain

# Check if login was successful
if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker login failed. Please check your credentials."
    exit 1
} else {
    Write-Host "Docker login successful."
}

# Set up environment variables
$env:AIRFLOW_UID = [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$env:AIRFLOW_GID = "0"
Set-Content .env "AIRFLOW_UID=$env:AIRFLOW_UID`nAIRFLOW_GID=0"

# Ensure we're in the correct directory
Set-Location -Path "C:\Users\itflo\Documents\DataScientestFF\JUL24_BMLOps_Stock_market"

# Install Python dependencies
Write-Host "Installing Python dependencies..."
pip install -r .\app\requirements.txt

# Stop any running Docker containers
Write-Host "Stopping any running Docker containers..."
Invoke-DockerCompose down

# Initialize Airflow
Write-Host "Initializing Airflow..."
# Invoke-DockerCompose "up" "airflow-init"
Invoke-DockerCompose "up" @("airflow-init")

# Start Docker containers
Write-Host "Starting Docker containers..."
# Invoke-DockerCompose "up" "-d"
Invoke-DockerCompose "up" @("-d")

# Set PYTHONPATH to include the app directory
$env:PYTHONPATH = ".\app"

# Start FastAPI application
Write-Host "Starting FastAPI application..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "uvicorn app.main:app --reload"

# Display logs for the dashboard service (optional)
Write-Host "Displaying logs for the dashboard service..."
# Invoke-DockerCompose "logs" "dashboard"
Invoke-DockerCompose "logs" @("dashboard")

# Reminder to update pip
Write-Host "Remember to update pip by running: python.exe -m pip install --upgrade pip"