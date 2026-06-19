# Terminal-Only Dev Verification

[← Back to README](../README.md)

This flow uses the project from the terminal in dev mode — it sets up dependencies, starts the FastAPI backend, checks the backend, then opens the Electron dev shell without creating a native installer/package. The repo exposes `backend:dev` and `frontend:dev` for this.

### macOS — Standard terminal-only test

<details>
<summary>Show clone → setup → build → launch commands</summary>

```bash
cd "$HOME/Downloads" # Go to the Downloads folder
if [ -d "DepthLensPro/.git" ]; then # Check if DepthLensPro is already cloned
cd "DepthLensPro" # Enter the existing project folder
git checkout main # Switch to the main branch
git pull --ff-only # Update the repo without creating merge commits
elif [ -e "DepthLensPro" ]; then # Check if a non-git folder with the same name exists
echo "DepthLensPro exists but is not a Git repo; rename/delete it first." # Warn without overwriting files
exit 1 # Stop safely
else # Run this if the repo is not downloaded yet
git clone https://github.com/AyushmanRaha/DepthLensPro.git "DepthLensPro" # Clone the project into Downloads/DepthLensPro
cd "DepthLensPro" # Enter the cloned project folder
fi # Finish project folder setup
npm run setup:mac # Install macOS dependencies and standard PyTorch model cache
npm run verify:resources # Verify standard resources
npm run backend:dev & # Start the FastAPI backend from terminal
BACKEND_PID=$! # Store the backend process ID
sleep 5 # Give the backend time to start
curl http://127.0.0.1:8765/live # Check that the backend is live
npm run frontend:dev # Open the dev app without building a native package
kill "$BACKEND_PID" 2>/dev/null || true # Stop the backend after closing the dev app
```

</details>

### macOS — ONNX terminal-only test

<details>
<summary>Show clone → setup → build → launch commands</summary>

```bash
cd "$HOME/Downloads" # Go to the Downloads folder
if [ -d "DepthLensPro/.git" ]; then # Check if DepthLensPro is already cloned
cd "DepthLensPro" # Enter the existing project folder
git checkout main # Switch to the main branch
git pull --ff-only # Update the repo without creating merge commits
elif [ -e "DepthLensPro" ]; then # Check if a non-git folder with the same name exists
echo "DepthLensPro exists but is not a Git repo; rename/delete it first." # Warn without overwriting files
exit 1 # Stop safely
else # Run this if the repo is not downloaded yet
git clone https://github.com/AyushmanRaha/DepthLensPro.git "DepthLensPro" # Clone the project into Downloads/DepthLensPro
cd "DepthLensPro" # Enter the cloned project folder
fi # Finish project folder setup
npm run setup:mac:onnx # Install macOS dependencies and generate/validate all ONNX models
npm run verify:onnx:required # Verify that all required ONNX models exist
npm run backend:dev & # Start the FastAPI backend from terminal
BACKEND_PID=$! # Store the backend process ID
sleep 5 # Give the backend time to start
curl http://127.0.0.1:8765/onnx/status # Check ONNX model/provider status
npm run frontend:dev # Open the dev app without building a native package
kill "$BACKEND_PID" 2>/dev/null || true # Stop the backend after closing the dev app
```

</details>

### Windows ARM64 — Standard terminal-only test

<details>
<summary>Show clone → setup → build → launch commands</summary>

```powershell
cd "$HOME\Downloads" # Go to the Downloads folder
if (Test-Path "DepthLensPro\.git") { # Check if DepthLensPro is already cloned
cd "DepthLensPro" # Enter the existing project folder
git checkout main # Switch to the main branch
git pull --ff-only # Update the repo without creating merge commits
} elseif (Test-Path "DepthLensPro") { # Check if a non-git folder with the same name exists
Write-Error "DepthLensPro exists but is not a Git repo; rename/delete it first." # Warn without overwriting files
exit 1 # Stop safely
} else { # Run this if the repo is not downloaded yet
git clone https://github.com/AyushmanRaha/DepthLensPro.git "DepthLensPro" # Clone the project into Downloads\DepthLensPro
cd "DepthLensPro" # Enter the cloned project folder
} # Finish project folder setup
npm run setup:win # Install Windows dependencies and standard PyTorch model cache
npm run verify:resources # Verify standard resources
$backend = Start-Process -FilePath "npm.cmd" -ArgumentList "run backend:dev" -PassThru # Start the FastAPI backend from terminal
Start-Sleep -Seconds 5 # Give the backend time to start
Invoke-RestMethod http://127.0.0.1:8765/live # Check that the backend is live
npm run frontend:dev # Open the dev app without building a native package
Stop-Process -Id $backend.Id -Force # Stop the backend after closing the dev app
```

</details>

### Windows ARM64 — ONNX terminal-only test

<details>
<summary>Show clone → setup → build → launch commands</summary>

```powershell
cd "$HOME\Downloads" # Go to the Downloads folder
if (Test-Path "DepthLensPro\.git") { # Check if DepthLensPro is already cloned
cd "DepthLensPro" # Enter the existing project folder
git checkout main # Switch to the main branch
git pull --ff-only # Update the repo without creating merge commits
} elseif (Test-Path "DepthLensPro") { # Check if a non-git folder with the same name exists
Write-Error "DepthLensPro exists but is not a Git repo; rename/delete it first." # Warn without overwriting files
exit 1 # Stop safely
} else { # Run this if the repo is not downloaded yet
git clone https://github.com/AyushmanRaha/DepthLensPro.git "DepthLensPro" # Clone the project into Downloads\DepthLensPro
cd "DepthLensPro" # Enter the cloned project folder
} # Finish project folder setup
npm run setup:win:onnx # Install Windows dependencies and generate/validate all ONNX models
npm run verify:onnx:required # Verify that all required ONNX models exist
$backend = Start-Process -FilePath "npm.cmd" -ArgumentList "run backend:dev" -PassThru # Start the FastAPI backend from terminal
Start-Sleep -Seconds 5 # Give the backend time to start
Invoke-RestMethod http://127.0.0.1:8765/onnx/status # Check ONNX model/provider status
npm run frontend:dev # Open the dev app without building a native package
Stop-Process -Id $backend.Id -Force # Stop the backend after closing the dev app
```

</details>

### Windows x86/x64 — Standard terminal-only test

<details>
<summary>Show clone → setup → build → launch commands</summary>

```powershell
cd "$HOME\Downloads" # Go to the Downloads folder
if (Test-Path "DepthLensPro\.git") { # Check if DepthLensPro is already cloned
cd "DepthLensPro" # Enter the existing project folder
git checkout main # Switch to the main branch
git pull --ff-only # Update the repo without creating merge commits
} elseif (Test-Path "DepthLensPro") { # Check if a non-git folder with the same name exists
Write-Error "DepthLensPro exists but is not a Git repo; rename/delete it first." # Warn without overwriting files
exit 1 # Stop safely
} else { # Run this if the repo is not downloaded yet
git clone https://github.com/AyushmanRaha/DepthLensPro.git "DepthLensPro" # Clone the project into Downloads\DepthLensPro
cd "DepthLensPro" # Enter the cloned project folder
} # Finish project folder setup
npm run setup:win # Install Windows x64/x86_64 dependencies and standard PyTorch model cache
npm run verify:resources # Verify standard resources
$backend = Start-Process -FilePath "npm.cmd" -ArgumentList "run backend:dev" -PassThru # Start the FastAPI backend from terminal
Start-Sleep -Seconds 5 # Give the backend time to start
Invoke-RestMethod http://127.0.0.1:8765/live # Check that the backend is live
npm run frontend:dev # Open the dev app without building a native package
Stop-Process -Id $backend.Id -Force # Stop the backend after closing the dev app
```

</details>

### Windows x86/x64 — ONNX terminal-only test

<details>
<summary>Show clone → setup → build → launch commands</summary>

```powershell
cd "$HOME\Downloads" # Go to the Downloads folder
if (Test-Path "DepthLensPro\.git") { # Check if DepthLensPro is already cloned
cd "DepthLensPro" # Enter the existing project folder
git checkout main # Switch to the main branch
git pull --ff-only # Update the repo without creating merge commits
} elseif (Test-Path "DepthLensPro") { # Check if a non-git folder with the same name exists
Write-Error "DepthLensPro exists but is not a Git repo; rename/delete it first." # Warn without overwriting files
exit 1 # Stop safely
} else { # Run this if the repo is not downloaded yet
git clone https://github.com/AyushmanRaha/DepthLensPro.git "DepthLensPro" # Clone the project into Downloads\DepthLensPro
cd "DepthLensPro" # Enter the cloned project folder
} # Finish project folder setup
npm run setup:win:onnx # Install Windows x64/x86_64 dependencies and generate/validate all ONNX models
npm run verify:onnx:required # Verify that all required ONNX models exist
$backend = Start-Process -FilePath "npm.cmd" -ArgumentList "run backend:dev" -PassThru # Start the FastAPI backend from terminal
Start-Sleep -Seconds 5 # Give the backend time to start
Invoke-RestMethod http://127.0.0.1:8765/onnx/status # Check ONNX model/provider status
npm run frontend:dev # Open the dev app without building a native package
Stop-Process -Id $backend.Id -Force # Stop the backend after closing the dev app
```

</details>

### Linux ARM64 — Standard terminal-only test

<details>
<summary>Show clone → setup → build → launch commands</summary>

```bash
cd "$HOME/Downloads" # Go to the Downloads folder
if [ -d "DepthLensPro/.git" ]; then # Check if DepthLensPro is already cloned
cd "DepthLensPro" # Enter the existing project folder
git checkout main # Switch to the main branch
git pull --ff-only # Update the repo without creating merge commits
elif [ -e "DepthLensPro" ]; then # Check if a non-git folder with the same name exists
echo "DepthLensPro exists but is not a Git repo; rename/delete it first." # Warn without overwriting files
exit 1 # Stop safely
else # Run this if the repo is not downloaded yet
git clone https://github.com/AyushmanRaha/DepthLensPro.git "DepthLensPro" # Clone the project into Downloads/DepthLensPro
cd "DepthLensPro" # Enter the cloned project folder
fi # Finish project folder setup
npm run setup:linux # Install Linux ARM64 dependencies and standard PyTorch model cache
npm run verify:resources # Verify standard resources
npm run backend:dev & # Start the FastAPI backend from terminal
BACKEND_PID=$! # Store the backend process ID
sleep 5 # Give the backend time to start
curl http://127.0.0.1:8765/live # Check that the backend is live
npm run frontend:dev # Open the dev app without building a native package
kill "$BACKEND_PID" 2>/dev/null || true # Stop the backend after closing the dev app
```

</details>

### Linux ARM64 — ONNX terminal-only test

<details>
<summary>Show clone → setup → build → launch commands</summary>

```bash
cd "$HOME/Downloads" # Go to the Downloads folder
if [ -d "DepthLensPro/.git" ]; then # Check if DepthLensPro is already cloned
cd "DepthLensPro" # Enter the existing project folder
git checkout main # Switch to the main branch
git pull --ff-only # Update the repo without creating merge commits
elif [ -e "DepthLensPro" ]; then # Check if a non-git folder with the same name exists
echo "DepthLensPro exists but is not a Git repo; rename/delete it first." # Warn without overwriting files
exit 1 # Stop safely
else # Run this if the repo is not downloaded yet
git clone https://github.com/AyushmanRaha/DepthLensPro.git "DepthLensPro" # Clone the project into Downloads/DepthLensPro
cd "DepthLensPro" # Enter the cloned project folder
fi # Finish project folder setup
npm run setup:linux:onnx # Install Linux ARM64 dependencies and generate/validate all ONNX models
npm run verify:onnx:required # Verify that all required ONNX models exist
npm run backend:dev & # Start the FastAPI backend from terminal
BACKEND_PID=$! # Store the backend process ID
sleep 5 # Give the backend time to start
curl http://127.0.0.1:8765/onnx/status # Check ONNX model/provider status
npm run frontend:dev # Open the dev app without building a native package
kill "$BACKEND_PID" 2>/dev/null || true # Stop the backend after closing the dev app
```

</details>

### Linux x86/x64 — Standard terminal-only test

<details>
<summary>Show clone → setup → build → launch commands</summary>

```bash
cd "$HOME/Downloads" # Go to the Downloads folder
if [ -d "DepthLensPro/.git" ]; then # Check if DepthLensPro is already cloned
cd "DepthLensPro" # Enter the existing project folder
git checkout main # Switch to the main branch
git pull --ff-only # Update the repo without creating merge commits
elif [ -e "DepthLensPro" ]; then # Check if a non-git folder with the same name exists
echo "DepthLensPro exists but is not a Git repo; rename/delete it first." # Warn without overwriting files
exit 1 # Stop safely
else # Run this if the repo is not downloaded yet
git clone https://github.com/AyushmanRaha/DepthLensPro.git "DepthLensPro" # Clone the project into Downloads/DepthLensPro
cd "DepthLensPro" # Enter the cloned project folder
fi # Finish project folder setup
npm run setup:linux # Install Linux x64/x86_64 dependencies and standard PyTorch model cache
npm run verify:resources # Verify standard resources
npm run backend:dev & # Start the FastAPI backend from terminal
BACKEND_PID=$! # Store the backend process ID
sleep 5 # Give the backend time to start
curl http://127.0.0.1:8765/live # Check that the backend is live
npm run frontend:dev # Open the dev app without building a native package
kill "$BACKEND_PID" 2>/dev/null || true # Stop the backend after closing the dev app
```

</details>

### Linux x86/x64 — ONNX terminal-only test

<details>
<summary>Show clone → setup → build → launch commands</summary>

```bash
cd "$HOME/Downloads" # Go to the Downloads folder
if [ -d "DepthLensPro/.git" ]; then # Check if DepthLensPro is already cloned
cd "DepthLensPro" # Enter the existing project folder
git checkout main # Switch to the main branch
git pull --ff-only # Update the repo without creating merge commits
elif [ -e "DepthLensPro" ]; then # Check if a non-git folder with the same name exists
echo "DepthLensPro exists but is not a Git repo; rename/delete it first." # Warn without overwriting files
exit 1 # Stop safely
else # Run this if the repo is not downloaded yet
git clone https://github.com/AyushmanRaha/DepthLensPro.git "DepthLensPro" # Clone the project into Downloads/DepthLensPro
cd "DepthLensPro" # Enter the cloned project folder
fi # Finish project folder setup
npm run setup:linux:onnx # Install Linux x64/x86_64 dependencies and generate/validate all ONNX models
npm run verify:onnx:required # Verify that all required ONNX models exist
npm run backend:dev & # Start the FastAPI backend from terminal
BACKEND_PID=$! # Store the backend process ID
sleep 5 # Give the backend time to start
curl http://127.0.0.1:8765/onnx/status # Check ONNX model/provider status
npm run frontend:dev # Open the dev app without building a native package
kill "$BACKEND_PID" 2>/dev/null || true # Stop the backend after closing the dev app
```

</details>

<div align="right"><sub><a href="../README.md#depthlens-pro">⬆ back to README</a></sub></div>

---
