@echo off
echo ========================================================
echo Pushing Microscope Image Classification to GitHub
echo Repository: https://github.com/pavithran22-ops/microscope-image-classification
echo ========================================================
git push -u origin main
if %errorlevel% neq 0 (
    echo.
    echo If GitHub requests authentication, you can also authenticate via:
    echo 1. Browser login window (Git Credential Manager)
    echo 2. Or using a Personal Access Token (PAT):
    echo    git push https://^<YOUR_GITHUB_TOKEN^>@github.com/pavithran22-ops/microscope-image-classification.git main
) else (
    echo.
    echo [SUCCESS] Successfully pushed all files and results to GitHub!
)
pause
