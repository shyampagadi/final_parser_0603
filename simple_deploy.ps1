Write-Host "=========================================================="
Write-Host "     Simple Lambda Deployment Package Creator          "
Write-Host "=========================================================="

# Step 1: Create directories
$deployDir = ".\lambda_deploy_new"
$tempDir = ".\lambda_deploy_temp"

Write-Host "Creating directories..."
New-Item -Path $deployDir -ItemType Directory -Force | Out-Null
New-Item -Path $tempDir -ItemType Directory -Force | Out-Null

# Step 2: Copy lambda function from the correct location
Write-Host "Copying Lambda function code..."
if (Test-Path ".\resume_parser_NEW_FINAL_with_docs\lambda_function.py") {
    Copy-Item ".\resume_parser_NEW_FINAL_with_docs\lambda_function.py" $tempDir
    Write-Host "Lambda function copied from resume_parser_NEW_FINAL_with_docs folder" -ForegroundColor Green
} else {
    Write-Host "ERROR: lambda_function.py not found in expected location!" -ForegroundColor Red
    exit
}

# Step 3: Install dependencies directly to temp directory
Write-Host "Installing dependencies (this may take a few minutes)..."
Set-Location -Path $tempDir

Write-Host "Installing opensearch-py..."
python -m pip install --target . opensearch-py

Write-Host "Installing requests_aws4auth..."
python -m pip install --target . requests_aws4auth

# boto3 is pre-installed in AWS Lambda, no need to include it
Write-Host "Note: boto3 is pre-installed in AWS Lambda environment, skipping installation" -ForegroundColor Cyan

# Step 4: Create ZIP file
Write-Host "Creating deployment package..."
Set-Location ..

# Check if zip file already exists and remove it
$zipFilePath = "$deployDir\lambda_deployment_v5.zip"
if (Test-Path $zipFilePath) {
    Write-Host "Existing deployment package found. Removing..." -ForegroundColor Yellow
    Remove-Item -Path $zipFilePath -Force
    Write-Host "Existing package removed successfully." -ForegroundColor Green
}

Write-Host "Creating new deployment package..."
Compress-Archive -Path "$tempDir\*" -DestinationPath $zipFilePath -Force

# Step 5: Clean up temp directory
Write-Host "Cleaning up..."
Remove-Item -Path $tempDir -Recurse -Force

# Step 6: Success message
$zipPath = Resolve-Path "$deployDir\lambda_deployment_v5.zip"
$zipSize = (Get-Item $zipPath).Length / 1MB

Write-Host "=========================================================="
Write-Host "Deployment package created successfully!" -ForegroundColor Green
Write-Host "Location: $zipPath"
Write-Host "Size: $([Math]::Round($zipSize, 2)) MB"
Write-Host ""
Write-Host "Next steps:"
Write-Host "1. Open the AWS Lambda console"
Write-Host "2. Select your function"
Write-Host "3. Go to the 'Code' tab"
Write-Host "4. Click 'Upload from' dropdown"
Write-Host "5. Select '.zip file'"
Write-Host "6. Choose the deployment package you just created"
Write-Host "7. Click 'Save' or 'Deploy'"
Write-Host "==========================================================" 