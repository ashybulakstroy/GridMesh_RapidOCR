# Install the missing language data for the existing optional E5 recognizer.
# The executable is discovered separately by App/ocr/e5_tesseract.py.
$ErrorActionPreference = 'Stop'
$modelDirectory = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../models/tesseract_kaz/tessdata'))
$targetModel = Join-Path $modelDirectory 'kaz.traineddata'
$expectedHash = 'FCC01EED3815A42B9C6321C4C9D3606F39B166CBF95ADE98B7D8D12063EAE53D'
$sourceUri = 'https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/87416418657359cb625c412a48b6e1d6d41c29bd/kaz.traineddata'
if (Test-Path -LiteralPath $targetModel) {
    if ((Get-FileHash -LiteralPath $targetModel -Algorithm SHA256).Hash -ne $expectedHash) {
        throw 'A different Kazakh model already exists; it was left unchanged.'
    }
    Write-Output "Kazakh model is already verified: $targetModel"
    exit 0
}
$cacheDirectory = Join-Path $env:TEMP 'GridMesh_tesseract_kaz_setup'
New-Item -ItemType Directory -Path $cacheDirectory -Force | Out-Null
$downloadedModel = Join-Path $cacheDirectory 'kaz.traineddata'
Invoke-WebRequest -Uri $sourceUri -OutFile $downloadedModel
if ((Get-FileHash -LiteralPath $downloadedModel -Algorithm SHA256).Hash -ne $expectedHash) {
    throw 'Downloaded model checksum does not match the pinned official asset.'
}
New-Item -ItemType Directory -Path $modelDirectory -Force | Out-Null
Copy-Item -LiteralPath $downloadedModel -Destination $targetModel
Write-Output "Installed verified Kazakh model: $targetModel"
