param(
    [Parameter(Mandatory = $true)]
    [string]$RunCmd
)

$ErrorActionPreference = "Stop"
$repoPath = (Resolve-Path ".").Path
$bootstrap = "python -m pip install -q opentelemetry-api opentelemetry-sdk arize-phoenix-otel pytest"
$inner = "$bootstrap && $RunCmd"

docker compose run --rm --no-deps -T -v "${repoPath}:/app" celery sh -lc "$inner"
