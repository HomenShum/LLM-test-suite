$ErrorActionPreference = "Stop"

# Remove tracked markdown files except our allow-list
$tracked = git ls-files *.md
foreach ($f in $tracked) {
  $name = Split-Path -Leaf $f
  if ($name -ieq 'README.md' -or $name -ieq 'DESIGN_SPECS.md' -or $name -ieq 'FOLDER_STRUCTURES.md' -or $name -ieq 'CHANGELOG.md') {
    continue
  }
  Write-Host "Removing $f"
  git rm -f -- $f
}

