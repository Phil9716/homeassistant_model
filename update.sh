#!/bin/sh
# Holt die neueste Version des Hausmodells. AppDaemon lädt geänderte Apps automatisch neu.
set -e
cd "$(dirname "$0")"
git fetch --quiet origin
git reset --quiet --hard origin/main
echo "Hausmodell aktualisiert auf $(git log -1 --format='%h %s')"
