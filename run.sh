#!/usr/bin/env bash
set -euo pipefail
: "${MEDIA_API_URL:?Set MEDIA_API_URL to the data query endpoint}"
curl -X POST "$MEDIA_API_URL" \
  -H "Content-Type: application/json" \
  -d '{
        "currentPage": 1,
        "pageSize": 20,
        "msgType": "2",
        "fileAttribute": ["2"],
        "startTime": "2025-02-26 00:00:00",
        "endTime": "2025-02-27 00:00:00"
  }'

