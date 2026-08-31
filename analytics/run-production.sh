#!/usr/bin/env bash
set -euo pipefail

log_dir=/var/log/nginx
state_dir=/var/lib/michael-analytics
python=/opt/michael-analytics/.venv/bin/python
aggregate=/opt/michael-analytics/aggregate.py
mmdb="$state_dir/DBIP-City-Lite.mmdb"

logs=()
[[ -r "$log_dir/michaeltechs.access.log.1" ]] && logs+=("$log_dir/michaeltechs.access.log.1")
[[ -r "$log_dir/michaeltechs.access.log" ]] && logs+=("$log_dir/michaeltechs.access.log")
((${#logs[@]})) || { echo "no readable MichaelTechs access log" >&2; exit 1; }

cat "${logs[@]}" | "$python" "$aggregate" - \
  --db "$state_dir/visitors.sqlite3" \
  --mmdb "$mmdb" \
  --provider "DB-IP City Lite" \
  --public-json "$state_dir/public.json" \
  --private-json "$state_dir/private.json" \
  --retention-days 0 \
  --minimum 1

chown root:www-data "$state_dir/public.json" "$state_dir/private.json"
chmod 0644 "$state_dir/public.json"
chmod 0640 "$state_dir/private.json"
chmod 0600 "$state_dir/visitors.sqlite3"
