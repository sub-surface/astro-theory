#!/usr/bin/env bash
# Fetch arXiv papers as clean markdown into research/.
# Tries native arXiv HTML (newer papers) then ar5iv (older), converts with
# Pandoc, strips raw-HTML wrappers for readability. Idempotent: skips existing.
set -u
PANDOC="/c/Users/Leon/AppData/Local/Pandoc/pandoc.exe"
cd "$(dirname "$0")" && mkdir -p research

# id|slug  (order ~ reading order)
PAPERS=(
  "2009.14826|secrest2021_catwise_dipole"
  "2206.05624|secrest2022_challenge_lcdm"
  "2306.17749|storeyfisher2024_quaia_catalog"
  "2311.14938|oayda2024_quaia_bayesian_dipole"
  "2212.04925|guandalin2023_theoretical_systematics"
  "2503.02470|2025_kinematic_contribution_dipole"
  "2509.18689|bohme2025_radio_dipole_tensions"
)

fetch_html(){ # $1=id -> stdout html ; try arxiv native then ar5iv
  local id="$1" code
  code=$(curl -sL -o /tmp/p.html -w "%{http_code}" --max-time 60 "https://arxiv.org/html/${id}")
  if [ "$code" = "200" ] && [ "$(wc -c < /tmp/p.html)" -gt 30000 ]; then echo "arxiv"; return 0; fi
  code=$(curl -sL -o /tmp/p.html -w "%{http_code}" --max-time 90 "https://ar5iv.org/html/${id}")
  if [ "$code" = "200" ] && [ "$(wc -c < /tmp/p.html)" -gt 30000 ]; then echo "ar5iv"; return 0; fi
  return 1
}

for entry in "${PAPERS[@]}"; do
  id="${entry%%|*}"; slug="${entry##*|}"; out="research/${slug}.md"
  if [ -s "$out" ]; then echo "skip  $slug (exists)"; continue; fi
  src=$(fetch_html "$id") || { echo "FAIL  $id (no html)"; continue; }
  printf '<!-- arXiv:%s  source:%s  https://arxiv.org/abs/%s -->\n\n' "$id" "$src" "$id" > "$out"
  "$PANDOC" -f html -t gfm-raw_html --wrap=none /tmp/p.html 2>/dev/null \
    | sed '/^:::/d' | cat -s >> "$out"
  echo "ok    $slug  ($src, $(wc -l < "$out") lines)"
  sleep 2
done
