#!/usr/bin/env bash
# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project

# SPDX-License-Identifier: MPL-2.0

set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="$repo_root/docs/developer_guidelines"
output_file="$output_dir/classes_cable_thermal_model.html"

cd "$repo_root"
mkdir -p "$output_dir"

pyreverse -o html --colorized -p cable_thermal_model cable_thermal_model/
python3 scripts/add_mermaid_zoom.py \
  classes_cable_thermal_model.html \
  --output "$output_file"

rm -f -- classes_cable_thermal_model.html packages_cable_thermal_model.html
