# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

"""Automatically generate the code for the API reference pages."""

from pathlib import Path

import mkdocs_gen_files

root = Path(__file__).parent.parent.parent
src = root / "cable_thermal_model"

for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(root).with_suffix("")
    doc_path = path.relative_to(root).with_suffix(".md")
    full_doc_path = Path("api_reference", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
    elif parts[-1] == "__main__":
        continue

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join(parts)
        print("::: " + identifier, file=fd)

    print(full_doc_path)
    print(Path("../") / path)
    print(path.relative_to(root))
    print()

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))
