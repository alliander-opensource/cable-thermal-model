# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

"""Automatically generate the code for the API reference pages."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

root = Path(__file__).parent.parent.parent
src = root / "cable_thermal_model"

EXCLUDED_MODULE_NAMES = ("_version",)

for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(root).with_suffix("")
    doc_path = path.relative_to(root).with_suffix(".md")
    full_doc_path = Path("api_reference", doc_path)

    parts = tuple(module_path.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
        # Check if the file contains an __all__ definition
        # If not, we do not render the __init__.py file in the documentation
        with open((Path(root) / module_path).with_suffix(".py")) as f:
            if "__all__" in f.read():
                doc_path = doc_path.with_name("index.md")
                full_doc_path = full_doc_path.with_name("index.md")
            else:
                continue

    elif parts[-1] == "__main__":
        continue

    # Exclude specified modules
    if parts[-1] in EXCLUDED_MODULE_NAMES:
        continue

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join(parts)
        print("::: " + identifier, file=fd)

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

with mkdocs_gen_files.open("api_reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
