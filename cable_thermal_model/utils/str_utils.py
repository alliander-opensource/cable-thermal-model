# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0


def tab_lines(message: str) -> str:
    """Prepends a tab to each line of the message."""
    return "\n".join("\t" + line for line in message.splitlines())
