#!/usr/bin/env python

# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

# -*- coding: utf-8 -*-


class MissingMaterialException(Exception):
    """Exception thrown in case there are missing material properties."""

    pass


class MissingAttributeException(Exception):
    """Exception thrown in case there are circuits has not yet been initialized."""

    pass
