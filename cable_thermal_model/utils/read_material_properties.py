# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path

import pandas as pd


def read_material_properties() -> pd.DataFrame:
    """Reads material properties from the CSV file and returns them as a Pandas DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing material properties indexed by material name.
    """
    return pd.read_csv(Path(__file__).parents[2].resolve() / "data" / "material_properties.csv", sep=";").set_index(
        "material"
    )
