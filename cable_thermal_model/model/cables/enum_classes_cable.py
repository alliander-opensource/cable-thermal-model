# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from enum import Enum, StrEnum

"""This module holds Enum classes based on the known possible values in the [example_cables.csv] data file.

The goal of these classes is to both prevent loose strings from populating classes with possible typo's and to keep
 technical debt upon change of these values to a minimum.

"""


# pragma: no cover


class Material(StrEnum):
    """Base class for cable material enumerations."""


class CableType(StrEnum):  # pragma: no cover
    """Cable type."""

    XLPE = "XLPE"
    PILC = "PILC"
    OilPressure = "oilPressure"


class CableConductorCount(Enum):
    """Enum class for possible numbers of conductors in cables. Values derived from example_cables.csv."""

    One = 1
    Three = 3


class CableConductorShape(StrEnum):
    """Enum class for possible cable conductor shapes. Values derived from example_cables.csv."""

    Round = "ccRound"
    Sector = "ccSector"
    Hollow = "ccHollow"


class CableConductorMaterial(Material):
    """Enum class for possible cable conductor materials. Values derived from example_cables.csv."""

    Copper = "cmCu"
    Aluminium = "cmAl"


class CableConductorSurfaceType(StrEnum):
    """Enum class for possible cable conductor surface types. Values derived from example_cables.csv."""

    Compact = "ctCompact"
    Milliken = "ctMilliken"
    Solid = "ctSolid"
    Stranded = "ctStranded"


class CableInsulationMaterial(Material):
    """Enum class for possible cable insulation layer materials. Values derived from example_cables.csv."""

    EPR = "imEPR"
    PaperExternalGasPressure = "imPaperExternalGasPressure"
    PaperMassImpregnated = "imPaperMassImpregnated"
    PaperMassIntGasPressure = "imPaperMassIntGasPressure"
    PaperOFSelfContained = "imPaperOFSelfContained"
    PaperOilPressure = "imPaperOilPressure"
    PaperPreIntGasPressure = "imPaperPreIntGasPressure"
    PE = "imPE"
    PPL = "imPPL"
    XLPEFilled = "imXLPEFilled"
    XLPEUnfilled = "imXLPEUnfilled"
    NONE = "imNone"


class CableConductorInsulationScreenMaterial(Material):
    """Enum class for possible conductor insulation screen materials. Values derived from example_cables.csv."""

    CopperTape = "scrCuTape"
    MetalizedPaper = "scrMetalizedPaper"
    MetalTape = "scrMetalTape"
    XLPE = "scrXLPE"
    SemiconductingXLPE = "scrSemiconductingXLPE"
    NONE = "scrNone"


class CableOilMaterial(Material):
    """Enum class for possible cable oil duct materials."""

    Oil = "odOil"


class CableScreenMaterial(Material):
    """Enum class for possible cable screen layer materials. Values derived from example_cables.csv."""

    Aluminium = "samAluminium"
    Copper = "samCopper"
    Lead = "samLead"
    StainlessSteel = "samStainlessSteel"
    NONE = "samNone"


class CableScreenType(StrEnum):
    """Enum class for possible cable screen types. Values derived from example_cables.csv."""

    Belted = "scBelted"
    Common = "scCommon"
    PipeType = "scPipeType"
    Separate = "scSeparate"
    SL = "scSL"


class CableBeddingMaterial(Material):
    """Enum class for possible cable bedding materials. Values derived from example_cables.csv."""

    BitumenJute = "imBitumenJute"
    EPR = "imEPR"
    PaperMassImpregnated = "imPaperMassImpregnated"
    PaperOFSelfContained = "imPaperOFSelfContained"
    PaperOilPressure = "imPaperOilPressure"
    PaperPreIntGasPressure = "imPaperPreIntGasPressure"
    PE = "imPE"
    PVC = "imPVC"
    Rubber = "imRubber"
    NONE = "imNone"


class CableArmourMaterial(Material):
    """Enum class for possible cable armour layer materials. Values derived from example_cables.csv."""

    Copper = "samCopper"
    StainlessSteel = "samStainlessSteel"
    Steel = "samSteel"
    NONE = "samNone"


class CableArmourConfiguration(StrEnum):
    """Enum class for possible cable armour layer configurations. Values derived from example_cables.csv."""

    Common = "acCommon"
    Separate = "acSeparate"
    NONE = "acNone"


class CableArmourType(StrEnum):
    """Enum class for possible cable armour layer types. Values derived from example_cables.csv."""

    Tape = "atTape"
    Wire = "atWire"
    NONE = "atNone"


class CableWireTapeLay(StrEnum):
    """Enum class for possible cable wire trap lay configurations. Values derived from example_cables.csv."""

    ContactLay = "alContactLay"
    LongLay = "alLongLay"
    ShortLay = "alShortLay"
    NONE = "alNone"


class CableSheathMaterial(Material):
    """Enum class for possible cable sheath materials. Values derived from example_cables.csv."""

    BitumenJute = "pcBitumenJute"
    Lead = "pcLead"
    PE = "pcPE"
    PVC = "pcPVC"


class CableOilDuct(StrEnum):
    """Enum class for possible cable oil duct configurations. Values derived from example_cables.csv."""

    Ductless = "odDuctless"
    External = "odExternal"
    Internal = "odInternal"
    NONE = "odNone"


class CablePipeMaterial(Material):
    """Enum class for possible cable pipe layer materials. Values derived from example_cables.csv."""

    Steel = "pmSteel"
    NONE = "pmNone"


class CableLayer(StrEnum):
    """Enum class for possible cable layer types. Values derived from those used in the old Cable init."""

    Armour = "Armour"
    Bedding = "Bedding"
    Conductor = "Conductor"
    ConductorScreen = "Conductor screen"
    InnerOilDuct = "Inner oil duct"
    Insulation = "Insulation"
    InsulationScreen = "Insulation screen"
    Pipe = "Pipe"
    PipeFill = "PipeFill"
    Screen = "Screen"
    Sheath = "Sheath"
    SoilOne = "Soil_1"
    SoilTwo = "Soil_2"
    SoilThree = "Soil_3"
    SoilFour = "Soil_4"

    @classmethod
    def soil_layers(cls) -> list[CableLayer]:
        """Return a list of all soil layers."""
        return [layer for layer in cls if layer.name.startswith("Soil")]


class CableScreenLossType(StrEnum):
    """Enum class for possible cable screen loss types."""

    ReturnZero = "_cable_screen_loss_method_return_zero"
    RoundOrOvalShapedWithScreen = "_cable_screen_loss_method_round_or_oval_with_screen"
    SectorShaped = "_cable_screen_loss_method_sector"
    CrossBondingOrOneSidedBondingTrefoil = "_cable_screen_loss_method_cross_bonding_or_one_sided_bonding_trefoil"
    CrossBondingOrOneSidedBondingLinearLeading = (
        "_cable_screen_loss_method_cross_bonding_or_one_sided_bonding_linear_leading"
    )
    CrossBondingOrOneSidedBondingLinearCenter = (
        "_cable_screen_loss_method_cross_bonding_or_one_sided_bonding_linear_center"
    )
    CrossBondingOrOneSidedBondingLinearLagging = (
        "_cable_screen_loss_method_cross_bonding_or_one_sided_bonding_linear_lagging"
    )
    TwoSidedBondingTrefoil = "_cable_screen_loss_method_two_sided_bonding_trefoil"
    TwoSidedBondingLinearLeading = "_cable_screen_loss_method_two_sided_bonding_linear_leading"
    TwoSidedBondingLinearCenter = "_cable_screen_loss_method_two_sided_bonding_linear_center"
    TwoSidedBondingLinearLagging = "_cable_screen_loss_method_two_sided_bonding_linear_lagging"
    TwoSidedBondingLeading = "_cable_screen_loss_method_two_sided_bonding_leading"
    TwoSidedBondingCenter = "_cable_screen_loss_method_two_sided_bonding_center"
    TwoSidedBondingLagging = "_cable_screen_loss_method_two_sided_bonding_lagging"
    SingleCableOilPressureOrXLPE = "_cable_screen_loss_method_single_cable_oil_pressure_or_xlpe"
    SingleCablePILC = "_cable_screen_loss_method_single_cable_pilc"


class CableConductorHeatSourceAt(StrEnum):
    """Enum class for possible locations in the conductor layer of a cable where to place internal heating."""

    Automatic = "automatic"
    Core = "core"
    Whole = "whole"
    Shell = "shell"


class PipeFillType(StrEnum):
    """Enum class for possible types of material filling the pipe surrounding a cable."""

    Air = "air"
    Water = "water"
