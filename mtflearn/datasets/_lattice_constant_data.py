# crystal_lattice_constants.py
"""
Crystal lattice constants database
Includes transition metal dichalcogenides (TMDs), perovskites, and elemental metals

Units: Angstroms (Å)
Compiled from literature sources (2024-2025)
"""

from typing import Optional, Dict, List, Tuple
import math

# ==================== TMD LATTICE CONSTANTS ====================
"""
1H/2H TMD lattice constants (in-plane parameter 'a' in Angstroms)
Hexagonal structure with space group P6_3/mmc
"""

TMD_LATTICE_CONSTANTS = {
    # Group 6 - Mo compounds
    'MoS2': 3.160,
    'MoSe2': 3.289,
    'MoTe2': 3.520,

    # Group 6 - W compounds
    'WS2': 3.153,
    'WSe2': 3.282,
    'WTe2': 3.477,

    # Group 5 - V compounds
    'VS2': 3.210,
    'VSe2': 3.350,
    'VTe2': 3.660,

    # Group 5 - Nb compounds
    'NbS2': 3.330,
    'NbSe2': 3.440,
    'NbTe2': 3.670,

    # Group 5 - Ta compounds
    'TaS2': 3.360,
    'TaSe2': 3.440,
    'TaTe2': 3.700,

    # Group 4 - Ti compounds
    'TiS2': 3.407,
    'TiSe2': 3.536,
    'TiTe2': 3.780,

    # Group 4 - Zr compounds
    'ZrS2': 3.662,
    'ZrSe2': 3.771,
    'ZrTe2': 3.950,

    # Group 4 - Hf compounds
    'HfS2': 3.640,
    'HfSe2': 3.750,
    'HfTe2': 3.950,

    # Group 7 - Re compounds (distorted)
    'ReS2': 3.140,
    'ReSe2': 3.160,

    # Cr compounds (magnetic)
    'CrS2': 3.020,
    'CrSe2': 3.410,
    'CrTe2': 3.660,

    # Janus TMDs
    'MoSSe': 3.225,
    'WSSe': 3.231,
    'MoSTe': 3.340,
    'WSTe': 3.330,
}

# ==================== PEROVSKITE LATTICE CONSTANTS ====================
"""
Perovskite (ABO3) lattice constants
Values are for pseudo-cubic lattice parameter 'a' in Angstroms
For non-cubic structures, values represent the pseudo-cubic parameter
or average of a, b, c parameters for comparison purposes.

Note: Many perovskites have temperature-dependent phases
Values given are typically for room temperature
"""

PEROVSKITE_LATTICE_CONSTANTS = {
    # ===== CUBIC PEROVSKITES (truly cubic at RT) =====
    # Titanates
    'SrTiO3': 3.905,      # Ideal cubic perovskite, substrate standard
    'KTaO3': 3.989,       # Cubic
    'CaTiO3': 3.84,       # Orthorhombic, pseudo-cubic value

    # ===== TETRAGONAL PEROVSKITES (at RT) =====
    # BaTiO3 - tetragonal at RT (a=3.992-3.995, c=4.034-4.038)
    'BaTiO3': 3.992,      # Using 'a' parameter (tetragonal at RT)

    # PbTiO3 - highly tetragonal (a~3.90, c~4.15)
    'PbTiO3': 3.90,       # Using 'a' parameter (tetragonal)

    # ===== ORTHORHOMBIC PEROVSKITES (pseudo-cubic values) =====
    # Rare earth aluminates
    'LaAlO3': 3.79,       # Rhombohedral R3c, pseudo-cubic
    'NdAlO3': 3.74,       # Orthorhombic
    'PrAlO3': 3.76,       # Orthorhombic
    'GdAlO3': 3.71,       # Orthorhombic

    # Rare earth gallates
    'NdGaO3': 3.86,       # Orthorhombic, often used as substrate
    'LaGaO3': 3.89,       # Orthorhombic
    'PrGaO3': 3.87,       # Orthorhombic

    # Rare earth scandates (important substrates)
    'DyScO3': 3.95,       # Orthorhombic, pseudo-cubic
    'GdScO3': 3.96,       # Orthorhombic
    'SmScO3': 3.98,       # Orthorhombic
    'NdScO3': 4.00,       # Orthorhombic
    'PrScO3': 4.02,       # Orthorhombic
    'LaScO3': 4.05,       # Orthorhombic, large lattice

    # Manganites
    'LaMnO3': 3.95,       # Orthorhombic Pbnm, pseudo-cubic
    'PrMnO3': 3.87,       # Orthorhombic
    'NdMnO3': 3.82,       # Orthorhombic
    'SmMnO3': 3.78,       # Orthorhombic

    # Other important perovskites
    'BiFeO3': 3.96,       # Rhombohedral R3c, pseudo-cubic
    'LaFeO3': 3.93,       # Orthorhombic
    'NdFeO3': 3.87,       # Orthorhombic
    'LaCrO3': 3.88,       # Orthorhombic
    'LaCoO3': 3.82,       # Rhombohedral

    # Niobates and tantalates
    'KNbO3': 4.00,        # Orthorhombic at RT
    'NaNbO3': 3.91,       # Orthorhombic
    'NaTaO3': 3.89,       # Orthorhombic
    'LiNbO3': 3.78,       # Rhombohedral (pseudo-cubic)
    'LiTaO3': 3.78,       # Rhombohedral

    # Ruthenates and vanadates
    'SrRuO3': 3.93,       # Orthorhombic, conducting
    'CaRuO3': 3.84,       # Orthorhombic
    'LaVO3': 3.93,        # Orthorhombic

    # Mixed substrate materials
    'LSAT': 3.868,        # (La,Sr)(Al,Ta)O3 - cubic substrate

    # Lead-based (highly tetragonal/rhombohedral)
    'PbZrO3': 4.15,       # Orthorhombic antiferroelectric

    # Strontium-based
    'SrZrO3': 4.10,       # Orthorhombic
    'SrHfO3': 4.07,       # Orthorhombic

    # Barium-based
    'BaZrO3': 4.19,       # Cubic
    'BaSnO3': 4.116,      # Cubic
    'BaHfO3': 4.17,       # Cubic

    # Calcium-based
    'CaZrO3': 4.03,       # Orthorhombic
    'CaSnO3': 3.94,       # Orthorhombic
    'CaHfO3': 4.00,       # Orthorhombic
}

# ==================== METAL LATTICE CONSTANTS ====================
"""
Elemental metal lattice constants organized by crystal structure
Values at room temperature unless noted

FCC (Face-Centered Cubic): a parameter
BCC (Body-Centered Cubic): a parameter  
HCP (Hexagonal Close-Packed): a and c parameters

For cubic structures (FCC, BCC): only 'a' is listed
For HCP structures: format is (a, c) or single 'a' value where c/a ≈ 1.633 (ideal)
"""

# ===== FCC METALS (Face-Centered Cubic) =====
# High ductility, excellent conductivity, packing factor = 0.74
FCC_METALS = {
    # Group 11 - Coinage metals (excellent conductors)
    'Cu': 3.6149,      # Copper - benchmark conductor
    'Ag': 4.0853,      # Silver - highest conductivity
    'Au': 4.0782,      # Gold - corrosion resistant

    # Group 10 - Platinum group
    'Ni': 3.524,       # Nickel - magnetic FCC metal
    'Pd': 3.8907,      # Palladium - hydrogen storage
    'Pt': 3.9242,      # Platinum - catalysis

    # Group 13 - Aluminum
    'Al': 4.0495,      # Aluminum - lightweight structural

    # Group 9 - Rhodium, Iridium
    'Rh': 3.8034,      # Rhodium - catalytic
    'Ir': 3.839,       # Iridium - extremely dense FCC

    # Group 4 - High temperature FCC phase
    'Ca': 5.5884,      # Calcium - FCC at RT
    'Sr': 6.0849,      # Strontium

    # Group 3 - Rare earths
    'Yb': 5.4847,      # Ytterbium - FCC rare earth

    # Others
    'Pb': 4.9508,      # Lead - heavy, soft FCC
    'Th': 5.0842,      # Thorium - actinide FCC
    'Ac': 5.67,        # Actinium
}

# ===== BCC METALS (Body-Centered Cubic) =====
# High strength, lower ductility, packing factor = 0.68
BCC_METALS = {
    # Group 6 - Chromium group (refractory)
    'Cr': 2.910,       # Chromium - hard, corrosion resistant
    'Mo': 3.147,       # Molybdenum - high melting point
    'W': 3.1652,       # Tungsten - highest melting point metal

    # Group 5 - Vanadium group
    'V': 3.030,        # Vanadium - steel alloying
    'Nb': 3.3004,      # Niobium - superconducting
    'Ta': 3.3013,      # Tantalum - corrosion resistant

    # Iron group
    'Fe': 2.8665,      # Iron (α-Fe) - most important engineering metal

    # Alkali metals (very soft BCC)
    'Li': 3.51,        # Lithium - lightest metal
    'Na': 4.2906,      # Sodium
    'K': 5.328,        # Potassium
    'Rb': 5.585,       # Rubidium
    'Cs': 6.141,       # Cesium - largest lattice constant

    # Alkaline earth
    'Ba': 5.028,       # Barium

    # Others
    'Eu': 4.581,       # Europium - BCC rare earth
    'Ra': 5.148,       # Radium - radioactive
}

# ===== HCP METALS (Hexagonal Close-Packed) =====
# Format: (a, c) in Angstroms
# Ideal c/a ratio = 1.633, but many metals deviate
HCP_METALS = {
    # Group 4 - Titanium group
    'Ti': (2.9508, 4.6855),    # Titanium - aerospace alloy, c/a = 1.587
    'Zr': (3.232, 5.147),      # Zirconium - nuclear applications, c/a = 1.593
    'Hf': (3.1964, 5.0511),    # Hafnium - nuclear control, c/a = 1.580

    # Group 3 - Scandium, Yttrium
    'Sc': (3.309, 5.2733),     # Scandium, c/a = 1.594
    'Y': (3.6474, 5.7306),     # Yttrium, c/a = 1.571

    # Group 12 - Zinc group
    'Zn': (2.6649, 4.9468),    # Zinc - galvanizing, c/a = 1.856
    'Cd': (2.9794, 5.6186),    # Cadmium, c/a = 1.886

    # Group 2 - Alkaline earth
    'Be': (2.2858, 3.5843),    # Beryllium - lightweight, c/a = 1.568
    'Mg': (3.2094, 5.2108),    # Magnesium - lightweight alloy, c/a = 1.623

    # Group 7 - Technetium, Rhenium
    'Tc': (2.735, 4.388),      # Technetium - radioactive, c/a = 1.604
    'Re': (2.761, 4.456),      # Rhenium - refractory, c/a = 1.614

    # Group 8 - Ruthenium, Osmium
    'Ru': (2.7059, 4.2815),    # Ruthenium, c/a = 1.582
    'Os': (2.7344, 4.3173),    # Osmium - densest element, c/a = 1.579

    # Group 9 - Cobalt (FCC at high T)
    'Co': (2.5071, 4.0695),    # Cobalt - magnetic HCP at RT, c/a = 1.623

    # Group 13 - Thallium, Indium
    'Tl': (3.4566, 5.5248),    # Thallium, c/a = 1.598
    'In': (3.2523, 4.9461),    # Indium, c/a = 1.521

    # Rare earths (most are HCP)
    'Gd': (3.636, 5.7826),     # Gadolinium, c/a = 1.590
    'Tb': (3.601, 5.6936),     # Terbium, c/a = 1.581
    'Dy': (3.593, 5.6537),     # Dysprosium, c/a = 1.573
    'Ho': (3.5773, 5.6158),    # Holmium, c/a = 1.570
    'Er': (3.5588, 5.5874),    # Erbium, c/a = 1.570
    'Tm': (3.5375, 5.5546),    # Thulium, c/a = 1.570
    'Lu': (3.5031, 5.5509),    # Lutetium, c/a = 1.585
}

# Combined simple format (for HCP, using 'a' parameter only)
METAL_LATTICE_CONSTANTS = {
    **FCC_METALS,
    **BCC_METALS,
    **{k: v[0] if isinstance(v, tuple) else v for k, v in HCP_METALS.items()}
}

# ==================== COMBINED DATABASE ====================

ALL_LATTICE_CONSTANTS = {
    **TMD_LATTICE_CONSTANTS,
    **PEROVSKITE_LATTICE_CONSTANTS,
    **METAL_LATTICE_CONSTANTS
}

# ==================== MATERIAL INFO ====================

TMD_INFO = {
    'MoS2': {'type': 'TMD', 'group': 6, 'metal': 'Mo', 'chalcogen': 'S', 'structure': '1H/2H'},
    'WS2': {'type': 'TMD', 'group': 6, 'metal': 'W', 'chalcogen': 'S', 'structure': '1H/2H'},
    'MoSe2': {'type': 'TMD', 'group': 6, 'metal': 'Mo', 'chalcogen': 'Se', 'structure': '1H/2H'},
    'WSe2': {'type': 'TMD', 'group': 6, 'metal': 'W', 'chalcogen': 'Se', 'structure': '1H/2H'},
}

PEROVSKITE_INFO = {
    'SrTiO3': {'type': 'perovskite', 'phase': 'cubic', 'space_group': 'Pm-3m', 'note': 'Most common substrate'},
    'BaTiO3': {'type': 'perovskite', 'phase': 'tetragonal_RT', 'space_group': 'P4mm', 'note': 'Ferroelectric'},
    'LaAlO3': {'type': 'perovskite', 'phase': 'rhombohedral', 'space_group': 'R3c', 'note': 'Slight distortion'},
    'PbTiO3': {'type': 'perovskite', 'phase': 'tetragonal', 'space_group': 'P4mm', 'note': 'Large c/a ratio'},
    'BiFeO3': {'type': 'perovskite', 'phase': 'rhombohedral', 'space_group': 'R3c', 'note': 'Multiferroic'},
    'KTaO3': {'type': 'perovskite', 'phase': 'cubic', 'space_group': 'Pm-3m', 'note': 'Quantum paraelectric'},
}

METAL_INFO = {
    'Cu': {'type': 'metal', 'structure': 'FCC', 'note': 'Excellent conductor'},
    'Fe': {'type': 'metal', 'structure': 'BCC', 'note': 'Most important engineering metal'},
    'Al': {'type': 'metal', 'structure': 'FCC', 'note': 'Lightweight structural'},
    'Ti': {'type': 'metal', 'structure': 'HCP', 'note': 'Aerospace alloy'},
    'Au': {'type': 'metal', 'structure': 'FCC', 'note': 'Corrosion resistant'},
    'Ag': {'type': 'metal', 'structure': 'FCC', 'note': 'Highest conductivity'},
    'W': {'type': 'metal', 'structure': 'BCC', 'note': 'Highest melting point'},
    'Mg': {'type': 'metal', 'structure': 'HCP', 'note': 'Lightweight alloy'},
}

ALL_MATERIAL_INFO = {
    **TMD_INFO,
    **PEROVSKITE_INFO,
    **METAL_INFO
}

# ==================== UTILITY FUNCTIONS ====================

def get_lattice_constant(formula: str) -> Optional[float]:
    """
    Get lattice constant for any material formula.

    Parameters
    ----------
    formula : str
        Chemical formula (e.g., 'MoS2', 'SrTiO3', 'Cu')

    Returns
    -------
    float or None
        Lattice constant in Angstroms, or None if not found
    """
    return ALL_LATTICE_CONSTANTS.get(formula, None)


def get_hcp_parameters(element: str) -> Optional[Tuple[float, float]]:
    """
    Get both a and c parameters for HCP metals.

    Parameters
    ----------
    element : str
        Element symbol (e.g., 'Ti', 'Mg')

    Returns
    -------
    tuple of (a, c) or None
        HCP lattice parameters in Angstroms
    """
    return HCP_METALS.get(element, None)


def get_material_info(formula: str) -> Optional[Dict]:
    """
    Get additional structural information for a material.

    Parameters
    ----------
    formula : str
        Chemical formula or element symbol

    Returns
    -------
    dict or None
        Dictionary with material information, or None if not found
    """
    return ALL_MATERIAL_INFO.get(formula, None)


def get_material_type(formula: str) -> Optional[str]:
    """
    Identify whether a material is TMD, perovskite, or metal.

    Parameters
    ----------
    formula : str
        Chemical formula or element symbol

    Returns
    -------
    str or None
        'TMD', 'perovskite', 'metal', or None if not found
    """
    if formula in TMD_LATTICE_CONSTANTS:
        return 'TMD'
    elif formula in PEROVSKITE_LATTICE_CONSTANTS:
        return 'perovskite'
    elif formula in METAL_LATTICE_CONSTANTS:
        return 'metal'
    else:
        return None


def get_crystal_structure(element: str) -> Optional[str]:
    """
    Get crystal structure type for elemental metals.

    Parameters
    ----------
    element : str
        Element symbol

    Returns
    -------
    str or None
        'FCC', 'BCC', 'HCP', or None if not a metal in database
    """
    if element in FCC_METALS:
        return 'FCC'
    elif element in BCC_METALS:
        return 'BCC'
    elif element in HCP_METALS:
        return 'HCP'
    else:
        return None


def get_materials_in_range(a_min: float, a_max: float,
                           material_type: Optional[str] = None) -> Dict[str, float]:
    """
    Get all materials with lattice constants in specified range.

    Parameters
    ----------
    a_min, a_max : float
        Minimum and maximum lattice constants in Angstroms
    material_type : str, optional
        Filter by type: 'TMD', 'perovskite', 'metal', or None for all

    Returns
    -------
    dict
        Dictionary of formula: lattice_constant pairs in range
    """
    if material_type == 'TMD':
        search_dict = TMD_LATTICE_CONSTANTS
    elif material_type == 'perovskite':
        search_dict = PEROVSKITE_LATTICE_CONSTANTS
    elif material_type == 'metal':
        search_dict = METAL_LATTICE_CONSTANTS
    else:
        search_dict = ALL_LATTICE_CONSTANTS

    return {formula: a for formula, a in search_dict.items()
            if a_min <= a <= a_max}


def list_all_tmds() -> List[str]:
    """Get list of all TMD formulas in database."""
    return list(TMD_LATTICE_CONSTANTS.keys())


def list_all_perovskites() -> List[str]:
    """Get list of all perovskite formulas in database."""
    return list(PEROVSKITE_LATTICE_CONSTANTS.keys())


def list_all_metals() -> List[str]:
    """Get list of all metal symbols in database."""
    return list(METAL_LATTICE_CONSTANTS.keys())


def list_metals_by_structure(structure: str) -> List[str]:
    """
    Get list of metals with specific crystal structure.

    Parameters
    ----------
    structure : str
        'FCC', 'BCC', or 'HCP'

    Returns
    -------
    list of str
        Element symbols
    """
    if structure.upper() == 'FCC':
        return list(FCC_METALS.keys())
    elif structure.upper() == 'BCC':
        return list(BCC_METALS.keys())
    elif structure.upper() == 'HCP':
        return list(HCP_METALS.keys())
    else:
        return []


def get_tmd_nearest_neighbor_distance(formula: str) -> Optional[float]:
    """
    Calculate nearest neighbor distance for TMD (a/√3).

    Parameters
    ----------
    formula : str
        TMD formula

    Returns
    -------
    float or None
        Nearest neighbor distance in Angstroms
    """
    a = TMD_LATTICE_CONSTANTS.get(formula, None)
    if a is not None:
        return a / math.sqrt(3)
    return None


def find_lattice_matched_pairs(tolerance: float = 0.05) -> List[Tuple[str, str, float]]:
    """
    Find material pairs with similar lattice constants.

    Parameters
    ----------
    tolerance : float
        Maximum difference in lattice constant (Angstroms)

    Returns
    -------
    list of tuples
        Each tuple: (material1, material2, lattice_mismatch)
    """
    pairs = []
    materials = list(ALL_LATTICE_CONSTANTS.keys())

    for i, mat1 in enumerate(materials):
        for mat2 in materials[i+1:]:
            a1 = ALL_LATTICE_CONSTANTS[mat1]
            a2 = ALL_LATTICE_CONSTANTS[mat2]
            diff = abs(a1 - a2)

            if diff <= tolerance:
                pairs.append((mat1, mat2, diff))

    return sorted(pairs, key=lambda x: x[2])

