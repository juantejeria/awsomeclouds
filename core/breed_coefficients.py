"""
Multiplicadores de corrección para estimación de peso por raza, categoría y edad.

Fuentes: Schaeffer (2006), MARC Germplasm Program, estudios morfométricos de cruzas zebuinas.

Fórmula base:   raw_weight = (BL * Girth² * lb) / 300
Ajustada:       weight = raw_weight * K_breed * K_category * K_age

Función principal: get_weight_multiplier(breed, category, age_range) → float
  Retorna el producto K_breed × K_category × K_age.
  Claves desconocidas devuelven 1.0 (sin corrección).

También exporta BREED_OPTIONS, CATEGORY_OPTIONS y AGE_OPTIONS para la UI.
"""

# ── Breed multipliers (K_breed) ────────────────────────────────────
# Reference breed: Angus (compact, well-muscled) = 1.00
BREED_COEFFICIENTS = {
    "angus":                1.00,
    "hereford":             0.98,
    "shorthorn":            0.96,
    "charolais":            1.04,
    "limousin":             0.98,
    "simmental":            1.01,
    "brahman":              0.93,
    "nelore":               0.91,
    "gyr":                  0.88,
    "brangus":              0.96,
    "bradford":             0.95,
    "braford":              0.95,
    "holando":              0.93,
    "holstein":             0.93,
    "generico":             1.00,
    "desconocido":          1.00,
}

# ── Category multipliers (K_category) ──────────────────────────────
CATEGORY_COEFFICIENTS = {
    "ternero":              0.84,
    "ternera":              0.84,
    "recria":               0.90,
    "vaquillona":           0.95,
    "novillito":            0.97,
    "novillo":              1.00,
    "vaca":                 0.95,
    "toro":                 1.08,
    "desconocido":          1.00,
}

# ── Age multipliers (K_age) ────────────────────────────────────────
AGE_COEFFICIENTS = {
    "0-6":                  0.85,
    "6-12":                 0.92,
    "12-18":                0.96,
    "18-24":                0.98,
    "24-36":                1.00,
    "36+":                  1.00,
    "desconocido":          1.00,
}

# ── Dropdown labels (for UI) ───────────────────────────────────────
BREED_OPTIONS = [
    ("desconocido",  "Desconocido"),
    ("angus",        "Angus"),
    ("hereford",     "Hereford"),
    ("shorthorn",    "Shorthorn"),
    ("charolais",    "Charolais"),
    ("limousin",     "Limousin"),
    ("simmental",    "Simmental"),
    ("brahman",      "Brahman"),
    ("nelore",       "Nelore"),
    ("gyr",          "Gyr"),
    ("brangus",      "Brangus"),
    ("bradford",     "Bradford / Braford"),
    ("holando",      "Holando (Holstein)"),
    ("generico",     "Genérico"),
]

CATEGORY_OPTIONS = [
    ("desconocido",  "Desconocido"),
    ("ternero",      "Ternero/a (cría)"),
    ("recria",       "Recría (8-12 m)"),
    ("vaquillona",   "Vaquillona"),
    ("novillito",    "Novillito"),
    ("novillo",      "Novillo"),
    ("vaca",         "Vaca"),
    ("toro",         "Toro"),
]

AGE_OPTIONS = [
    ("desconocido",  "Desconocido"),
    ("0-6",          "0 – 6 meses"),
    ("6-12",         "6 – 12 meses"),
    ("12-18",        "12 – 18 meses"),
    ("18-24",        "18 – 24 meses"),
    ("24-36",        "24 – 36 meses"),
    ("36+",          "36+ meses"),
]


# ── Weight ranges per category (kg) ──────────────────────────────
# Used by batch screening to flag outliers.
# Source: INTA / Argentine cattle production references.
WEIGHT_RANGES = {
    "ternero":     (50, 180),
    "ternera":     (50, 180),
    "recria":      (120, 250),
    "vaquillona":  (170, 350),
    "novillito":   (180, 400),
    "novillo":     (300, 550),
    "vaca":        (300, 600),
    "toro":        (400, 900),
    "desconocido": (100, 700),
}


# ── Estimated bbox height by category × age (cm) ─────────────────
# Approximate YOLO bounding-box height for Angus/Hereford cattle.
# Not shoulder height — includes head up and legs down as seen by YOLO.
# Source: field observations + INTA morphometric references.
# Used as default scale when no calibration posts are available.
HEIGHT_BY_CATEGORY_AGE = {
    "ternero": {
        "0-6":   70,
        "6-12":  85,
        "12-18": 95,
        "18-24": 100,
        "24-36": 105,
        "36+":   105,
    },
    "ternera": {
        "0-6":   68,
        "6-12":  83,
        "12-18": 92,
        "18-24": 97,
        "24-36": 100,
        "36+":   100,
    },
    "recria": {
        "0-6":   75,
        "6-12":  88,
        "12-18": 98,
        "18-24": 105,
        "24-36": 110,
        "36+":   110,
    },
    "vaquillona": {
        "0-6":   75,
        "6-12":  90,
        "12-18": 100,
        "18-24": 108,
        "24-36": 115,
        "36+":   118,
    },
    "novillito": {
        "0-6":   78,
        "6-12":  92,
        "12-18": 105,
        "18-24": 112,
        "24-36": 118,
        "36+":   120,
    },
    "novillo": {
        "0-6":   80,
        "6-12":  95,
        "12-18": 108,
        "18-24": 118,
        "24-36": 125,
        "36+":   130,
    },
    "vaca": {
        "0-6":   80,
        "6-12":  95,
        "12-18": 105,
        "18-24": 112,
        "24-36": 120,
        "36+":   125,
    },
    "toro": {
        "0-6":   85,
        "6-12":  100,
        "12-18": 115,
        "18-24": 125,
        "24-36": 135,
        "36+":   145,
    },
}


def get_estimated_height(category="desconocido", age_range="desconocido"):
    """Return estimated bbox height in cm for the given category and age.

    Falls back to 120 cm (generic adult) if category/age not found.
    """
    cat = category.lower().strip()
    age = age_range.lower().strip()
    cat_heights = HEIGHT_BY_CATEGORY_AGE.get(cat)
    if cat_heights:
        return cat_heights.get(age, cat_heights.get("24-36", 120))
    # desconocido: use novillo as reference
    novillo = HEIGHT_BY_CATEGORY_AGE.get("novillo", {})
    return novillo.get(age, 120)


def get_weight_range(category="desconocido"):
    """Return (min_kg, max_kg) for the given category."""
    return WEIGHT_RANGES.get(category.lower().strip(), WEIGHT_RANGES["desconocido"])


def get_weight_multiplier(breed="desconocido", category="desconocido", age_range="desconocido"):
    """Return the combined correction multiplier K_breed * K_category * K_age.

    Unknown / missing keys fall back to 1.0 (no correction).
    """
    k_breed = BREED_COEFFICIENTS.get(breed.lower().strip(), 1.0)
    k_category = CATEGORY_COEFFICIENTS.get(category.lower().strip(), 1.0)
    k_age = AGE_COEFFICIENTS.get(age_range.lower().strip(), 1.0)
    return k_breed * k_category * k_age
