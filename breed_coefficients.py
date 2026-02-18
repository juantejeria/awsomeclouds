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


def get_weight_multiplier(breed="desconocido", category="desconocido", age_range="desconocido"):
    """Return the combined correction multiplier K_breed * K_category * K_age.

    Unknown / missing keys fall back to 1.0 (no correction).
    """
    k_breed = BREED_COEFFICIENTS.get(breed.lower().strip(), 1.0)
    k_category = CATEGORY_COEFFICIENTS.get(category.lower().strip(), 1.0)
    k_age = AGE_COEFFICIENTS.get(age_range.lower().strip(), 1.0)
    return k_breed * k_category * k_age
