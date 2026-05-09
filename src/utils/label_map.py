"""
label_map.py — Single source of truth for MalariAI class definitions.

NEVER derive the label map dynamically from a CSV. Always import from here.
This guarantees the same integer ↔ class mapping across every module:
    prepare_data, dataset, faster_rcnn, efficientnet, metrics, gradcam.

Class index conventions
-----------------------
    0   background          (reserved; Faster R-CNN treats 0 as background)
    1   red blood cell
    2   trophozoite
    3   ring
    4   schizont
    5   gametocyte
    6   leukocyte

"difficult" annotations are skipped during data preparation and do NOT
appear in this map.
"""

# ── Core mappings ─────────────────────────────────────────────────────────────

LABEL_TO_INT: dict[str, int] = {
    "background":    0,
    "red blood cell": 1,
    "trophozoite":   2,
    "ring":          3,
    "schizont":      4,
    "gametocyte":    5,
    "leukocyte":     6,
}

INT_TO_LABEL: dict[int, str] = {v: k for k, v in LABEL_TO_INT.items()}

# ── Derived constants ─────────────────────────────────────────────────────────

NUM_CLASSES: int = len(LABEL_TO_INT)          # 7  (includes background)
NUM_FOREGROUND_CLASSES: int = NUM_CLASSES - 1  # 6

BACKGROUND_IDX: int = 0

# Parasitic stage classes (infected cells) — used for imbalance weighting
PARASITE_CLASSES: list[str] = ["trophozoite", "ring", "schizont", "gametocyte"]
PARASITE_INDICES: list[int] = [LABEL_TO_INT[c] for c in PARASITE_CLASSES]

# All foreground class names in index order (useful for tables / confusion matrix axes)
FOREGROUND_CLASSES: list[str] = [
    INT_TO_LABEL[i] for i in range(1, NUM_CLASSES)
]

# Abbreviations for compact tables and plot axes
LABEL_ABBREV: dict[str, str] = {
    "red blood cell": "RBC",
    "trophozoite":    "Troph",
    "ring":           "Ring",
    "schizont":       "Schiz",
    "gametocyte":     "Gam",
    "leukocyte":      "Leuk",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def encode(label: str) -> int:
    """Return the integer index for a class name string.
    Raises KeyError for any label not in LABEL_TO_INT.
    """
    if label not in LABEL_TO_INT:
        raise KeyError(
            f"Unknown label '{label}'. "
            f"Valid labels: {list(LABEL_TO_INT.keys())}"
        )
    return LABEL_TO_INT[label]


def decode(idx: int) -> str:
    """Return the class name string for an integer index.
    Raises KeyError for any index not in INT_TO_LABEL.
    """
    if idx not in INT_TO_LABEL:
        raise KeyError(
            f"Unknown index {idx}. "
            f"Valid indices: {list(INT_TO_LABEL.keys())}"
        )
    return INT_TO_LABEL[idx]


def is_parasite(label: str) -> bool:
    """True if the class is an infected/parasitic stage."""
    return label in PARASITE_CLASSES


# ── Self-test (run as script to verify) ───────────────────────────────────────

if __name__ == "__main__":
    print("MalariAI Label Map")
    print("=" * 36)
    for idx in range(NUM_CLASSES):
        name = decode(idx)
        abbrev = LABEL_ABBREV.get(name, "-")
        parasite = "★" if is_parasite(name) else " "
        print(f"  {idx}  {parasite}  {name:<20}  ({abbrev})")
    print()
    print(f"Total classes (incl. background): {NUM_CLASSES}")
    print(f"Foreground classes:               {NUM_FOREGROUND_CLASSES}")
    print(f"Parasite stage indices:           {PARASITE_INDICES}")
