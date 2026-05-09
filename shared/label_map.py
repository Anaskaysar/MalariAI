"""
shared/label_map.py
Single source of truth for class definitions across ALL phases.

Import from any phase like this:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from shared.label_map import LABEL_TO_INT, INT_TO_LABEL, NUM_CLASSES

Class index layout
------------------
  0  background      (Faster R-CNN reserves 0 for background — do not use for cells)
  1  red blood cell
  2  trophozoite
  3  ring
  4  schizont
  5  gametocyte
  6  leukocyte

"difficult" annotations are SKIPPED during parsing — not included here.
"""

LABEL_TO_INT = {
    "background":     0,
    "red blood cell": 1,
    "trophozoite":    2,
    "ring":           3,
    "schizont":       4,
    "gametocyte":     5,
    "leukocyte":      6,
}

INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}

NUM_CLASSES          = 7   # includes background
NUM_FOREGROUND       = 6   # excludes background

# Parasitic / infected stages
PARASITE_CLASSES  = ["trophozoite", "ring", "schizont", "gametocyte"]
PARASITE_INDICES  = [LABEL_TO_INT[c] for c in PARASITE_CLASSES]

# All foreground names in index order (1..6)
FOREGROUND_NAMES  = [INT_TO_LABEL[i] for i in range(1, NUM_CLASSES)]

# Short display names for plots and tables
SHORT_NAME = {
    "red blood cell": "RBC",
    "trophozoite":    "Troph",
    "ring":           "Ring",
    "schizont":       "Schiz",
    "gametocyte":     "Gam",
    "leukocyte":      "Leuk",
    "background":     "BG",
}

# Colours per class (BGR for OpenCV, RGB for matplotlib)
CLASS_COLOUR_RGB = {
    "red blood cell": (220, 50,  50),
    "trophozoite":    (50,  180, 50),
    "ring":           (50,  50,  220),
    "schizont":       (200, 130, 0),
    "gametocyte":     (160, 0,   200),
    "leukocyte":      (0,   180, 200),
    "background":     (128, 128, 128),
}


def encode(label: str) -> int:
    if label not in LABEL_TO_INT:
        raise KeyError(f"Unknown label '{label}'. Valid: {list(LABEL_TO_INT.keys())}")
    return LABEL_TO_INT[label]


def decode(idx: int) -> str:
    if idx not in INT_TO_LABEL:
        raise KeyError(f"Unknown index {idx}. Valid: {list(INT_TO_LABEL.keys())}")
    return INT_TO_LABEL[idx]


if __name__ == "__main__":
    print(f"{'IDX':<4} {'CLASS':<22} {'SHORT':<8} {'PARASITE'}")
    print("-" * 45)
    for i in range(NUM_CLASSES):
        name = decode(i)
        print(f"{i:<4} {name:<22} {SHORT_NAME[name]:<8} {'yes' if name in PARASITE_CLASSES else ''}")
    print(f"\nTotal classes (incl. background): {NUM_CLASSES}")
    print(f"Foreground classes:               {NUM_FOREGROUND}")
