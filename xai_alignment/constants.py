"""Shared names and experiment defaults."""

METHODS = ("GC", "GGC", "IG", "SL")
GT_TYPES = ("region", "boundary", "bbox", "point")
FORMULATIONS = ("soft", "hard")
METRIC_DIRECTIONS = {
    "iou": 1,
    "pg": 1,
    "epg": 1,
    "ber": -1,
    "distance": -1,
}
ANALYSIS_METRICS = ("iou", "pg", "epg", "distance")
