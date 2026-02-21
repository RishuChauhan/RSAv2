"""
Centralized constants for the Rifle Shooting Analysis application.
"""

# Thresholds for sway velocity (mm/s)
SWAY_LOW_THRESHOLD = 5.0       # mm/s - considered stable
SWAY_HIGH_THRESHOLD = 10.0     # mm/s - considered unstable

# Thresholds for postural deviation (px)
DEV_LOW_THRESHOLD = 10.0
DEV_HIGH_THRESHOLD = 20.0

# Follow-through score thresholds (0-1)
FOLLOW_THROUGH_POOR = 0.35
FOLLOW_THROUGH_GOOD = 0.70

# Recording and Metrics
MAX_RECORDING_METRICS = 300    # 5 fps × 60s
POST_SHOT_WAIT_MS = 1500       # 1.5 seconds

# Fuzzy Logic ranges (max values)
FUZZY_SWAY_MAX = 20.0
FUZZY_DEV_MAX = 30.0

# Colors
COLOR_RED = "#E53935"
COLOR_YELLOW = "#FFB300"
COLOR_GREEN = "#43A047"
