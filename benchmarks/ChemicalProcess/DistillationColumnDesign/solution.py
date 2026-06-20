"""Baseline: conservative design (10 trays, high reflux, center feed)."""
import numpy as np
def design_column(alpha, x_feed, x_dist_target, x_bot_target):
    return {"n_trays": 10, "reflux_ratio": 5.0, "feed_tray": 5}
