"""Pair classifier and probability scoring.

The initial model is LightGBM over engineered pair features. Thresholds are selected
against entity-level macro F0.5 rather than assuming 0.5 is optimal.
"""
