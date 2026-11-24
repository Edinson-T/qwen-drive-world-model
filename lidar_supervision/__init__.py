"""Standalone LiDAR-supervision sandbox for Qwen-Drive.

This package is intentionally separate from ``src/qwen_drive`` and
``src/qwen_drive_perception``: it imports nothing from them and never writes
into them. It only reads the bundled demo frames under ``data/demo/perception``.

Purpose: validate the LiDAR -> camera -> depth-target pipeline and prove that a
small geometry head can learn, before any large training run is attempted.
"""

__version__ = "0.1.0"
