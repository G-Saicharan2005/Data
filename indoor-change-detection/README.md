3D Indoor Change Detection using Point Clouds

This project detects changes in indoor environments by comparing two 3D point-cloud scans captured at different times.
It identifies added, removed, or moved objects using geometric analysis, clustering, and 2D map generation.

The system works fully end-to-end:
Raw PLY files → Preprocessing → ICP alignment → Change detection → Clustering → 3D & 2D visualizations.

🔍 How It Works (Short Explanation)

Preprocessing

Load PLY files

Downsample (voxel grid)

Remove outliers

Scan Alignment (ICP)

Align T₂ onto T₁

Ensures both are in same coordinate frame

Change Detection

Compute nearest-neighbor distances

Points with high distance → changes

Red = removed (in T₁)

Green = added (in T₂)

Clustering (DBSCAN)

Group changed points into objects

Detect obstacles, moved furniture, etc.

Top-Down Map

Project changed points to XY plane

Generate a 2D occupancy-like image
