3D Indoor Change Detection using Point Clouds

This project detects changes in indoor environments by comparing two 3D point-cloud scans captured at different times.
It identifies added, removed, or moved objects using geometric analysis, clustering, and 2D map generation.

The system works fully end-to-end:
Raw PLY files → Preprocessing → ICP alignment → Change detection → Clustering → 3D & 2D visualizations.

📂 Structure

indoor-change-detection/
│
├── data/
│   ├── room_t1.ply        # input scan at time T1
│   ├── room_t2.ply        # input scan at time T2 (or generated)
│
├── output/
│   ├── visualizations/    # 3D views, clusters
│   └── maps/              # top-down 2D change maps
│
├── src/
│   ├── load_and_preprocess.py
│   ├── register_icp.py
│   ├── detect_changes.py
│   ├── topdown_map.py
│   └── visualize.py
│
├── scripts/
│   └── synthetic_changes.py
│
├── main.py
└── README.md

🔍 How It Works

1)Preprocessing:
 -Load PLY files
 -Downsample (voxel grid)
 -Remove outliers

2)Scan Alignment (ICP)
 -Align T₂ onto T₁
 -Ensures both are in same coordinate frame

3)Change Detection
 -Compute nearest-neighbor distances
 -Points with high distance → changes
 -Red = removed (in T₁)
 -Green = added (in T₂)
 
4)Clustering (DBSCAN)
 -Group changed points into objects
 -Detect obstacles, moved furniture, etc.

5)Top-Down Map
 -Project changed points to XY plane
 -Generate a 2D occupancy-like image
