import os

from src.load_and_preprocess import load_pointcloud
from src.register_icp import align_icp
from src.detect_changes import compute_change_masks, cluster_changed_points
from src.topdown_map import create_topdown_change_map, save_topdown_image
from src.visualize import show_3d_changes, show_clusters, save_2d_map


def main():
    # ---- Paths ----
    data_dir = "data"
    out_dir = "output"
    vis_dir = os.path.join(out_dir, "visualizations")
    maps_dir = os.path.join(out_dir, "maps")

    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(maps_dir, exist_ok=True)

    path_t1 = os.path.join(data_dir, "room_t1.ply")
    path_t2 = os.path.join(data_dir, "room_t2.ply")

    # ---- 1. Load & preprocess ----
    print("[INFO] Loading point clouds...")
    pcd_t1 = load_pointcloud(path_t1, voxel_size=0.03)
    pcd_t2 = load_pointcloud(path_t2, voxel_size=0.03)

    # ---- 2. Registration (align t2 to t1) ----
    print("[INFO] Aligning t2 to t1 with ICP...")
    pcd_t2_aligned = align_icp(pcd_t2, pcd_t1, max_correspondence_distance=0.1)

    # ---- 3. Change detection: masks ----
    print("[INFO] Computing change masks...")
    mask_removed_t1, mask_new_t2 = compute_change_masks(
        pcd_t1,
        pcd_t2_aligned,
        dist_threshold=0.08,
    )

    print(f"[INFO] Removed points (t1): {mask_removed_t1.sum()}")
    print(f"[INFO] New points (t2): {mask_new_t2.sum()}")

    # ---- 4. Clustering changed points ----
    print("[INFO] Clustering new changes...")
    clusters_new = cluster_changed_points(
        pcd_t2_aligned,
        mask_new_t2,
        eps=0.2,
        min_points=30,
    )

    print(f"[INFO] Found {len(clusters_new)} new-change clusters.")

    # ---- 5. Top-down map of new changes ----
    print("[INFO] Creating top-down change map for new points...")
    grid, extent = create_topdown_change_map(
        pcd_t2_aligned,
        mask_new_t2,
        resolution=0.05,
        height_range=(0.1, 2.0),
    )

    out_map_path = os.path.join(maps_dir, "change_map.png")
    save_topdown_image(grid, out_map_path)
    print(f"[INFO] Saved top-down change map to: {out_map_path}")
    print(f"[INFO] Map extent (xmin, xmax, ymin, ymax): {extent}")

    # ---- 6. Visualizations ----
    print("[INFO] Showing 3D visualization (may require GUI)...")
    show_3d_changes(pcd_t1, pcd_t2_aligned, mask_removed_t1, mask_new_t2)

    print("[INFO] Showing clusters of new changes...")
    show_clusters(clusters_new, window_name="New Change Clusters")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
