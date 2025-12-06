from typing import List

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def show_3d_changes(
    pcd_t1: o3d.geometry.PointCloud,
    pcd_t2_aligned: o3d.geometry.PointCloud,
    mask_removed_t1: np.ndarray,
    mask_new_t2: np.ndarray,
) -> None:
    """
    Visualize two clouds with changed points highlighted:
    - t1: removed points in red
    - t2: new points in green
    """
    pcd1_vis = o3d.geometry.PointCloud(pcd_t1)
    pcd2_vis = o3d.geometry.PointCloud(pcd_t2_aligned)

    pts1 = np.asarray(pcd1_vis.points)
    pts2 = np.asarray(pcd2_vis.points)

    colors1 = np.tile([0.6, 0.6, 0.6], (len(pts1), 1))  # gray
    colors1[mask_removed_t1] = [1.0, 0.0, 0.0]  # red = removed
    pcd1_vis.colors = o3d.utility.Vector3dVector(colors1)

    colors2 = np.tile([0.6, 0.6, 0.6], (len(pts2), 1))
    colors2[mask_new_t2] = [0.0, 1.0, 0.0]  # green = new
    pcd2_vis.colors = o3d.utility.Vector3dVector(colors2)

    o3d.visualization.draw_geometries([pcd1_vis, pcd2_vis])


def show_clusters(
    clusters: List[o3d.geometry.PointCloud],
    window_name: str = "Change Clusters",
) -> None:
    """
    Display all cluster point clouds, each in a different random color.
    """
    geometries = []
    for cluster in clusters:
        pts = np.asarray(cluster.points)
        if pts.shape[0] == 0:
            continue
        color = np.random.rand(3)
        colors = np.tile(color, (len(pts), 1))
        cluster_vis = o3d.geometry.PointCloud(cluster)
        cluster_vis.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(cluster_vis)

    if len(geometries) == 0:
        print("No clusters to visualize.")
        return

    o3d.visualization.draw_geometries(geometries, window_name=window_name)


def save_2d_map(grid, out_path: str) -> None:
    """
    Save a 2D numpy array as an image (simple wrapper).
    """
    plt.figure()
    plt.imshow(grid, origin="lower")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()
