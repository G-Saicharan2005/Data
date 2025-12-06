import numpy as np
import open3d as o3d
from typing import Tuple, List


def _compute_nn_distances(
    src: o3d.geometry.PointCloud,
    dst_tree: o3d.geometry.KDTreeFlann,
) -> np.ndarray:
    """
    For each point in `src`, compute nearest-neighbor distance to points in `dst_tree`.
    """
    points = np.asarray(src.points)
    dists = np.empty(len(points), dtype=np.float32)

    for i, pt in enumerate(points):
        k, idx, dist2 = dst_tree.search_knn_vector_3d(pt, 1)
        if k == 0:
            dists[i] = np.inf
        else:
            dists[i] = float(np.sqrt(dist2[0]))

    return dists


def compute_change_masks(
    pcd_t1: o3d.geometry.PointCloud,
    pcd_t2_aligned: o3d.geometry.PointCloud,
    dist_threshold: float = 0.08,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute boolean masks indicating changed points in each cloud.
    """
    tree_t1 = o3d.geometry.KDTreeFlann(pcd_t1)
    tree_t2 = o3d.geometry.KDTreeFlann(pcd_t2_aligned)

    d1 = _compute_nn_distances(pcd_t1, tree_t2)
    d2 = _compute_nn_distances(pcd_t2_aligned, tree_t1)

    mask_removed = d1 > dist_threshold
    mask_new = d2 > dist_threshold

    return mask_removed, mask_new


def cluster_changed_points(
    pcd: o3d.geometry.PointCloud,
    change_mask: np.ndarray,
    eps: float = 0.2,
    min_points: int = 20,
) -> List[o3d.geometry.PointCloud]:
    """
    Cluster changed points using DBSCAN.
    """
    pts = np.asarray(pcd.points)
    changed_pts = pts[change_mask]

    if changed_pts.shape[0] == 0:
        return []

    changed_pcd = o3d.geometry.PointCloud()
    changed_pcd.points = o3d.utility.Vector3dVector(changed_pts)

    labels = np.array(
        changed_pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
    )

    if labels.max() < 0:
        return []

    clusters = []
    for cid in range(labels.max() + 1):
        mask = labels == cid
        cluster_pcd = o3d.geometry.PointCloud()
        cluster_pcd.points = o3d.utility.Vector3dVector(changed_pts[mask])
        clusters.append(cluster_pcd)

    return clusters
