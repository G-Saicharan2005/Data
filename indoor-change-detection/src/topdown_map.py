from typing import Tuple

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def create_topdown_change_map(
    pcd: o3d.geometry.PointCloud,
    change_mask: np.ndarray,
    resolution: float = 0.05,
    height_range: Tuple[float, float] = (0.1, 2.0),
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """
    Project changed points to a top-down occupancy-like grid.

    Parameters
    ----------
    pcd : PointCloud
        Point cloud of the scene.
    change_mask : np.ndarray (N, bool)
        Mask of changed points to project.
    resolution : float
        Grid resolution (meters per cell).
    height_range : (z_min, z_max)
        Only points with z in this range are considered.

    Returns
    -------
    grid : np.ndarray (H, W)
        2D array where cells with changes are > 0.
    extent : (xmin, xmax, ymin, ymax)
        Bounds of the grid in world coordinates.
    """
    pts = np.asarray(pcd.points)
    pts = pts[change_mask]

    if pts.shape[0] == 0:
        # Return an empty 1x1 grid to avoid downstream crashes.
        return np.zeros((1, 1), dtype=np.uint8), (0, 0, 0, 0)

    z_min, z_max = height_range
    mask_height = (pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)
    pts = pts[mask_height]

    if pts.shape[0] == 0:
        return np.zeros((1, 1), dtype=np.uint8), (0, 0, 0, 0)

    xs, ys = pts[:, 0], pts[:, 1]

    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()

    width = int(np.ceil((xmax - xmin) / resolution)) + 1
    height = int(np.ceil((ymax - ymin) / resolution)) + 1

    grid = np.zeros((height, width), dtype=np.uint8)

    # Convert world coordinates to grid indices
    ix = ((xs - xmin) / resolution).astype(int)
    iy = ((ys - ymin) / resolution).astype(int)

    # Note: iy is row index, ix is column index
    grid[iy, ix] = 255  # mark changed cells

    return grid, (xmin, xmax, ymin, ymax)


def save_topdown_image(grid: np.ndarray, out_path: str) -> None:
    """
    Save a 2D grid as an image.
    """
    plt.figure()
    plt.imshow(grid, origin="lower")  # origin lower so it matches XY convention
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()
