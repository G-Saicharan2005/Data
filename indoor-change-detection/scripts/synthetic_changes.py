import os
import numpy as np
import open3d as o3d


def add_box_obstacle(pcd, center, size=(0.5, 0.5, 0.7), density=5000):
    """
    Add a synthetic box-shaped cluster of points to the point cloud.

    center: (x, y, z)
    size: (sx, sy, sz)
    """
    cx, cy, cz = center
    sx, sy, sz = size

    # Sample random points within the box
    xs = np.random.uniform(cx - sx / 2, cx + sx / 2, density)
    ys = np.random.uniform(cy - sy / 2, cy + sy / 2, density)
    zs = np.random.uniform(cz - sz / 2, cz + sz / 2, density)

    pts = np.stack([xs, ys, zs], axis=1)
    obstacle = o3d.geometry.PointCloud()
    obstacle.points = o3d.utility.Vector3dVector(pts)

    return obstacle


def remove_region(pcd, min_xyz, max_xyz):
    """
    Remove points from pcd that fall within the axis-aligned box [min_xyz, max_xyz].
    """
    pts = np.asarray(pcd.points)
    mask_keep = ~(
        (pts[:, 0] >= min_xyz[0]) & (pts[:, 0] <= max_xyz[0]) &
        (pts[:, 1] >= min_xyz[1]) & (pts[:, 1] <= max_xyz[1]) &
        (pts[:, 2] >= min_xyz[2]) & (pts[:, 2] <= max_xyz[2])
    )
    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(pts[mask_keep])
    return new_pcd


def main():
    data_dir = "data"
    in_path = os.path.join(data_dir, "room_t1.ply")
    out_path = os.path.join(data_dir, "room_t2.ply")

    print(f"[INFO] Loading {in_path}")
    pcd = o3d.io.read_point_cloud(in_path)
    if pcd.is_empty():
        raise ValueError("room_t1.ply is empty or missing!")

    pts = np.asarray(pcd.points)
    xmin, ymin, zmin = pts.min(axis=0)
    xmax, ymax, zmax = pts.max(axis=0)

    # 1) Remove a region near one corner (simulate moved furniture)
    remove_min = (xmin + 0.1 * (xmax - xmin),
                  ymin + 0.1 * (ymax - ymin),
                  zmin)
    remove_max = (xmin + 0.3 * (xmax - xmin),
                  ymin + 0.5 * (ymax - ymin),
                  zmin + 1.5)  # up to 1.5m height

    print("[INFO] Removing region to simulate object removal...")
    pcd_mod = remove_region(pcd, remove_min, remove_max)

    # 2) Add a box obstacle in the middle of the room
    center = (0.5 * (xmin + xmax),
              0.5 * (ymin + ymax),
              zmin + 0.5)  # 0.5m high

    print("[INFO] Adding synthetic obstacle...")
    obstacle = add_box_obstacle(pcd_mod, center=center, size=(0.6, 0.6, 0.7), density=8000)

    pcd_mod += obstacle

    print(f"[INFO] Saving modified point cloud to {out_path}")
    o3d.io.write_point_cloud(out_path, pcd_mod)


if __name__ == "__main__":
    main()
