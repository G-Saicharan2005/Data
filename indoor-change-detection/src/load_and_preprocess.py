import open3d as o3d


def load_pointcloud(
    path: str,
    voxel_size: float = 0.03,
    remove_outliers: bool = True,
) -> o3d.geometry.PointCloud:
    """
    Load a point cloud from disk and apply basic preprocessing:
    - voxel downsampling
    - optional statistical outlier removal
    """
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        raise ValueError(f"Loaded point cloud is empty: {path}")

    # Voxel downsample
    if voxel_size is not None and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    # Estimate normals (useful for ICP point-to-plane, later extensions)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2 if voxel_size else 0.1,
            max_nn=30,
        )
    )

    # Optional noise removal
    if remove_outliers:
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=20,
            std_ratio=2.0,
        )

    return pcd
