import copy
import open3d as o3d


def align_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    max_correspondence_distance: float = 0.1,
    max_iterations: int = 50,
    init_transform=None,
) -> o3d.geometry.PointCloud:
    """
    Align `source` to `target` using ICP and return a transformed copy of `source`.

    Parameters
    ----------
    source : open3d.geometry.PointCloud
        Point cloud to be aligned.
    target : open3d.geometry.PointCloud
        Reference point cloud.
    max_correspondence_distance : float
        Maximum correspondence distance (in meters).
    max_iterations : int
        Maximum ICP iterations.
    init_transform : np.ndarray or None
        Optional 4x4 initial guess.

    Returns
    -------
    aligned_source : open3d.geometry.PointCloud
        Copy of source transformed into target's frame.
    """
    if init_transform is None:
        init_transform = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, 0])
        init_transform = [[1.0, 0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0],
                          [0.0, 0.0, 0.0, 1.0]]

    # Use point-to-plane ICP (better for structured scenes)
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=max_iterations
    )

    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_correspondence_distance,
        init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria,
    )

    aligned = copy.deepcopy(source)
    aligned.transform(result.transformation)
    return aligned
