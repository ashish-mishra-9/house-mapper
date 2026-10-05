"""
Gravity Aligner: Transform point cloud so floor is at Z=0 and walls align with axes.

ARKit uses Y-up world coordinates. This module:
1. Rotates from Y-up to Z-up (known from ARKit convention)
2. Detects the exact floor Z via RANSAC plane fitting
3. Translates so floor is at Z=0
4. Applies Manhattan world alignment so walls are axis-aligned
"""

import open3d as o3d
import numpy as np
import copy


def align_to_gravity(point_cloud):
    """
    Align point cloud so floor is at Z≈0 and dominant wall directions align with X/Y axes.

    The input point cloud is assumed to be in ARKit world coordinates (Y-up).
    We convert to Z-up, detect the floor, and align walls to axes.

    Args:
        point_cloud: o3d.geometry.PointCloud

    Returns:
        dict with:
            'point_cloud': aligned o3d.geometry.PointCloud
            'floor_z': float - Z coordinate of the floor plane (≈0)
            'ceiling_z': float - Z coordinate of the ceiling plane
            'height': float - floor-to-ceiling height
            'transform': np.array(4,4) - the full transform applied
    """
    pcd = copy.deepcopy(point_cloud)

    # ── Step 1: ARKit Y-up → Z-up rotation ──
    # ARKit:  X=right, Y=up, Z=towards-viewer
    # Target: X=right, Y=forward, Z=up
    # Rotation: -90° around X axis (Y→Z, Z→-Y)
    R_yup_to_zup = np.array([
        [1,  0,  0],
        [0,  0, -1],
        [0,  1,  0],
    ], dtype=np.float64)

    T_yup = np.eye(4)
    T_yup[:3, :3] = R_yup_to_zup
    pcd.transform(T_yup)
    print("  Applied ARKit Y-up → Z-up rotation")

    # ── Step 2: Detect floor plane with RANSAC ──
    pts = np.asarray(pcd.points)
    z_vals = pts[:, 2]

    # The floor should be the lowest large horizontal plane
    # First, try RANSAC on the bottom 30% of points (likely floor region)
    z_threshold = np.percentile(z_vals, 30)
    bottom_mask = z_vals <= z_threshold
    bottom_indices = np.where(bottom_mask)[0]

    floor_z_raw = None
    if len(bottom_indices) > 100:
        bottom_pcd = pcd.select_by_index(bottom_indices)
        try:
            plane_model, inliers = bottom_pcd.segment_plane(
                distance_threshold=0.05, ransac_n=3, num_iterations=1000
            )
            a, b, c, d = plane_model
            normal = np.array([a, b, c])
            normal /= np.linalg.norm(normal)

            # Check if it's roughly horizontal (normal ≈ [0, 0, ±1])
            if abs(normal[2]) > 0.7:
                # Floor Z = -d/c (from ax + by + cz + d = 0, at x=y=0)
                inlier_pts = np.asarray(bottom_pcd.points)[inliers]
                floor_z_raw = float(np.median(inlier_pts[:, 2]))
                print(f"  RANSAC floor plane found: normal={normal}, z={floor_z_raw:.3f}m")

                # Fine-tune gravity alignment if the floor normal isn't perfectly [0,0,1]
                if abs(normal[2]) < 0.999:
                    # Compute small rotation to make floor exactly horizontal
                    if normal[2] < 0:
                        normal = -normal
                    axis = np.cross(normal, [0, 0, 1])
                    axis_norm = np.linalg.norm(axis)
                    if axis_norm > 1e-6:
                        axis /= axis_norm
                        angle = np.arccos(np.clip(normal[2], -1, 1))
                        # Rodrigues rotation formula
                        K = np.array([
                            [0, -axis[2], axis[1]],
                            [axis[2], 0, -axis[0]],
                            [-axis[1], axis[0], 0]
                        ])
                        R_fine = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
                        T_fine = np.eye(4)
                        T_fine[:3, :3] = R_fine
                        pcd.transform(T_fine)
                        T_yup = T_fine @ T_yup
                        # Recompute floor z after fine rotation
                        pts = np.asarray(pcd.points)
                        z_vals = pts[:, 2]
                        z_threshold = np.percentile(z_vals, 30)
                        bottom_indices_new = np.where(z_vals <= z_threshold)[0]
                        if len(bottom_indices_new) > 10:
                            floor_z_raw = float(np.percentile(z_vals[bottom_indices_new], 50))
                        print(f"  Fine-tuned gravity: floor_z={floor_z_raw:.3f}m")
        except Exception as e:
            print(f"  RANSAC failed: {e}")

    if floor_z_raw is None:
        # Fallback: use the 5th percentile of Z as floor estimate
        floor_z_raw = float(np.percentile(z_vals, 5))
        print(f"  Using Z percentile fallback: floor_z={floor_z_raw:.3f}m")

    # ── Step 3: Translate so floor is at Z=0 ──
    T_translate = np.eye(4)
    T_translate[2, 3] = -floor_z_raw
    pcd.transform(T_translate)

    pts = np.asarray(pcd.points)
    z_vals = pts[:, 2]

    # ── Step 4: Detect ceiling ──
    # Ignore distant returns and furniture; typical residential ceilings are 2.0-3.0m.
    ceiling_min, ceiling_max = 2.0, 3.0
    ceiling_points = z_vals[(z_vals >= ceiling_min) & (z_vals <= ceiling_max)]
    if len(ceiling_points) > 100:
        hist, bin_edges = np.histogram(ceiling_points, bins=40, range=(ceiling_min, ceiling_max))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        peak_index = int(np.argmax(hist))
        ceiling_z = float(bin_centers[peak_index]) if 0 < peak_index < len(hist) - 1 else 2.5
    else:
        ceiling_z = 2.5

    height = ceiling_z
    floor_z = 0.0

    print(f"  Ceiling at Z={ceiling_z:.3f}m, height={height:.3f}m")

    # ── Step 5: Manhattan world alignment ──
    # Use a narrow band that avoids floor, furniture, and ceiling returns.
    wall_z_min = 1.0
    wall_z_max = min(1.5, height - 0.5)
    wall_mask = (z_vals >= wall_z_min) & (z_vals <= wall_z_max)
    wall_pts = pts[wall_mask]

    T_manhattan = np.eye(4)
    if len(wall_pts) > 200:
        # Use PCA on the 2D projection of wall points
        xy = wall_pts[:, :2]
        xy_centered = xy - np.mean(xy, axis=0)
        cov = np.cov(xy_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # The dominant direction (largest eigenvalue) should align with X-axis
        dominant_dir = eigenvectors[:, 1]  # eigenvectors sorted ascending
        angle = np.arctan2(dominant_dir[1], dominant_dir[0])

        # Snap to nearest 90° if close
        snap_angles = [0, np.pi/2, np.pi, -np.pi/2, -np.pi]
        snap_diffs = [abs(angle - sa) for sa in snap_angles]
        min_diff = min(snap_diffs)
        if min_diff < np.deg2rad(5):
            angle = snap_angles[snap_diffs.index(min_diff)]

        # Rotate by -angle around Z
        cos_a, sin_a = np.cos(-angle), np.sin(-angle)
        R_manhattan = np.array([
            [cos_a, -sin_a, 0],
            [sin_a,  cos_a, 0],
            [0,      0,     1],
        ])
        T_manhattan[:3, :3] = R_manhattan
        pcd.transform(T_manhattan)
        print(f"  Manhattan alignment: rotated {np.degrees(angle):.1f}° around Z")
    else:
        print("  Skipping Manhattan alignment (too few wall points)")

    # ── Compose full transform ──
    T_total = T_manhattan @ T_translate @ T_yup

    return {
        'point_cloud': pcd,
        'floor_z': floor_z,
        'ceiling_z': ceiling_z,
        'height': height,
        'transform': T_total,
    }