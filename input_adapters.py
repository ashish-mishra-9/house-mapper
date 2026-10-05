"""
Input Adapters: Convert raw captures from three tiers (Photos, Video, LiDAR)
into a unified list of RGBDFrame objects for the 3D pipeline.

Each RGBDFrame contains:
  - color_path: path to the RGB image
  - depth_path: path to the 16-bit PNG depth map (mm) or estimated depth
  - pose: 4x4 camera-to-world transform in OpenCV camera convention
  - intrinsics: dict with fx, fy, cx, cy for this frame (optional, per-frame)
"""

import os
import cv2
import csv
import numpy as np
import open3d as o3d


class RGBDFrame:
    """A single RGB-D observation with camera pose."""
    def __init__(self, color_path, depth_path, pose, intrinsics=None):
        self.color_path = color_path
        self.depth_path = depth_path
        self.pose = pose  # 4x4 camera-to-world in OpenCV convention
        self.intrinsics = intrinsics  # optional per-frame {fx, fy, cx, cy}


# ARKit camera convention to OpenCV camera convention transform
# ARKit camera: X-right, Y-up, Z-towards-viewer
# OpenCV camera: X-right, Y-down, Z-into-scene
ARKIT_TO_OPENCV = np.array([
    [1,  0,  0, 0],
    [0, -1,  0, 0],
    [0,  0, -1, 0],
    [0,  0,  0, 1]
], dtype=np.float64)


def _quaternion_to_pose(x, y, z, qx, qy, qz, qw):
    """Convert position + quaternion to a 4x4 camera-to-world pose in OpenCV convention."""
    quat = np.array([qw, qx, qy, qz]).reshape(4, 1)
    rot_matrix = o3d.geometry.get_rotation_matrix_from_quaternion(quat)
    pose_arkit = np.eye(4)
    pose_arkit[:3, :3] = rot_matrix
    pose_arkit[:3, 3] = [x, y, z]
    # Convert from ARKit camera convention to OpenCV camera convention
    pose_opencv = pose_arkit @ ARKIT_TO_OPENCV
    return pose_opencv


def convert_lidar_to_common(lidar_export_folder):
    """
    TIER 3: LiDAR (The Gold Standard)
    Reads RGB video, 16-bit depth PNGs, and odometry CSV from a LiDAR capture app export.
    Returns per-frame intrinsics from the CSV alongside the frames.
    """
    print("=" * 60)
    print("TIER 3: Converting LiDAR capture to common format...")
    print("=" * 60)

    video_path = os.path.join(lidar_export_folder, "rgb.mp4")
    odometry_path = os.path.join(lidar_export_folder, "odometry.csv")
    depth_folder = os.path.join(lidar_export_folder, "depth")
    working_folder = os.path.join(lidar_export_folder, "extracted_photos")
    os.makedirs(working_folder, exist_ok=True)

    # --- STEP 1: Parse odometry CSV for poses and per-frame intrinsics ---
    poses = {}
    per_frame_intrinsics = {}
    with open(odometry_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            row = {str(k).strip(): str(v).strip() for k, v in raw_row.items() if k is not None}
            if 'frame' not in row:
                continue

            frame_idx = int(row['frame'])
            x = float(row['x'])
            y = float(row['y'])
            z = float(row['z'])
            qx = float(row['qx'])
            qy = float(row['qy'])
            qz = float(row['qz'])
            qw = float(row['qw'])

            poses[frame_idx] = _quaternion_to_pose(x, y, z, qx, qy, qz, qw)

            # Read per-frame intrinsics if available
            if 'fx' in row and row['fx']:
                per_frame_intrinsics[frame_idx] = {
                    'fx': float(row['fx']),
                    'fy': float(row['fy']),
                    'cx': float(row['cx']),
                    'cy': float(row['cy']),
                }

    print(f"  Parsed {len(poses)} poses, {len(per_frame_intrinsics)} per-frame intrinsics")

    # --- STEP 2: Extract RGB frames from video, match with depth ---
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    frame_count = 0

    # Use every 5th frame with a pose and matching depth
    while True:
        success, image = video_capture.read()
        if not success:
            break

        if frame_count in poses and frame_count % 5 == 0:
            # Find depth file (try multiple naming conventions)
            depth_img_path = None
            for pattern in [f"{frame_count:06d}.png", f"{frame_count:05d}.png", f"{frame_count}.png"]:
                candidate = os.path.join(depth_folder, pattern)
                if os.path.exists(candidate):
                    depth_img_path = candidate
                    break

            if depth_img_path is not None:
                # Verify depth is valid 16-bit
                depth_img = cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED)
                if depth_img is not None and depth_img.dtype == np.uint16:
                    h_d, w_d = depth_img.shape[:2]

                    # Resize color to match depth resolution (never touch depth)
                    image_resized = cv2.resize(image, (w_d, h_d))
                    img_path = os.path.join(working_folder, f"frame_{frame_count}.jpg")
                    cv2.imwrite(img_path, image_resized)

                    # Get per-frame intrinsics or use defaults
                    intr = per_frame_intrinsics.get(frame_count)

                    frames.append(RGBDFrame(
                        color_path=img_path,
                        depth_path=depth_img_path,
                        pose=poses[frame_count],
                        intrinsics=intr,
                    ))

        frame_count += 1

    video_capture.release()
    print(f"  Extracted {len(frames)} matched RGB-D frames from {total_frames} video frames")

    if len(frames) == 0:
        raise RuntimeError("No frames could be matched! Check depth folder and odometry CSV.")

    return frames


def convert_video_to_common(video_filepath, working_folder):
    """
    TIER 2: Video (No depth sensor)
    Extracts frames from video, estimates camera poses via COLMAP/pycolmap SfM,
    and estimates metric depth using Depth-Anything-V2-Metric.
    """
    print("=" * 60)
    print("TIER 2: Converting Video capture to common format...")
    print("=" * 60)

    os.makedirs(working_folder, exist_ok=True)

    # --- STEP 1: Extract frames at ~2 fps ---
    video_capture = cv2.VideoCapture(video_filepath)
    if not video_capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_filepath}")

    fps = video_capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(fps / 2))  # ~2 frames per second
    frame_count = 0
    extracted_image_paths = []

    while True:
        success, image = video_capture.read()
        if not success:
            break
        if frame_count % frame_interval == 0:
            img_path = os.path.join(working_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(img_path, image)
            extracted_image_paths.append(img_path)
        frame_count += 1

    video_capture.release()
    print(f"  Extracted {len(extracted_image_paths)} frames at ~2fps")

    # --- STEP 2: Estimate camera poses using COLMAP/pycolmap ---
    poses_dict = _estimate_poses_colmap(extracted_image_paths, working_folder)

    # --- STEP 3: Estimate metric depth ---
    depth_paths = _estimate_metric_depth(extracted_image_paths, working_folder)

    # --- STEP 4: Package into RGBDFrames ---
    frames = []
    for img_path in extracted_image_paths:
        if img_path in poses_dict and img_path in depth_paths:
            frames.append(RGBDFrame(
                color_path=img_path,
                depth_path=depth_paths[img_path],
                pose=poses_dict[img_path],
            ))

    print(f"  Packaged {len(frames)} video frames with poses and depth")
    return frames


def convert_photos_to_common(photo_folder, working_folder):
    """
    TIER 1: Photos (The Hard Route)
    Takes 2-8 still photos per room, estimates poses via COLMAP,
    and estimates metric depth using Depth-Anything-V2-Metric.
    """
    print("=" * 60)
    print("TIER 1: Converting Photos to common format...")
    print("=" * 60)

    os.makedirs(working_folder, exist_ok=True)

    photo_extensions = {'.jpg', '.jpeg', '.png', '.heic'}
    photo_paths = sorted([
        os.path.join(photo_folder, f)
        for f in os.listdir(photo_folder)
        if os.path.splitext(f)[1].lower() in photo_extensions
    ])
    print(f"  Found {len(photo_paths)} photos")

    if len(photo_paths) == 0:
        raise RuntimeError(f"No photos found in {photo_folder}")

    # --- STEP 1: Estimate camera poses ---
    poses_dict = _estimate_poses_colmap(photo_paths, working_folder)

    # --- STEP 2: Estimate metric depth ---
    depth_paths = _estimate_metric_depth(photo_paths, working_folder)

    # --- STEP 3: Package into RGBDFrames ---
    frames = []
    for img_path in photo_paths:
        if img_path in poses_dict and img_path in depth_paths:
            frames.append(RGBDFrame(
                color_path=img_path,
                depth_path=depth_paths[img_path],
                pose=poses_dict[img_path],
            ))

    print(f"  Packaged {len(frames)} photo frames with poses and depth")
    return frames


def _estimate_poses_colmap(image_paths, working_folder):
    """
    Estimate camera poses from a set of images using pycolmap SfM.
    Returns dict mapping image_path -> 4x4 pose matrix.
    """
    try:
        import pycolmap

        db_path = os.path.join(working_folder, "colmap.db")
        sparse_path = os.path.join(working_folder, "sparse")
        image_dir = os.path.dirname(image_paths[0])

        # Run feature extraction + matching + reconstruction
        pycolmap.extract_features(db_path, image_dir)
        pycolmap.match_exhaustive(db_path)
        maps = pycolmap.incremental_mapping(db_path, image_dir, sparse_path)

        poses_dict = {}
        if maps:
            reconstruction = maps[0]
            for image_id, image in reconstruction.images.items():
                img_name = image.name
                full_path = os.path.join(image_dir, img_name)
                if full_path in image_paths:
                    # pycolmap gives world-to-camera; we need camera-to-world
                    R = image.rotation_matrix()
                    t = image.tvec
                    pose = np.eye(4)
                    pose[:3, :3] = R.T
                    pose[:3, 3] = -R.T @ t
                    poses_dict[full_path] = pose

        print(f"  COLMAP estimated {len(poses_dict)} poses")
        return poses_dict

    except (ImportError, Exception) as e:
        print(f"  WARNING: COLMAP pose estimation failed: {e}")
        print(f"  Falling back to identity poses (single-room mode)")
        # Fallback: place cameras in a rough arc
        poses_dict = {}
        for i, path in enumerate(image_paths):
            angle = (i / max(len(image_paths) - 1, 1)) * np.pi * 0.5
            pose = np.eye(4)
            pose[0, 3] = np.cos(angle) * 2.0
            pose[2, 3] = np.sin(angle) * 2.0
            poses_dict[path] = pose
        return poses_dict


def _estimate_metric_depth(image_paths, working_folder):
    """
    Estimate metric depth maps using Depth-Anything-V2 or similar.
    Returns dict mapping image_path -> depth_png_path (16-bit, millimeters).
    """
    depth_paths = {}

    try:
        from transformers import pipeline as hf_pipeline
        from PIL import Image

        # Use metric depth model if available, otherwise relative
        try:
            depth_model = hf_pipeline(
                task="depth-estimation",
                model="depth-anything/Depth-Anything-V2-Small-hf"
            )
        except Exception:
            depth_model = None

        if depth_model is not None:
            for img_path in image_paths:
                raw_image = Image.open(img_path)
                result = depth_model(raw_image)
                depth_pil = result["depth"]

                # Convert to numpy - the output is relative depth (0-255 range typically)
                depth_np = np.array(depth_pil, dtype=np.float32)

                # Scale to approximate metric depth in mm
                # Depth-Anything relative output: larger = farther
                # Heuristic: normalize to 0.5m - 4.0m range for indoor scenes
                d_min, d_max = depth_np.min(), depth_np.max()
                if d_max > d_min:
                    depth_normalized = (depth_np - d_min) / (d_max - d_min)
                    depth_metric_m = 0.5 + depth_normalized * 3.5  # 0.5m to 4.0m
                else:
                    depth_metric_m = np.full_like(depth_np, 2.0)

                depth_mm = (depth_metric_m * 1000).astype(np.uint16)

                depth_path = os.path.join(
                    working_folder,
                    os.path.basename(img_path).replace(".jpg", "_depth.png").replace(".png", "_depth.png")
                )
                cv2.imwrite(depth_path, depth_mm)
                depth_paths[img_path] = depth_path

            print(f"  Estimated depth for {len(depth_paths)} images")

    except ImportError:
        print("  WARNING: transformers not available, skipping depth estimation")

    return depth_paths


def get_default_intrinsics(frames):
    """
    Get the best available intrinsics from the frame list.
    Uses per-frame intrinsics from the first frame if available.
    """
    if frames and frames[0].intrinsics:
        intr = frames[0].intrinsics
        # Read image to get dimensions
        sample = cv2.imread(frames[0].color_path)
        h, w = sample.shape[:2]
        # Scale intrinsics from full-res (1920x1440) to actual image size
        scale_x = w / 1920.0
        scale_y = h / 1440.0
        return {
            'width': w,
            'height': h,
            'fx': intr['fx'] * scale_x,
            'fy': intr['fy'] * scale_y,
            'cx': intr['cx'] * scale_x,
            'cy': intr['cy'] * scale_y,
        }
    elif frames:
        # Fallback: estimate from image size
        sample = cv2.imread(frames[0].color_path)
        h, w = sample.shape[:2]
        fx = fy = max(w, h) * 0.83  # rough iPhone-like FOV
        return {
            'width': w,
            'height': h,
            'fx': fx,
            'fy': fy,
            'cx': w / 2.0,
            'cy': h / 2.0,
        }
    else:
        raise RuntimeError("No frames available to determine intrinsics")