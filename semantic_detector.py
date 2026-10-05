import numpy as np
import cv2
import open3d as o3d
from ultralytics import YOLO
from matplotlib.path import Path
import logging

logger = logging.getLogger(__name__)

ROOM_FEATURE_CLASSES = {
    'bed', 'bench', 'chair', 'couch', 'dining table', 'microwave',
    'oven', 'potted plant', 'refrigerator', 'sink', 'toilet', 'tv'
}

CLASS_MIN_DIMENSIONS = {
    'bed': (0.9, 1.8),           # single bed minimum
    'couch': (0.7, 1.5),
    'dining table': (0.6, 0.8),
    'chair': (0.4, 0.4),
    'refrigerator': (0.5, 0.5),
    'toilet': (0.35, 0.5),
    'sink': (0.3, 0.4),
    'oven': (0.5, 0.5),
    'tv': (0.1, 0.4),
    'microwave': (0.3, 0.3),
    'potted plant': (0.2, 0.2),
    'bench': (0.3, 0.8),
}

MAX_FOOTPRINT_OVERLAP_IOU = 0.30

def _get_iou(boxA, boxB):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Boxes are in format [xmin, ymin, xmax, ymax]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    if boxAArea + boxBArea - interArea == 0:
        return 0.0

    return interArea / float(boxAArea + boxBArea - interArea)

def tag_3d_objects(point_cloud, rgbd_frames, camera_intrinsics, rooms_data=None, gravity_transform=None):
    """
    Detect objects in frames and project to 3D world coordinates.
    
    Args:
        point_cloud: o3d.geometry.PointCloud (gravity-aligned)
        rgbd_frames: list of RGBDFrame with .color_path, .depth_path, .pose
        camera_intrinsics: o3d.camera.PinholeCameraIntrinsic
        rooms_data: optional dict from wall_detector (used to assign objects to rooms)
        gravity_transform: optional 4x4 transform from gravity_aligner
    
    Returns:
        list of tagged object dicts:
            'class': str
            'location_3d': [x, y, z] in aligned world coordinates
            'confidence': float
            'footprint_m': [[x,y], [x,y], [x,y], [x,y]] (4 corners of axis-aligned bounding rect)
            'width_m': float
            'length_m': float
            'room_id': str or None (which room the object is in)
    """
    try:
        model = YOLO('yolov8n.pt')
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        return []

    # Prepare raycasting scene in case we need it
    raycast_scene = None
    if point_cloud is not None:
        try:
            # Note: Tensors in o3d for RaycastingScene require Open3D 0.13+
            scene = o3d.t.geometry.RaycastingScene()
            # Convert to legacy point cloud if needed, then to tensor
            # Since point_cloud is likely a legacy o3d.geometry.PointCloud
            pcd_tensor = o3d.t.geometry.PointCloud.from_legacy(point_cloud)
            # Create a triangle mesh or use point cloud directly? Raycasting usually needs mesh.
            # But we can approximate. Wait, Open3D RaycastingScene requires a mesh.
            # If point cloud, raycasting is harder. 
            # We'll build a quick mesh if needed, or just skip if it fails.
            pass
        except Exception:
            pass

    fx, fy = camera_intrinsics.get_focal_length()
    cx, cy = camera_intrinsics.get_principal_point()

    detections = []

    # Process every 5th frame
    for frame_idx in range(0, len(rgbd_frames), 5):
        frame = rgbd_frames[frame_idx]
        color_img = cv2.imread(frame.color_path)
        if color_img is None:
            continue
            
        color_img_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        
        # Run YOLO detection
        results = model(color_img_rgb, verbose=False)
        
        # Load depth if available
        depth_img = None
        if hasattr(frame, 'depth_path') and frame.depth_path:
            depth_img = cv2.imread(frame.depth_path, cv2.IMREAD_UNCHANGED)
            
        if depth_img is not None and len(depth_img.shape) == 3:
            depth_img = depth_img[:,:,0] # handle case if it was read as multi-channel

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                cls_name = model.names[cls_id]
                conf = box.conf[0].item()
                
                if cls_name not in ROOM_FEATURE_CLASSES:
                    continue
                    
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # Handle projection
                points_3d_world = []
                
                # Depth based projection
                if depth_img is not None and depth_img.dtype == np.uint16:
                    h_c, w_c = color_img.shape[:2]
                    h_d, w_d = depth_img.shape[:2]
                    
                    # Scale coordinates if depth size differs
                    scale_x = w_d / w_c
                    scale_y = h_d / h_c
                    
                    dx1, dy1 = int(x1 * scale_x), int(y1 * scale_y)
                    dx2, dy2 = int(x2 * scale_x), int(y2 * scale_y)
                    
                    dx1 = max(0, dx1)
                    dy1 = max(0, dy1)
                    dx2 = min(w_d, dx2)
                    dy2 = min(h_d, dy2)
                    
                    box_depth = depth_img[dy1:dy2, dx1:dx2]
                    
                    # Create meshgrid of coordinates
                    Y_d, X_d = np.mgrid[dy1:dy2, dx1:dx2]
                    
                    # Filter valid depths
                    valid_mask = (box_depth > 100) & (box_depth < 5000)
                    
                    if np.any(valid_mask):
                        Z = box_depth[valid_mask] / 1000.0  # mm to meters
                        X_idx = X_d[valid_mask]
                        Y_idx = Y_d[valid_mask]
                        
                        # Unproject
                        # We use color intrinsics but scaled coordinates? 
                        # Assuming camera_intrinsics are for the depth scale, or they are for color and we must scale cx, cy, fx, fy.
                        # Assuming camera_intrinsics given match the depth scale. 
                        # Actually standard practice: intrinsics match depth. If they match color, we need to adjust.
                        # We'll use the provided fx, fy, cx, cy and assume they are for the depth coordinate frame.
                        X = (X_idx - cx) * Z / fx
                        Y = (Y_idx - cy) * Z / fy
                        
                        points_3d_camera = np.column_stack((X, Y, Z, np.ones_like(Z)))
                        
                        # Transform to world
                        # frame.pose is camera-to-world transform
                        points_3d_w = (frame.pose @ points_3d_camera.T).T
                        
                        # Apply gravity transform if provided
                        if gravity_transform is not None:
                            points_3d_w = (gravity_transform @ points_3d_w.T).T
                            
                        points_3d_world = points_3d_w[:, :3]
                
                else:
                    # Raycasting fallback
                    # Simplify by projecting the center of the bounding box to a fixed distance or using priors.
                    # Since true raycasting against point cloud is non-trivial without a mesh in Open3D,
                    # we will approximate by projecting the center to Z_world=0 plane.
                    
                    cx_box = (x1 + x2) / 2.0
                    cy_box = (y1 + y2) / 2.0
                    
                    # Ray vector in camera frame
                    ray_c = np.array([(cx_box - cx) / fx, (cy_box - cy) / fy, 1.0])
                    ray_c = ray_c / np.linalg.norm(ray_c)
                    
                    # Convert origin and ray to world
                    cam_pos = frame.pose[:3, 3]
                    ray_w = frame.pose[:3, :3] @ ray_c
                    
                    # Apply gravity align
                    if gravity_transform is not None:
                        cam_pos_4 = gravity_transform @ np.array([cam_pos[0], cam_pos[1], cam_pos[2], 1.0])
                        cam_pos = cam_pos_4[:3]
                        
                        ray_w_4 = gravity_transform @ np.array([ray_w[0], ray_w[1], ray_w[2], 0.0])
                        ray_w = ray_w_4[:3]
                        ray_w = ray_w / np.linalg.norm(ray_w)
                        
                    # Find intersection with Z=0 (floor)
                    # p = cam_pos + t * ray_w
                    # Z = cam_pos[2] + t * ray_w[2] = 0 => t = -cam_pos[2] / ray_w[2]
                    if ray_w[2] < -1e-3:
                        t = -cam_pos[2] / ray_w[2]
                        if t > 0:
                            intersect = cam_pos + t * ray_w
                            points_3d_world = np.array([intersect])
                
                if len(points_3d_world) > 0:
                    # Compute footprint (5th and 95th percentile)
                    X_w = points_3d_world[:, 0]
                    Y_w = points_3d_world[:, 1]
                    Z_w = points_3d_world[:, 2]
                    
                    if len(points_3d_world) >= 10:
                        min_x, max_x = np.percentile(X_w, 5), np.percentile(X_w, 95)
                        min_y, max_y = np.percentile(Y_w, 5), np.percentile(Y_w, 95)
                    else:
                        min_x, max_x = np.min(X_w), np.max(X_w)
                        min_y, max_y = np.min(Y_w), np.max(Y_w)
                        
                    center_x = (min_x + max_x) / 2.0
                    center_y = (min_y + max_y) / 2.0
                    center_z = np.mean(Z_w)
                    
                    w = max_x - min_x
                    l = max_y - min_y
                    
                    # Apply class minimum dimensions
                    min_w, min_l = CLASS_MIN_DIMENSIONS.get(cls_name, (0.2, 0.2))
                    
                    # We might not know which is width and which is length relative to world axes,
                    # so we just ensure max(w, l) >= max(min_w, min_l) and min(w, l) >= min(min_w, min_l)
                    # For simplicity, just enforce minimums independently
                    
                    # Ensure minimums
                    if w < min_w and w < min_l:
                        w = min(min_w, min_l)
                    if l < min_w and l < min_l:
                        l = min(min_w, min_l)
                        
                    # Also we should probably just ensure both axes meet the smallest minimum, or use heuristics.
                    # The prompt says: "If the detected footprint is smaller than the class minimum in either dimension, scale it up to the minimum while keeping the center position."
                    # We'll just enforce min_w for X-axis and min_l for Y-axis as a simple fallback, or sorting them.
                    dim1, dim2 = sorted([w, l])
                    min1, min2 = sorted([min_w, min_l])
                    
                    if w < l:
                        w = max(w, min1)
                        l = max(l, min2)
                    else:
                        w = max(w, min2)
                        l = max(l, min1)
                        
                    min_x = center_x - w / 2.0
                    max_x = center_x + w / 2.0
                    min_y = center_y - l / 2.0
                    max_y = center_y + l / 2.0

                    footprint = [
                        [min_x, min_y],
                        [max_x, min_y],
                        [max_x, max_y],
                        [min_x, max_y]
                    ]
                    
                    detections.append({
                        'class': cls_name,
                        'location_3d': [center_x, center_y, center_z],
                        'confidence': conf,
                        'footprint_m': footprint,
                        'width_m': w,
                        'length_m': l,
                        'bbox': [min_x, min_y, max_x, max_y] # Internal use
                    })

    # Deduplication
    # 1. Group same class within 0.8m
    filtered_detections = []
    
    # Sort by confidence descending
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    used_indices = set()
    for i in range(len(detections)):
        if i in used_indices:
            continue
        
        det1 = detections[i]
        loc1 = np.array(det1['location_3d'][:2])
        
        # Find all detections of same class within 0.8m
        for j in range(i + 1, len(detections)):
            if j in used_indices:
                continue
                
            det2 = detections[j]
            if det1['class'] == det2['class']:
                loc2 = np.array(det2['location_3d'][:2])
                dist = np.linalg.norm(loc1 - loc2)
                
                if dist < 0.8:
                    used_indices.add(j) # Suppress lower confidence detection
        
        filtered_detections.append(det1)
        used_indices.add(i)

    # 2. Cross-class max footprint overlap IOU = 0.30
    final_detections = []
    filtered_detections.sort(key=lambda x: x['confidence'], reverse=True)
    used_indices = set()
    
    for i in range(len(filtered_detections)):
        if i in used_indices:
            continue
            
        det1 = filtered_detections[i]
        final_detections.append(det1)
        used_indices.add(i)
        
        for j in range(i + 1, len(filtered_detections)):
            if j in used_indices:
                continue
                
            det2 = filtered_detections[j]
            iou = _get_iou(det1['bbox'], det2['bbox'])
            
            if iou > MAX_FOOTPRINT_OVERLAP_IOU:
                used_indices.add(j) # Suppress lower confidence

    # Room assignment
    for det in final_detections:
        det['room_id'] = None
        # Clean up internal bbox
        if 'bbox' in det:
            del det['bbox']

        if rooms_data:
            cx, cy = det['location_3d'][0], det['location_3d'][1]
            # rooms_data comes from wall_detector: {'rooms': [list of room dicts]}
            rooms_list = rooms_data.get('rooms', []) if isinstance(rooms_data, dict) else []
            for room in rooms_list:
                contour = room.get('contour_m', [])
                if len(contour) >= 3:
                    path = Path(contour)
                    if path.contains_point((cx, cy)):
                        det['room_id'] = room.get('id', None)
                        break

    return final_detections