import numpy as np
import open3d as o3d
import cv2
import math

def detect_walls_and_rooms(point_cloud, floor_z, ceiling_z):
    """
    Detect walls, openings, and room boundaries from an aligned point cloud.
    
    The point cloud is already gravity-aligned: Z is up, floor at Z≈floor_z.
    
    Args:
        point_cloud: o3d.geometry.PointCloud (gravity-aligned)
        floor_z: float
        ceiling_z: float
    
    Returns:
        dict with:
            'rooms': list of room dicts
            'adjacency': list of [room_id_1, room_id_2] pairs
            'occupancy_grid': np.ndarray (uint8)
            'grid_origin': [x_min, y_min]
            'grid_resolution': float
    """
    pts = np.asarray(point_cloud.points)
    if pts.size == 0:
        return {'rooms': [], 'adjacency': [], 'occupancy_grid': np.zeros((10,10), dtype=np.uint8), 'grid_origin': [0,0], 'grid_resolution': 0.02}

    height = ceiling_z - floor_z

    # Keep a narrow band clear of floor clutter and ceiling returns.
    z_min_slice = floor_z + 1.0
    z_max_slice = min(floor_z + 1.5, ceiling_z - 0.5)
    wall_mask = (pts[:, 2] >= z_min_slice) & (pts[:, 2] <= z_max_slice)
    wall_pts = pts[wall_mask]
    
    # 2. Create 2D occupancy grid
    resolution = 0.02 # 0.02m per pixel
    
    if wall_pts.shape[0] > 0:
        wall_cloud = o3d.geometry.PointCloud()
        wall_cloud.points = o3d.utility.Vector3dVector(wall_pts)
        wall_cloud = wall_cloud.voxel_down_sample(0.04)
        wall_cloud.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.09, max_nn=30)
        )
        wall_normals = np.asarray(wall_cloud.normals)
        wall_pts = np.asarray(wall_cloud.points)[np.abs(wall_normals[:, 2]) < 0.35]
    
    x_min = np.min(pts[:, 0]) - 1.0 # padding
    x_max = np.max(pts[:, 0]) + 1.0
    y_min = np.min(pts[:, 1]) - 1.0
    y_max = np.max(pts[:, 1]) + 1.0
    
    width_px = int(np.ceil((x_max - x_min) / resolution))
    height_px = int(np.ceil((y_max - y_min) / resolution))
    
    grid = np.zeros((height_px, width_px), dtype=np.int32)
    
    wall_x_px = ((wall_pts[:, 0] - x_min) / resolution).astype(int)
    wall_y_px = ((wall_pts[:, 1] - y_min) / resolution).astype(int)
    
    valid_mask = (wall_x_px >= 0) & (wall_x_px < width_px) & (wall_y_px >= 0) & (wall_y_px < height_px)
    wall_x_px = wall_x_px[valid_mask]
    wall_y_px = wall_y_px[valid_mask]
    
    np.add.at(grid, (wall_y_px, wall_x_px), 1)
    
    binary_grid = np.zeros_like(grid, dtype=np.uint8)
    occupied_counts = grid[grid > 0]
    density_threshold = max(1, int(np.ceil(np.percentile(occupied_counts, 75)))) if occupied_counts.size else 1
    binary_grid[grid >= density_threshold] = 255

    # Bridge only one-cell gaps without widening or eroding thin wall traces.
    kernel3 = np.ones((3, 3), np.uint8)
    clean_grid = cv2.morphologyEx(binary_grid, cv2.MORPH_CLOSE, kernel3, iterations=1)

    floor_pts = pts[(pts[:, 2] >= floor_z - 0.08) & (pts[:, 2] <= floor_z + 0.08)]
    floor_grid = np.zeros_like(binary_grid)
    floor_x_px = ((floor_pts[:, 0] - x_min) / resolution).astype(int)
    floor_y_px = ((floor_pts[:, 1] - y_min) / resolution).astype(int)
    floor_valid = (floor_x_px >= 0) & (floor_x_px < width_px) & (floor_y_px >= 0) & (floor_y_px < height_px)
    floor_grid[floor_y_px[floor_valid], floor_x_px[floor_valid]] = 255
    
    # 4. Room detection (flood fill approach)
    # Invert grid: 0=wall, 255=free space
    free_space = cv2.bitwise_not(clean_grid)
    
    # Flood fill exterior from border
    h, w = free_space.shape
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(free_space, mask, (0,0), 0)
    cv2.floodFill(free_space, mask, (w-1,0), 0)
    cv2.floodFill(free_space, mask, (0,h-1), 0)
    cv2.floodFill(free_space, mask, (w-1,h-1), 0)
    
    # free_space now contains rooms as 255, walls and exterior as 0
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(free_space, connectivity=8)
    
    rooms_data = []
    
    # To snap angles:
    def snap_angle(vector):
        angle = np.arctan2(vector[1], vector[0])
        # Multiples of 45 deg (pi/4)
        target_angles = np.arange(-np.pi, np.pi + np.pi/4, np.pi/4)
        diff = np.abs(angle - target_angles)
        min_idx = np.argmin(diff)
        if diff[min_idx] < np.deg2rad(5):
            return target_angles[min_idx]
        return angle

    for i in range(1, num_labels):
        area_px = stats[i, cv2.CC_STAT_AREA]
        area_m2 = area_px * (resolution ** 2)
        if area_m2 <= 1.0:
            continue
            
        # 5. Room contour extraction
        room_mask = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(room_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
            
        c = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        
        if len(approx) < 3:
            continue
            
        contour_m = []
        for pt in approx:
            px, py = pt[0]
            cx = px * resolution + x_min
            cy = py * resolution + y_min
            contour_m.append([cx, cy])
            
        # 6. Wall segment extraction
        walls = []
        num_pts = len(contour_m)
        for j in range(num_pts):
            p1 = np.array(contour_m[j])
            p2 = np.array(contour_m[(j+1) % num_pts])
            vec = p2 - p1
            length = np.linalg.norm(vec)
            if length < 0.01:
                continue
                
            dir_vec = vec / length
            
            # Snap to dominant angles
            snapped_ang = snap_angle(dir_vec)
            snapped_dir = np.array([np.cos(snapped_ang), np.sin(snapped_ang)])
            p2_snapped = p1 + snapped_dir * length
            
            # Normal (inward pointing)
            # Assuming contour is clockwise or counter-clockwise.
            # OpenCV contours for external are usually counter-clockwise (in image coords, so clockwise in standard coords if Y is down).
            # We compute normal by rotating dir_vec by 90 degrees.
            normal = np.array([-snapped_dir[1], snapped_dir[0]])
            # Check if normal points inward by testing center
            center = np.mean(contour_m, axis=0)
            if np.dot(normal, center - p1) < 0:
                normal = -normal
                
            walls.append({
                'start_m': p1.tolist(),
                'end_m': p2_snapped.tolist(),
                'length_m': float(length),
                'normal': normal.tolist()
            })
            
            # Update contour to snapped (simple approach)
            contour_m[(j+1) % num_pts] = p2_snapped.tolist()
            
        # 7. Opening detection
        openings = []
        for w_idx, w in enumerate(walls):
            p1 = np.array(w['start_m'])
            p2 = np.array(w['end_m'])
            length = w['length_m']
            if length < 0.5:
                continue
                
            dir_vec = (p2 - p1) / length
            
            # Filter pts within 0.15m of wall
            vec_to_pts = pts[:, :2] - p1
            proj_len = np.dot(vec_to_pts, dir_vec)
            perp_dist = np.linalg.norm(vec_to_pts - proj_len[:, None] * dir_vec, axis=1)
            
            wall_region_mask = (perp_dist <= 0.15) & (proj_len >= 0) & (proj_len <= length)
            wall_region_pts = pts[wall_region_mask]
            
            if len(wall_region_pts) == 0:
                continue
                
            # Density at wall height
            w_h_mask = (wall_region_pts[:, 2] >= z_min_slice) & (wall_region_pts[:, 2] <= z_max_slice)
            w_h_pts = wall_region_pts[w_h_mask]
            
            if len(w_h_pts) == 0:
                continue
                
            # Bins of 0.1m
            num_bins = int(np.ceil(length / 0.1))
            if num_bins == 0: continue
            
            bin_proj = np.dot(w_h_pts[:, :2] - p1, dir_vec)
            hist, bin_edges = np.histogram(bin_proj, bins=num_bins, range=(0, length))
            
            # Find gaps (bins with < 2 points) spanning > 0.5m (5 bins)
            gap_mask = hist < 2
            gap_starts = []
            gap_ends = []
            in_gap = False
            start_idx = 0
            for k, is_gap in enumerate(gap_mask):
                if is_gap and not in_gap:
                    in_gap = True
                    start_idx = k
                elif not is_gap and in_gap:
                    in_gap = False
                    if (k - start_idx) * 0.1 >= 0.5:
                        gap_starts.append(start_idx)
                        gap_ends.append(k)
            if in_gap and (num_bins - start_idx) * 0.1 >= 0.5:
                gap_starts.append(start_idx)
                gap_ends.append(num_bins)
                
            for gs, ge in zip(gap_starts, gap_ends):
                gap_center = (gs + ge) / 2 * 0.1
                gap_width = (ge - gs) * 0.1
                pos_m = p1 + dir_vec * gap_center
                
                # Check door vs window
                # low density at z between 0 and 0.5*height (using floor_z as 0)
                z_door_min = floor_z
                z_door_max = floor_z + 0.5 * height
                
                door_mask = (wall_region_pts[:, 2] >= z_door_min) & (wall_region_pts[:, 2] <= z_door_max)
                door_pts = wall_region_pts[door_mask]
                
                if len(door_pts) > 0:
                    door_proj = np.dot(door_pts[:, :2] - p1, dir_vec)
                    door_hist, _ = np.histogram(door_proj, bins=num_bins, range=(0, length))
                    gap_door_density = np.sum(door_hist[gs:ge])
                    # If few points in bottom region, it's a door
                    op_type = 'door' if gap_door_density < (ge - gs) * 2 else 'window'
                else:
                    op_type = 'door'
                    
                openings.append({
                    'position_m': pos_m.tolist(),
                    'width_m': float(gap_width),
                    'wall_index': w_idx,
                    'type': op_type
                })
        
        # 9. Per-room ceiling height
        # Use bounding box of the room contour for fast filtering
        room_px_y, room_px_x = np.where(labels == i)
        room_x_min = float(np.min(room_px_x) * resolution + x_min)
        room_x_max = float(np.max(room_px_x) * resolution + x_min)
        room_y_min = float(np.min(room_px_y) * resolution + y_min)
        room_y_max = float(np.max(room_px_y) * resolution + y_min)

        ceil_h = ceiling_z - floor_z

        rooms_data.append({
            'id': f'room_{len(rooms_data)}',
            'contour_m': contour_m,
            'walls': walls,
            'openings': openings,
            'area_m2': float(area_m2),
            'width_m': float(room_x_max - room_x_min),
            'length_m': float(room_y_max - room_y_min),
            'ceiling_height_m': float(ceil_h),
            'center_m': [float((room_x_min + room_x_max) / 2), float((room_y_min + room_y_max) / 2)]
        })

    # 8. Room adjacency
    adjacency = []
    num_rooms = len(rooms_data)
    for i in range(num_rooms):
        for j in range(i+1, num_rooms):
            # simple check: bounding box intersection expanded by 0.3m
            r1 = rooms_data[i]
            r2 = rooms_data[j]
            
            dx = abs(r1['center_m'][0] - r2['center_m'][0])
            dy = abs(r1['center_m'][1] - r2['center_m'][1])
            
            if dx < (r1['width_m'] + r2['width_m'])/2 + 0.3 and \
               dy < (r1['length_m'] + r2['length_m'])/2 + 0.3:
                
                # Check walls
                adjacent = False
                for w1 in r1['walls']:
                    if adjacent: break
                    for w2 in r2['walls']:
                        # If normals are roughly opposite
                        n1 = np.array(w1['normal'])
                        n2 = np.array(w2['normal'])
                        if np.dot(n1, n2) < -0.8:
                            # check distance between wall lines
                            p1 = np.array(w1['start_m'])
                            v1 = np.array(w1['end_m']) - p1
                            len1 = np.linalg.norm(v1)
                            if len1 == 0: continue
                            dir1 = v1 / len1
                            
                            p2 = np.array(w2['start_m'])
                            vec_to_p2 = p2 - p1
                            perp_dist = np.linalg.norm(vec_to_p2 - np.dot(vec_to_p2, dir1) * dir1)
                            if perp_dist < 0.3:
                                adjacent = True
                                break
                if adjacent:
                    adjacency.append([r1['id'], r2['id']])

    return {
        'rooms': rooms_data,
        'adjacency': adjacency,
        'occupancy_grid': binary_grid,
        'floor_coverage_grid': floor_grid,
        'grid_origin': [float(x_min), float(y_min)],
        'floor_grid_origin': [float(x_min), float(y_min)],
        'grid_resolution': resolution,
        'ceiling_height_m': float(height),
        'wall_height_band_m': [float(z_min_slice), float(z_max_slice)],
        'floor_coverage_band_m': [float(floor_z - 0.08), float(floor_z + 0.08)],
        'occupancy_threshold_points': density_threshold
    }
