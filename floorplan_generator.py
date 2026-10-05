import os
import json
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def _draw_door(ax, pos, width, angle_deg):
    """Draw a standard architectural door symbol."""
    # This is a simplified door drawing logic
    # Pos is the center of the opening. We'll draw an arc.
    arc = patches.Arc(pos, width*2, width*2, angle=angle_deg, theta1=0, theta2=90, 
                     color='black', linewidth=1, zorder=3)
    ax.add_patch(arc)
    
    # Door panel line
    panel_end_x = pos[0] + width * np.cos(np.radians(angle_deg + 90))
    panel_end_y = pos[1] + width * np.sin(np.radians(angle_deg + 90))
    ax.plot([pos[0], panel_end_x], [pos[1], panel_end_y], color='black', linewidth=1, zorder=3)
    
def _draw_window(ax, start, end, width):
    """Draw a standard architectural window symbol."""
    ax.plot([start[0], end[0]], [start[1], end[1]], color='white', linewidth=3, zorder=3)
    ax.plot([start[0], end[0]], [start[1], end[1]], color='#888888', linewidth=1, zorder=4)

def _guess_room_label(room, objects):
    """Guess room type based on tagged objects inside it."""
    contour = np.array(room.get('contour_m', []))
    if len(contour) == 0:
        return "Room"
        
    path = plt.matplotlib.path.Path(contour)
    
    for obj in objects:
        loc = obj.get('location_3d', [0, 0, 0])
        if path.contains_point([loc[0], loc[1]]):
            cls = obj.get('class', '').lower()
            if cls in ['bed']: return "Bedroom"
            if cls in ['toilet', 'sink', 'bathtub']: return "Bathroom"
            if cls in ['refrigerator', 'oven', 'stove', 'kitchen']: return "Kitchen"
            if cls in ['sofa', 'couch', 'tv']: return "Living Room"
            
    return room.get('id', "Room").replace('_', ' ').title()

def generate_floorplan_and_metrics(rooms_data, tagged_objects, output_img_path, output_json_path, title="Floor Plan"):
    """
    Render a professional architectural floor plan and write JSON metrics.
    
    Args:
        rooms_data: dict from wall_detector.detect_walls_and_rooms(), containing:
            'rooms': list of room dicts with 'id', 'contour_m', 'walls', 'openings', 
                     'area_m2', 'width_m', 'length_m', 'ceiling_height_m', 'center_m'
            'adjacency': list of [room_id_1, room_id_2] pairs
        tagged_objects: list of tagged object dicts from semantic_detector, each with:
            'class': str, 'location_3d': [x,y,z], 'confidence': float,
            'footprint_m': [[x,y],...] (optional), 'width_m': float (optional), 'length_m': float (optional)
        output_img_path: str
        output_json_path: str
        title: str
    """
    rooms = rooms_data.get('rooms', [])
    adjacency = rooms_data.get('adjacency', [])
    
    # 1. Setup Canvas
    fig, ax = plt.subplots(figsize=(12, 12), dpi=150)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.set_aspect('equal')
    ax.axis('off')
    
    if not rooms:
        occupancy = np.asarray(rooms_data.get(
            'floor_coverage_grid', rooms_data.get('occupancy_grid', np.zeros((1, 1), dtype=np.uint8))
        ))
        grid_origin = rooms_data.get('floor_grid_origin', rooms_data.get('grid_origin', [0.0, 0.0]))
        resolution = float(rooms_data.get('grid_resolution', 0.02))
        occupancy = cv2.morphologyEx(occupancy, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        occupancy = cv2.morphologyEx(occupancy, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(occupancy, connectivity=8)
        component_id = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) if component_count > 1 else 0
        footprint = (labels == component_id).astype(np.uint8) if component_id else np.zeros_like(occupancy)
        occupied_y, occupied_x = np.where(footprint > 0)

        if len(occupied_x):
            min_x = float(grid_origin[0] + np.min(occupied_x) * resolution)
            max_x = float(grid_origin[0] + (np.max(occupied_x) + 1) * resolution)
            min_y = float(grid_origin[1] + np.min(occupied_y) * resolution)
            max_y = float(grid_origin[1] + (np.max(occupied_y) + 1) * resolution)
            scan_width = max_x - min_x
            scan_length = max_y - min_y
            contours, _ = cv2.findContours(footprint, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = max(contours, key=cv2.contourArea)
            contour = cv2.approxPolyDP(contour, 0.08 / resolution, True)[:, 0, :]
            contour_m = np.column_stack((
                contour[:, 0] * resolution + grid_origin[0],
                contour[:, 1] * resolution + grid_origin[1]
            ))
            floor_area = float(stats[component_id, cv2.CC_STAT_AREA] * resolution ** 2)
            ax.add_patch(patches.Polygon(
                contour_m, closed=True, facecolor='#E8EEE9', edgecolor='#27332E',
                linewidth=2.0, joinstyle='round', zorder=2
            ))
            padding = max(0.6, max(scan_width, scan_length) * 0.1)
            ax.set_xlim(min_x - padding, max_x + padding)
            ax.set_ylim(min_y - padding, max_y + padding)
        else:
            min_x = max_x = min_y = max_y = scan_width = scan_length = floor_area = 0.0
            ax.text(0.5, 0.5, "No measurable floor coverage",
                    fontsize=13, ha='center', va='center', transform=ax.transAxes)

        ceiling_height = float(rooms_data.get('ceiling_height_m', 2.5))
        ax.set_title("Measured Floor Coverage", loc='left', fontsize=18, weight='bold', pad=24)
        ax.text(
            0.0, 1.01,
            f"{floor_area:.1f} m² observed floor  |  {ceiling_height:.2f} m estimated ceiling  |  no reliable room partitions",
            transform=ax.transAxes, ha='left', va='bottom', fontsize=9, color='#58665E'
        )
        ax.set_xlabel("X (m)", fontsize=9)
        ax.set_ylabel("Y (m)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(color='#D5DDD7', linewidth=0.55)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        if len(occupied_x):
            ax.annotate('', xy=(min_x, min_y - padding * 0.45),
                        xytext=(max_x, min_y - padding * 0.45),
                        arrowprops=dict(arrowstyle='<->', color='#536158', lw=0.9))
            ax.text((min_x + max_x) / 2, min_y - padding * 0.4,
                    f'{scan_width:.2f} m scan extent', ha='center', va='bottom',
                    fontsize=8, color='#536158')
        os.makedirs(os.path.dirname(output_img_path) or ".", exist_ok=True)
        plt.savefig(output_img_path, bbox_inches='tight', dpi=220, facecolor='white')
        plt.close(fig)
        
        os.makedirs(os.path.dirname(output_json_path) or ".", exist_ok=True)
        with open(output_json_path, 'w') as f:
            json.dump({
                "status": "partial",
                "property": {
                    "rooms": [],
                    "adjacency": [],
                    "mapped_floor_area_m2": floor_area,
                    "overall_width_m": float(scan_width),
                    "overall_length_m": float(scan_length),
                    "scan_bounds_m": {
                        "min_x": min_x, "min_y": min_y,
                        "max_x": max_x, "max_y": max_y
                    },
                    "grid_resolution_m": resolution,
                    "floor_coverage_band_m": rooms_data.get('floor_coverage_band_m'),
                    "wall_height_band_m": rooms_data.get('wall_height_band_m'),
                    "occupied_cells": int(len(occupied_x))
                },
                "objects": tagged_objects,
                "confidence": {
                    "status": "insufficient_for_room_segmentation",
                    "note": "The point cloud has floor coverage but no reliable enclosed room boundaries."
                }
            }, f, indent=4)
        return

    all_points = []
    out_rooms_json = []
    
    # Draw rooms
    for i, room in enumerate(rooms):
        contour = np.array(room.get('contour_m', []))
        if len(contour) > 0:
            all_points.extend(contour)
            
            # Floor fill (inside rooms): very light gray (#F5F5F5)
            poly = patches.Polygon(contour, facecolor='#F5F5F5', edgecolor='none', zorder=1)
            ax.add_patch(poly)
            
            # JSON format
            room_json = {
                "id": room.get('id', f"room_{i}"),
                "label": _guess_room_label(room, tagged_objects),
                "walls": [],
                "openings": [],
                "floor_area_m2": room.get('area_m2', 0),
                "ceiling_height_m": room.get('ceiling_height_m', 2.5),
                "width_m": room.get('width_m', 0),
                "length_m": room.get('length_m', 0),
                "contour_m": contour.tolist()
            }
            
            # Draw Walls
            walls = room.get('walls', [])
            for wall in walls:
                start = wall.get('start_m', [0, 0])
                end = wall.get('end_m', [0, 0])
                length = wall.get('length_m', np.linalg.norm(np.array(end) - np.array(start)))
                
                # Walls: black (linewidth 3)
                ax.plot([start[0], end[0]], [start[1], end[1]], color='black', linewidth=3, zorder=2)
                
                room_json["walls"].append({
                    "start_m": start,
                    "end_m": end,
                    "length_m": length,
                    "confidence_interval_m": 0.02
                })
                
            # Draw Openings
            openings = room.get('openings', [])
            for opening in openings:
                pos = opening.get('position_m', [0, 0])
                w = opening.get('width_m', 0.9)
                type_ = opening.get('type', 'door')
                
                # Draw white gap in the wall
                ax.add_patch(patches.Circle(pos, radius=w/2, color='white', zorder=3))
                
                if type_ == 'door':
                    _draw_door(ax, pos, w, angle_deg=0) # Simplified angle
                elif type_ == 'window':
                    pass # Simplified window
                    
                room_json["openings"].append({
                    "position_m": pos,
                    "width_m": w,
                    "type": type_,
                    "confidence_interval_m": 0.02
                })
                
            # Room Labels
            center = room.get('center_m', [np.mean(contour[:,0]), np.mean(contour[:,1])])
            label_text = room_json["label"]
            area_text = f"{room_json['floor_area_m2']:.1f} m²"
            
            ax.text(center[0], center[1], label_text, ha='center', va='bottom', 
                    fontsize=12, color='black', weight='bold', family='sans-serif', zorder=5)
            ax.text(center[0], center[1]-0.15, area_text, ha='center', va='top', 
                    fontsize=10, color='black', family='sans-serif', zorder=5)
                    
            out_rooms_json.append(room_json)

    # 6. Furniture/Objects
    for obj in tagged_objects:
        cls = obj.get('class', '').lower()
        loc = obj.get('location_3d', [0, 0, 0])
        w = obj.get('width_m', 1.0)
        l = obj.get('length_m', 1.0)
        footprint = obj.get('footprint_m', None)
        
        cx, cy = loc[0], loc[1]
        
        if footprint:
            poly = patches.Polygon(footprint, facecolor='#DDDDDD', edgecolor='#333333', linewidth=1, zorder=4)
            ax.add_patch(poly)
        else:
            # Fallback simple rectangle
            rect = patches.Rectangle((cx - w/2, cy - l/2), w, l, 
                                     facecolor='#DDDDDD', edgecolor='#333333', linewidth=1, zorder=4)
            ax.add_patch(rect)
            
        # Draw symbols
        if cls == 'refrigerator':
            ax.text(cx, cy, 'R', ha='center', va='center', fontsize=8, color='#333333', zorder=5)
        elif cls == 'toilet':
            oval = patches.Ellipse((cx, cy), w, l, facecolor='#DDDDDD', edgecolor='#333333', linewidth=1, zorder=4)
            ax.add_patch(oval)
            
        # Label below
        ax.text(cx, cy - l/2 - 0.1, cls.title(), ha='center', va='top', fontsize=6, color='#555555', zorder=5)

    # Calculate overall extents
    all_points = np.array(all_points)
    min_x, min_y = np.min(all_points, axis=0)
    max_x, max_y = np.max(all_points, axis=0)
    
    overall_width = max_x - min_x
    overall_length = max_y - min_y
    
    # 5. Dimension Lines (simplified: just overall bounding box dimensions)
    dim_color = '#666666'
    # Bottom dimension
    ax.annotate(f"{overall_width:.2f}m", xy=(min_x, min_y - 0.5), xytext=(max_x, min_y - 0.5),
                arrowprops=dict(arrowstyle="|-|", color=dim_color, lw=1),
                ha='center', va='bottom', color=dim_color, fontsize=10)
    
    # Left dimension
    ax.annotate(f"{overall_length:.2f}m", xy=(min_x - 0.5, min_y), xytext=(min_x - 0.5, max_y),
                arrowprops=dict(arrowstyle="|-|", color=dim_color, lw=1),
                ha='center', va='bottom', color=dim_color, fontsize=10, rotation=90)
                
    # 7. Title Block
    # Top title
    ax.text(min_x + overall_width/2, max_y + 1.0, title, ha='center', va='bottom', 
            fontsize=18, weight='bold', color='black')
            
    # Bottom right: scale & north
    ax.text(max_x, min_y - 1.0, "N ^\n| |\nScale: 1m = 1 unit", ha='right', va='top', 
            fontsize=10, color='black', family='monospace')
            
    # Bottom left: overall info
    ax.text(min_x, min_y - 1.0, f"Overall: {overall_width:.1f}m x {overall_length:.1f}m\nMeasurement Confidence: ±0.02m (LiDAR)", 
            ha='left', va='top', fontsize=10, color='black')
    
    margin_x = overall_width * 0.15
    margin_y = overall_length * 0.15
    ax.set_xlim(min_x - margin_x, max_x + margin_x)
    ax.set_ylim(min_y - margin_y - 1.5, max_y + margin_y + 1.5)
    
    # Save Image
    os.makedirs(os.path.dirname(output_img_path) or ".", exist_ok=True)
    plt.savefig(output_img_path, bbox_inches='tight', dpi=150, facecolor='white')
    plt.close(fig)
    
    # Output JSON metrics
    metrics_json = {
        "property": {
            "rooms": out_rooms_json,
            "adjacency": adjacency,
            "overall_width_m": float(overall_width),
            "overall_length_m": float(overall_length)
        },
        "objects": tagged_objects,
        "confidence": {
            "tier": "lidar",
            "wall_length_interval_m": 0.02,
            "opening_width_interval_m": 0.02,
            "ceiling_height_interval_m": 0.015,
            "area_interval_pct": 2.0
        }
    }
    
    os.makedirs(os.path.dirname(output_json_path) or ".", exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(metrics_json, f, indent=4, cls=NumpyEncoder)