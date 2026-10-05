"""
House Mapper Pipeline - Main Entry Point
=========================================
End-to-end floor plan generation from iPhone captures.

Usage:
    python main.py --input <path> --tier <photos|video|lidar> [--output <dir>]

Tiers:
    photos  - 2-8 still photos per room (any iPhone 15+)
    video   - Handheld walkthrough video (any iPhone 15+)
    lidar   - LiDAR depth + poses + RGB (iPhone Pro/Pro Max)
"""

import argparse
import os
import sys
import time
import numpy as np

from input_adapters import (
    convert_photos_to_common,
    convert_video_to_common,
    convert_lidar_to_common,
    get_default_intrinsics,
)
from generator_3d import build_point_cloud, build_mesh_tsdf, load_camera_intrinsics
from gravity_aligner import align_to_gravity
from wall_detector import detect_walls_and_rooms
from semantic_detector import tag_3d_objects
from floorplan_generator import generate_floorplan_and_metrics


def run_pipeline(input_path, tier, output_dir="output/"):
    """
    Run the full house-mapping pipeline from raw capture to floor plan.

    Args:
        input_path: Path to the input data (folder for photos/lidar, file for video)
        tier: One of 'photos', 'video', 'lidar'
        output_dir: Directory for all output artifacts
    """
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"  HOUSE MAPPER PIPELINE")
    print(f"  Tier: {tier.upper()}")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_dir}")
    print("=" * 60 + "\n")

    # ── Stage 1: Convert input to common RGB-D frame format ──
    print("\n▶ STAGE 1: Input Conversion")
    if tier == "photos":
        frames = convert_photos_to_common(input_path, os.path.join(output_dir, "temp_photos"))
    elif tier == "video":
        frames = convert_video_to_common(input_path, os.path.join(output_dir, "temp_video"))
    elif tier == "lidar":
        frames = convert_lidar_to_common(input_path)
    else:
        raise ValueError(f"Invalid tier: {tier}. Must be 'photos', 'video', or 'lidar'.")

    print(f"  ✓ Got {len(frames)} RGB-D frames")

    # ── Stage 2: Determine camera intrinsics ──
    print("\n▶ STAGE 2: Camera Intrinsics")
    intr_dict = get_default_intrinsics(frames)
    intrinsics = load_camera_intrinsics(
        intr_dict['width'], intr_dict['height'],
        intr_dict['fx'], intr_dict['fy'],
        intr_dict['cx'], intr_dict['cy']
    )
    print(f"  ✓ Image: {intr_dict['width']}x{intr_dict['height']}")
    print(f"  ✓ Focal: fx={intr_dict['fx']:.1f}, fy={intr_dict['fy']:.1f}")
    print(f"  ✓ Principal: cx={intr_dict['cx']:.1f}, cy={intr_dict['cy']:.1f}")

    # ── Stage 3: Build 3D point cloud ──
    print("\n▶ STAGE 3: 3D Reconstruction")
    point_cloud = build_point_cloud(
        frames, intrinsics,
        output_path=os.path.join(output_dir, "raw_cloud.ply")
    )
    num_points = len(np.asarray(point_cloud.points))
    print(f"  ✓ Point cloud: {num_points} points")

    if num_points < 100:
        print("  ⚠ WARNING: Very few points reconstructed. Results may be poor.")

    # ── Stage 4: Gravity alignment ──
    print("\n▶ STAGE 4: Gravity Alignment")
    alignment = align_to_gravity(point_cloud)
    aligned_cloud = alignment['point_cloud']
    floor_z = alignment['floor_z']
    ceiling_z = alignment['ceiling_z']
    height = alignment['height']
    gravity_transform = alignment['transform']
    print(f"  ✓ Floor at Z={floor_z:.3f}m")
    print(f"  ✓ Ceiling at Z={ceiling_z:.3f}m")
    print(f"  ✓ Height: {height:.3f}m")

    # ── Stage 5: Wall detection and room segmentation ──
    print("\n▶ STAGE 5: Wall Detection & Room Segmentation")
    rooms_data = detect_walls_and_rooms(aligned_cloud, floor_z, ceiling_z)
    rooms = rooms_data.get('rooms', [])
    adjacency = rooms_data.get('adjacency', [])
    print(f"  ✓ Detected {len(rooms)} room(s)")
    for room in rooms:
        n_walls = len(room.get('walls', []))
        n_openings = len(room.get('openings', []))
        print(f"    - {room['id']}: {room['width_m']:.2f}m × {room['length_m']:.2f}m, "
              f"area={room['area_m2']:.1f}m², {n_walls} walls, {n_openings} openings, "
              f"ceiling={room['ceiling_height_m']:.2f}m")
    if adjacency:
        print(f"  ✓ Room adjacency: {adjacency}")

    # Save occupancy grid for debugging
    occ_grid = rooms_data.get('occupancy_grid')
    if occ_grid is not None:
        import cv2
        cv2.imwrite(os.path.join(output_dir, "occupancy_grid.png"), occ_grid)
        print(f"  ✓ Saved occupancy grid for debug")
    floor_grid = rooms_data.get('floor_coverage_grid')
    if floor_grid is not None:
        cv2.imwrite(os.path.join(output_dir, "floor_coverage.png"), floor_grid)
        print(f"  ✓ Saved floor coverage grid")

    # ── Stage 6: Semantic object detection ──
    print("\n▶ STAGE 6: Object Detection")
    tagged_objects = tag_3d_objects(
        aligned_cloud, frames, intrinsics,
        rooms_data=rooms_data,
        gravity_transform=gravity_transform
    )
    print(f"  ✓ Tagged {len(tagged_objects)} objects")
    for obj in tagged_objects:
        print(f"    - {obj['class']} (conf={obj['confidence']:.2f}): "
              f"{obj.get('width_m', 0):.2f}m × {obj.get('length_m', 0):.2f}m")

    # ── Stage 7: Generate floor plan and metrics ──
    print("\n▶ STAGE 7: Floor Plan Rendering")
    plan_path = os.path.join(output_dir, "plan.png")
    json_path = os.path.join(output_dir, "output.json")

    generate_floorplan_and_metrics(
        rooms_data=rooms_data,
        tagged_objects=tagged_objects,
        output_img_path=plan_path,
        output_json_path=json_path,
        title="Floor Plan"
    )
    print(f"  ✓ Saved floor plan: {plan_path}")
    print(f"  ✓ Saved metrics JSON: {json_path}")

    # ── Summary ──
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"  PIPELINE COMPLETE in {elapsed:.1f}s")
    print(f"  Rooms: {len(rooms)}")
    print(f"  Objects: {len(tagged_objects)}")
    print(f"  Output: {output_dir}")
    print("=" * 60 + "\n")

    return {
        'rooms': rooms,
        'objects': tagged_objects,
        'plan_path': plan_path,
        'json_path': json_path,
        'elapsed_seconds': elapsed,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="House Mapper - Generate architectural floor plans from iPhone captures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --input raw_data/sample_lidar --tier lidar
    python main.py --input walkthrough.mp4 --tier video
    python main.py --input room_photos/ --tier photos --output results/
        """
    )
    parser.add_argument("--input", required=True, help="Path to input (folder or video file)")
    parser.add_argument("--tier", required=True, choices=["photos", "video", "lidar"],
                        help="Input tier: photos, video, or lidar")
    parser.add_argument("--output", default="output/", help="Output directory (default: output/)")

    args = parser.parse_args()
    run_pipeline(args.input, args.tier, args.output)