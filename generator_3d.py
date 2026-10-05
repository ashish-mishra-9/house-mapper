import open3d as o3d
import numpy as np
import os

def load_camera_intrinsics(width, height, fx, fy, cx, cy):
    """
    Load pinhole camera intrinsics.
    """
    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

def build_point_cloud(rgbd_frames, camera_intrinsics, output_path="output/raw_cloud.ply"):
    """
    Fuse RGB-D frames into a single cleaned point cloud.
    
    Args:
        rgbd_frames: List of RGBDFrame objects with .color_path, .depth_path, .pose (4x4 camera-to-world)
        camera_intrinsics: o3d.camera.PinholeCameraIntrinsic
        output_path: Where to save the PLY file
    
    Returns:
        o3d.geometry.PointCloud - fused, downsampled, outlier-removed point cloud
    """
    pcd_combined = o3d.geometry.PointCloud()
    
    for i, frame in enumerate(rgbd_frames):
        color = o3d.io.read_image(frame.color_path)
        depth = o3d.io.read_image(frame.depth_path)
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_scale=1000.0, depth_trunc=5.0, convert_rgb_to_intensity=False
        )
        
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsics)
        pcd.transform(frame.pose)
        
        pcd_combined += pcd
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1} / {len(rgbd_frames)} frames for point cloud.")
            
    print("Voxel downsampling point cloud...")
    pcd_combined = pcd_combined.voxel_down_sample(voxel_size=0.02)
    
    print("Removing statistical outliers...")
    pcd_combined, ind = pcd_combined.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    o3d.io.write_point_cloud(output_path, pcd_combined)
    print(f"Saved point cloud to {output_path}")
    
    return pcd_combined


def build_mesh_tsdf(rgbd_frames, camera_intrinsics, output_path="output/raw_mesh.ply"):
    """
    Also build a triangle mesh via TSDF for visualization.
    Uses relaxed params: voxel_length=0.04, sdf_trunc=0.12, depth_trunc=5.0
    Returns o3d.geometry.TriangleMesh
    """
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.04,
        sdf_trunc=0.12,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    
    for i, frame in enumerate(rgbd_frames):
        color = o3d.io.read_image(frame.color_path)
        depth = o3d.io.read_image(frame.depth_path)
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_scale=1000.0, depth_trunc=5.0, convert_rgb_to_intensity=False
        )
        
        extrinsic = np.linalg.inv(frame.pose)
        volume.integrate(rgbd, camera_intrinsics, extrinsic)
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1} / {len(rgbd_frames)} frames for TSDF volume.")
            
    print("Extracting triangle mesh from TSDF volume...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"Saved TSDF mesh to {output_path}")
    
    return mesh