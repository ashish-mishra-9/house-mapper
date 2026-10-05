# House Mapper

House Mapper is an experimental Python pipeline for turning phone-based RGB-D/LiDAR captures into a 3D point cloud, estimated floor coverage, room geometry, and detected furniture objects.

> **Current status:** This repository is a prototype, not a production or submission-ready surveying tool. The included LiDAR sample currently produces a measured floor-coverage outline, but no reliable room polygons. Crack detection is not implemented.

## Current Scope

The command-line pipeline in `main.py` supports three input tiers:

| Tier | Input | Current status |
| --- | --- | --- |
| `lidar` | RGB video, 16-bit depth images, and per-frame odometry | Primary path; an included sample export is available under `raw_data/sample_lidar/`. |
| `video` | Walkthrough video | Experimental; depends on estimated camera poses and estimated depth. |
| `photos` | A directory of room photographs | Experimental; depends on estimated camera poses and estimated depth. |

The pipeline converts inputs into RGB-D frames, fuses depth into a point cloud, estimates gravity alignment and ceiling height, builds a wall-height occupancy grid, attempts room segmentation, detects selected furniture/fixture classes, and writes a plan image and JSON metrics.

The current object detector recognizes common room objects such as beds, chairs, couches, sinks, toilets, and refrigerators using `yolov8n.pt`. It is not a building-defect detector.

### Important Limitations

- **Room boundaries are not currently reliable.** Room segmentation requires sufficiently continuous wall evidence to enclose regions. In the included sample, the detector returns zero reliable rooms.
- When room segmentation fails, `plan.png` shows the observed floor-return coverage. This is a coverage visualization, not a floor plan, and its outline must not be interpreted as a surveyed room boundary.
- The sample's floor-return coverage is approximately 16.6 m². This is an estimate of observed floor coverage, not a verified room area or property measurement.
- Open doorways, occlusions, furniture, sparse wall returns, pose drift, and depth noise can break or distort boundaries. The current pipeline does not reliably resolve these cases.
- Ceiling height is estimated from the point cloud with a residential-height fallback. The result is not a substitute for a measured ceiling height.
- The `photos` and `video` tiers are experimental. Their pose/depth fallbacks can produce approximate or unusable geometry and require additional optional dependencies.
- **Crack detection is not implemented.** There is no crack-specific model, image-processing stage, output field, or validation data in this repository.
- No automated test suite or dependency lock file is currently included.

Do not use generated geometry or measurements for construction, safety, structural assessment, or contractual surveying without independent verification.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `main.py` | CLI entry point and pipeline orchestration |
| `input_adapters.py` | Adapters for LiDAR, video, and photo inputs |
| `generator_3d.py` | RGB-D point-cloud construction and optional TSDF mesh helper |
| `gravity_aligner.py` | Floor/gravity alignment, ceiling estimate, and horizontal-axis alignment |
| `wall_detector.py` | Wall occupancy, floor coverage, room segmentation, and opening estimates |
| `semantic_detector.py` | YOLO-based furniture and fixture detection and 3D projection |
| `floorplan_generator.py` | PNG plan/coverage rendering and JSON metrics |
| `yolov8n.pt` | YOLO object-detection weights used by the semantic detector |
| `yolov8n-seg.pt` | Segmentation weights present in the repository; not currently used by the pipeline |
| `raw_data/sample_lidar/` | Sample RGB video, depth maps, odometry, and extracted frames |
| `output/` | Generated point cloud, images, and JSON metrics |

## Requirements

- Windows, macOS, or Linux
- Python 3.10-3.12 recommended
- The core LiDAR pipeline uses NumPy, Open3D, OpenCV, Matplotlib, and Ultralytics/PyTorch
- Sufficient memory and disk space for RGB-D reconstruction; the included sample contains over 1,700 frames and generates a multi-million-point cloud

## Setup (Windows PowerShell)

Run these commands from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy open3d opencv-python matplotlib ultralytics
```

If PowerShell blocks virtual-environment activation, use the environment's interpreter directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install numpy open3d opencv-python matplotlib ultralytics
```

The object detector expects `yolov8n.pt` in the current working directory. Run commands from the repository root or update the model-loading path before running from elsewhere. Ultralytics installs PyTorch as a dependency; GPU-specific PyTorch installation may require a package command matched to the installed CUDA version.

For the experimental `photos` and `video` tiers, pose estimation uses `pycolmap`, and depth estimation uses `transformers` and Pillow. Install these separately if you are evaluating those tiers:

```powershell
python -m pip install pycolmap transformers pillow
```

These optional dependencies do not make the photo/video results production-ready. In particular, the current depth scaling is heuristic and should not be treated as calibrated metric depth.

## Run the Included LiDAR Sample

```powershell
python main.py --input raw_data/sample_lidar --tier lidar --output output
```

The LiDAR input directory is expected to contain:

```text
sample_lidar/
	rgb.mp4
	odometry.csv
	depth/
		000000.png
		000001.png
		...
```

The depth images must be 16-bit PNGs with depth values in millimeters. The odometry CSV must contain frame indices, position (`x`, `y`, `z`), quaternion (`qx`, `qy`, `qz`, `qw`), and, where available, camera intrinsics (`fx`, `fy`, `cx`, `cy`). The adapter matches every fifth video frame with odometry and a matching depth image.

Other supported CLI forms are:

```powershell
python main.py --input path\to\room_photos --tier photos --output output\photos
python main.py --input path\to\walkthrough.mp4 --tier video --output output\video
```

Photo input should be a directory of `.jpg`, `.jpeg`, `.png`, or `.heic` images. The video tier accepts a video file. These two tiers are experimental and may fail or produce only approximate geometry without the optional pose/depth packages.

## Generated Artifacts

| Artifact | Description |
| --- | --- |
| `raw_cloud.ply` | Fused point cloud before gravity alignment |
| `plan.png` | Room plan when rooms are detected; otherwise a measured floor-coverage sheet |
| `output.json` | Room/object metrics and a `status` field; `partial` means reliable room polygons were not recovered |
| `occupancy_grid.png` | Debug image of wall-height returns after filtering |
| `floor_coverage.png` | Debug image of returns near the estimated floor |

Always inspect both `plan.png` and `output.json`. A `partial` status and an empty `property.rooms` list mean the result is **not a room layout**.

## Roadmap

1. **Reliable room geometry:** improve pose-drift handling, wall-plane extraction, corner and doorway detection, and room-topology validation. Add confidence estimates and reject unsupported dimensions instead of guessing.
2. **Crack detection:** define the target crack types and minimum detectable width; capture high-resolution, well-lit close-ups; assemble labeled training/validation data; train or integrate a crack-specific detector; map image detections back to 3D surfaces; and report calibrated confidence and scale. The current furniture detector cannot fulfill this requirement.
3. **Calibration and input validation:** use per-frame intrinsics consistently, validate RGB/depth registration and units, and report missing or mismatched frames before reconstruction.
4. **Quality and reproducibility:** add automated unit/integration tests, dependency manifests and pinned versions, representative sample fixtures, and repeatable accuracy checks against measured ground truth.
5. **Usable deliverables:** distinguish coverage maps from architectural plans in the UI and exports; add review/edit tools for walls, doors, room labels, dimensions, and confidence.

## Development Notes

- Run the CLI from the repository root so relative paths to sample data and YOLO weights resolve.
- The pipeline writes intermediate and final files into the selected output directory. Use a separate output directory to preserve a previous run.
- Review estimates against independent measurements before treating them as facts. The current output is exploratory and has not been validated to a surveying accuracy standard.
