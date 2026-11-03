"""
RunwayAI Main Pipeline:
Demonstrates pure monocular vision-based runway detection and dual-category feasibility assessment:
- Big Commercial Aircraft vs. RC Plane / Tactical UAV.
- Certified Paved Runway vs. Unpaved Grass Strip vs. Obstructed Hazard.
"""

import os
import sys
import cv2
import numpy as np

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation.camera_model import PinholeCamera
from vision.runway_detector import RunwayDetector
from vision.monocular_geometry import MonocularGeometryEngine
from feasibility.evaluator import RunwayFeasibilityEvaluator
from visualizer.hud_renderer import HUDRenderer
from demo_scenarios import DemoScenarios


def process_scenario(name, frames_data, camera, detector, geometry, evaluator, hud, output_dir):
    print(f"\n=======================================================")
    print(f"Executing: {name} ({len(frames_data)} frames)")
    print(f"=======================================================")

    video_path = os.path.join(output_dir, f"{name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, 20.0, (camera.width, camera.height))

    last_hud_frame = None
    mid_hud_frame = None

    for idx, (raw_frame, gt, cam_pos, cam_angles) in enumerate(frames_data):
        # 1. Monocular Runway Detection
        det = detector.detect(raw_frame)

        vp = None
        heading_err = 0.0
        aspect_ratio = 1.0
        ortho_view = None

        if det is not None:
            corners = det["corners"]
            # 2. Monocular Geometry: Vanishing point & Heading error
            vp = geometry.compute_vanishing_point(corners)
            heading_err = geometry.compute_relative_heading_error(vp)

            # 3. Inverse Perspective Mapping (IPM) & Aspect Ratio
            H, aspect_ratio = geometry.compute_ipm_aspect_ratio(corners)
            ortho_view = geometry.extract_topdown_ortho_view(raw_frame, H)

        # 4. Feasibility Decision Engine
        if det is not None:
            feasibility = evaluator.evaluate_all(det, aspect_ratio, heading_err)
        else:
            feasibility = {
                "big_aircraft": {"status": "SEARCHING", "color": (120, 120, 120), "reasons": ["No candidate strip detected in frame"]},
                "rc_plane": {"status": "SEARCHING", "color": (120, 120, 120), "reasons": ["No suitable patch identified"]}
            }

        # 5. Tactical Aviation HUD Rendering
        hud_frame = hud.draw_hud(raw_frame, det, vp, heading_err, aspect_ratio, ortho_view, feasibility)
        writer.write(hud_frame)

        if idx == len(frames_data) // 2:
            mid_hud_frame = hud_frame
        if idx == len(frames_data) - 1:
            last_hud_frame = hud_frame

        # Print periodic progress
        if idx % 20 == 0 or idx == len(frames_data) - 1:
            big_stat = feasibility["big_aircraft"]["status"]
            rc_stat = feasibility["rc_plane"]["status"]
            surf = det["surface_type"] if det else "NONE"
            print(f"Frame [{idx:02d}/{len(frames_data):02d}] Alt: {cam_pos[2]:5.1f}m | Surf: {surf:11s} | HeadingErr: {heading_err:+4.1f}° | Big: {big_stat:18s} | RC: {rc_stat}")

    writer.release()
    print(f"[+] Saved HUD Video: {video_path}")

    # Save snapshot images
    if mid_hud_frame is not None:
        snap_path = os.path.join(output_dir, f"{name}_mid_approach.png")
        cv2.imwrite(snap_path, mid_hud_frame)
    if last_hud_frame is not None:
        snap_path = os.path.join(output_dir, f"{name}_touchdown_phase.png")
        cv2.imwrite(snap_path, last_hud_frame)

    return video_path


def main():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    print("Initializing RunwayAI Monocular Detection & Feasibility Engine...")
    camera = PinholeCamera(width=1280, height=720, hfov_deg=65.0)
    detector = RunwayDetector()
    geometry = MonocularGeometryEngine(camera.K)
    evaluator = RunwayFeasibilityEvaluator()
    hud = HUDRenderer(camera.fx, camera.fy, camera.cx, camera.cy)
    scenarios = DemoScenarios(camera)

    # Run Demonstration Scenarios
    scenarios_to_run = [
        scenarios.run_commercial_scenario(num_frames=60),
        scenarios.run_rc_grass_scenario(num_frames=60),
        scenarios.run_obstructed_scenario(num_frames=60)
    ]

    for name, frames_data in scenarios_to_run:
        process_scenario(name, frames_data, camera, detector, geometry, evaluator, hud, output_dir)

    print("\n=======================================================")
    print("ALL DEMONSTRATION RUNS COMPLETED SUCCESSFULLY!")
    print(f"Output artifacts available in: {output_dir}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
