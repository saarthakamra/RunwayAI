"""
Procedural 3D World Generator:
Renders synthetic camera frames for:
1. Certified Commercial Runway (Asphalt, threshold piano keys, centerline, aiming markers)
2. RC Plane Grass Airfield / Unpaved Strip (Mowed grass, rough meadow surroundings, perimeter trees)
3. Obstructed / Infeasible Field (Obstacle in touchdown flare corridor)
"""

import cv2
import numpy as np
from simulation.camera_model import PinholeCamera


class WorldGenerator:
    def __init__(self, camera: PinholeCamera):
        self.camera = camera
        self.w = camera.width
        self.h = camera.height

    def create_sky_ground_base(self, cam_pos, cam_angles):
        """Creates an atmospheric sky gradient and ground plane background."""
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        
        # Calculate true horizon in image plane from distant point
        horizon_pt, _ = self.camera.project_3d_to_2d([100000.0, 0.0, 0.0], cam_pos, cam_angles)
        horizon_y = int(np.clip(horizon_pt[0, 1], 0, self.h))

        # Sky gradient (Deep aviation blue down to atmospheric horizon haze)
        if horizon_y > 0:
            for y in range(horizon_y):
                alpha = y / max(1, horizon_y)
                b = int(220 + 25 * alpha)
                g = int(140 + 60 * alpha)
                r = int(60 + 130 * alpha)
                frame[y, :] = (b, g, r)

        # Ground base (Natural terrain tone)
        if horizon_y < self.h:
            for y in range(horizon_y, self.h):
                alpha = (y - horizon_y) / max(1, self.h - horizon_y)
                b = int(45 - 15 * alpha)
                g = int(85 + 25 * alpha)
                r = int(65 - 10 * alpha)
                frame[y, :] = (b, g, r)

        return frame

    def render_quad_3d(self, frame, corners_3d, cam_pos, cam_angles, color, border_color=None):
        """Projects a 3D planar quad into 2D and fills it on the frame."""
        uv, valid = self.camera.project_3d_to_2d(corners_3d, cam_pos, cam_angles)
        if not np.all(valid):
            return frame

        pts = uv.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(frame, [pts], color, lineType=cv2.LINE_AA)
        if border_color is not None:
            cv2.polylines(frame, [pts], isClosed=True, color=border_color, thickness=1, lineType=cv2.LINE_AA)
        return frame

    def render_commercial_runway(self, cam_pos, cam_angles):
        """
        Renders a certified commercial runway:
        - 2200m length x 45m width
        - Dark asphalt surface (RGB 55, 55, 58)
        - White runway edge lines, threshold keys, centerline dashes, aiming markers
        """
        frame = self.create_sky_ground_base(cam_pos, cam_angles)

        rw_length = 2200.0
        rw_width = 45.0
        half_w = rw_width / 2.0

        # Surrounding Airport Apron
        apron_3d = np.array([
            [-150.0, -90.0, 0.0],
            [-150.0, 90.0, 0.0],
            [rw_length + 200.0, 90.0, 0.0],
            [rw_length + 200.0, -90.0, 0.0]
        ])
        self.render_quad_3d(frame, apron_3d, cam_pos, cam_angles, (45, 95, 55))

        # Main Runway Asphalt Surface (Dark Slate Gray: BGR 55, 55, 58)
        runway_3d = np.array([
            [0.0, -half_w, 0.0],
            [0.0, half_w, 0.0],
            [rw_length, half_w, 0.0],
            [rw_length, -half_w, 0.0]
        ])
        self.render_quad_3d(frame, runway_3d, cam_pos, cam_angles, (55, 55, 58), border_color=(220, 220, 220))

        # Threshold Piano Keys (10 stripes across threshold)
        num_keys = 10
        key_length = 30.0
        key_width = 1.8
        key_gap = (rw_width - 4.0 - (num_keys * key_width)) / (num_keys - 1)
        start_y = -half_w + 2.0

        for i in range(num_keys):
            ky = start_y + i * (key_width + key_gap)
            key_3d = np.array([
                [6.0, ky, 0.01],
                [6.0, ky + key_width, 0.01],
                [6.0 + key_length, ky + key_width, 0.01],
                [6.0 + key_length, ky, 0.01]
            ])
            self.render_quad_3d(frame, key_3d, cam_pos, cam_angles, (240, 240, 240))

        # Aiming Point Markers (Large blocks 320m down runway)
        for side in [-1, 1]:
            aim_y = side * (half_w * 0.45)
            aim_w = 4.0
            aim_3d = np.array([
                [320.0, aim_y - aim_w/2, 0.01],
                [320.0, aim_y + aim_w/2, 0.01],
                [370.0, aim_y + aim_w/2, 0.01],
                [370.0, aim_y - aim_w/2, 0.01]
            ])
            self.render_quad_3d(frame, aim_3d, cam_pos, cam_angles, (245, 245, 245))

        # Centerline Dashes (30m stripe, 20m gap)
        dash_len = 30.0
        gap_len = 20.0
        dash_w = 0.9
        cur_x = 60.0
        while cur_x < rw_length - 60.0:
            dash_3d = np.array([
                [cur_x, -dash_w/2, 0.01],
                [cur_x, dash_w/2, 0.01],
                [cur_x + dash_len, dash_w/2, 0.01],
                [cur_x + dash_len, -dash_w/2, 0.01]
            ])
            self.render_quad_3d(frame, dash_3d, cam_pos, cam_angles, (240, 240, 240))
            cur_x += (dash_len + gap_len)

        gt_metadata = {
            "type": "certified_commercial",
            "surface": "asphalt",
            "length_m": rw_length,
            "width_m": rw_width,
            "aspect_ratio": rw_length / rw_width,
            "corners_3d": runway_3d,
            "has_markings": True,
            "has_obstacles": False
        }
        return frame, gt_metadata

    def render_rc_grass_strip(self, cam_pos, cam_angles, has_obstacle=False):
        """
        Renders an unpaved mowed grass airfield for RC planes & UAVs:
        - 90m length x 12m width (vibrant fairway green: BGR 45, 145, 70)
        - Perimeter trees and corner boundary markers
        - Optional obstacle on strip
        """
        frame = self.create_sky_ground_base(cam_pos, cam_angles)

        strip_length = 90.0
        strip_width = 12.0
        half_w = strip_width / 2.0

        # Rough meadow base (Darker olive-green)
        meadow_3d = np.array([
            [-40.0, -45.0, 0.0],
            [-40.0, 45.0, 0.0],
            [strip_length + 60.0, 45.0, 0.0],
            [strip_length + 60.0, -45.0, 0.0]
        ])
        self.render_quad_3d(frame, meadow_3d, cam_pos, cam_angles, (35, 75, 45))

        # Mowed Landing Strip (Lighter vibrant green fairway: BGR 45, 145, 70)
        strip_3d = np.array([
            [0.0, -half_w, 0.0],
            [0.0, half_w, 0.0],
            [strip_length, half_w, 0.0],
            [strip_length, -half_w, 0.0]
        ])
        self.render_quad_3d(frame, strip_3d, cam_pos, cam_angles, (45, 145, 70), border_color=(60, 175, 85))

        # White corner boundary cones
        corner_markers = [
            [0.0, -half_w], [0.0, half_w],
            [strip_length, -half_w], [strip_length, half_w]
        ]
        for cx, cy in corner_markers:
            cone_3d = np.array([
                [cx - 0.4, cy - 0.4, 0.01],
                [cx - 0.4, cy + 0.4, 0.01],
                [cx + 0.4, cy + 0.4, 0.01],
                [cx + 0.4, cy - 0.4, 0.01]
            ])
            self.render_quad_3d(frame, cone_3d, cam_pos, cam_angles, (245, 245, 245))

        # Perimeter Trees
        tree_positions = [
            [-10.0, -20.0], [25.0, -22.0], [60.0, -20.0], [85.0, -21.0],
            [-10.0, 20.0], [30.0, 22.0], [65.0, 21.0], [90.0, 22.0]
        ]
        for tx, ty in tree_positions:
            canopy_3d = np.array([
                [tx - 2.0, ty - 2.0, 3.5],
                [tx - 2.0, ty + 2.0, 3.5],
                [tx + 2.0, ty + 2.0, 3.5],
                [tx + 2.0, ty - 2.0, 3.5]
            ])
            self.render_quad_3d(frame, canopy_3d, cam_pos, cam_angles, (20, 60, 25))

        # Obstacle on Runway (3D Hazard Red Block / Buggy: BGR 20, 20, 230)
        if has_obstacle:
            obs_x, obs_y = 35.0, 0.0
            obs_box_3d = np.array([
                [obs_x - 2.0, obs_y - 1.5, 1.2],
                [obs_x - 2.0, obs_y + 1.5, 1.2],
                [obs_x + 2.0, obs_y + 1.5, 1.2],
                [obs_x + 2.0, obs_y - 1.5, 1.2]
            ])
            self.render_quad_3d(frame, obs_box_3d, cam_pos, cam_angles, (20, 20, 230), border_color=(255, 255, 255))

        gt_metadata = {
            "type": "unpaved_rc_strip",
            "surface": "mowed_grass",
            "length_m": strip_length,
            "width_m": strip_width,
            "aspect_ratio": strip_length / strip_width,
            "corners_3d": strip_3d,
            "has_markings": False,
            "has_obstacles": has_obstacle
        }
        return frame, gt_metadata
