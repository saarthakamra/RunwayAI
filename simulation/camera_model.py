"""
Camera Model: Pinhole camera geometry and 3D-to-2D projection engine.
Pure monocular camera with intrinsic matrix K and 6-DoF pose (position + orientation).
Uses standard aeronautical World-to-Body (NED) and Body-to-Camera transformations.
"""

import numpy as np


class PinholeCamera:
    def __init__(self, width=1280, height=720, hfov_deg=65.0):
        self.width = width
        self.height = height
        self.hfov_deg = hfov_deg
        
        # Calculate focal length from horizontal FOV
        hfov_rad = np.radians(hfov_deg)
        self.fx = (width / 2.0) / np.tan(hfov_rad / 2.0)
        self.fy = self.fx  # Square pixels
        self.cx = width / 2.0
        self.cy = height / 2.0

        # Intrinsic Matrix K
        self.K = np.array([
            [self.fx, 0.0,     self.cx],
            [0.0,     self.fy, self.cy],
            [0.0,     0.0,     1.0    ]
        ], dtype=np.float64)

    def get_rotation_matrix(self, yaw_deg, pitch_deg, roll_deg):
        """
        Aviation convention:
        - World frame: X forward along runway, Y right, Z up (Altitude > 0)
        - Aircraft body frame: X forward, Y right, Z down
        - Camera frame: X right, Y down, Z forward (OpenCV standard)
        """
        yaw = np.radians(yaw_deg)
        pitch = np.radians(pitch_deg)
        roll = np.radians(roll_deg)

        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)

        # Standard aeronautical Direction Cosine Matrix (DCM):
        Rz = np.array([[cy, sy, 0], [-sy, cy, 0], [0, 0, 1]])
        Ry = np.array([[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]])
        Rx = np.array([[1, 0, 0], [0, cr, sr], [0, -sr, cr]])

        # World is Z-up; convert to NED (Z-down) first:
        R_up2ned = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64)
        R_w2b = Rx @ Ry @ Rz @ R_up2ned

        # From Body (X fwd, Y right, Z down) to Camera (X right, Y down, Z forward):
        R_b2c = np.array([
            [0,  1,  0],
            [0,  0,  1],
            [1,  0,  0]
        ], dtype=np.float64)

        return R_b2c @ R_w2b

    def project_3d_to_2d(self, points_world, cam_pos, cam_angles):
        """
        Projects 3D world points (N, 3) to 2D image coordinates (N, 2).
        Returns:
            uv: (N, 2) pixel coordinates
            valid_mask: (N,) boolean array where Z_cam > 0.2 (in front of camera)
        """
        points_world = np.atleast_2d(points_world)
        cam_pos = np.asarray(cam_pos, dtype=np.float64)
        
        yaw, pitch, roll = cam_angles
        R = self.get_rotation_matrix(yaw, pitch, roll)

        diff = points_world - cam_pos
        points_cam = (R @ diff.T).T  # (N, 3)

        z_cam = points_cam[:, 2]
        valid_mask = z_cam > 0.2

        uv = np.zeros((len(points_world), 2), dtype=np.float64)
        safe_z = np.where(valid_mask, z_cam, 1.0)
        u = (self.fx * points_cam[:, 0] / safe_z) + self.cx
        v = (self.fy * points_cam[:, 1] / safe_z) + self.cy

        uv[:, 0] = u
        uv[:, 1] = v
        return uv, valid_mask
