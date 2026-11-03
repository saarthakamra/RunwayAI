"""
Pure Monocular Geometry Engine:
Computes vanishing point, visual horizon, relative heading error,
homography matrix for Inverse Perspective Mapping (IPM), and aspect ratio (L/W).
"""

import cv2
import numpy as np


class MonocularGeometryEngine:
    def __init__(self, camera_matrix):
        """
        camera_matrix: 3x3 intrinsic matrix K
        [[fx, 0, cx],
         [0, fy, cy],
         [0,  0,  1]]
        """
        self.K = np.asarray(camera_matrix, dtype=np.float64)
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]

    def line_intersection(self, line1, line2):
        """
        Computes intersection of two 2D lines.
        Each line is defined by two points: ((x1, y1), (x2, y2)).
        """
        (x1, y1), (x2, y2) = line1
        (x3, y3), (x4, y4) = line2

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return None  # Parallel lines

        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
        return np.array([px, py], dtype=np.float64)

    def compute_vanishing_point(self, runway_corners):
        """
        runway_corners: 4 points ordered [near_left, near_right, far_right, far_left]
        Left edge: (near_left, far_left)
        Right edge: (near_right, far_right)
        """
        p_nl, p_nr, p_fr, p_fl = runway_corners
        left_edge = (p_nl, p_fl)
        right_edge = (p_nr, p_fr)

        vp = self.line_intersection(left_edge, right_edge)
        return vp

    def compute_relative_heading_error(self, vp):
        """
        Calculates relative aircraft heading deviation (yaw error) from vanishing point.
        Angle in degrees: positive = runway is to the right, negative = to the left.
        """
        if vp is None:
            return 0.0
        u_v, v_v = vp
        delta_psi_rad = np.arctan((u_v - self.cx) / self.fx)
        return float(np.degrees(delta_psi_rad))

    def compute_glideslope_indicator(self, vp, runway_threshold_center):
        """
        Estimates visual vertical glideslope vector.
        Angle between horizon (at vanishing point) and runway threshold.
        """
        if vp is None or runway_threshold_center is None:
            return 3.0
        dy = runway_threshold_center[1] - vp[1]
        pitch_angle_rad = np.arctan(max(1.0, dy) / self.fy)
        return float(np.degrees(pitch_angle_rad))

    def compute_ipm_aspect_ratio(self, runway_corners, output_size=(300, 600)):
        """
        Inverse Perspective Mapping (IPM):
        Maps the perspective trapezoid of the runway into an orthorectified top-down bird's-eye view.
        Returns:
            ipm_image_homography: 3x3 homography matrix H
            unwarped_aspect_ratio: Estimated physical Length-to-Width ratio (L/W)
        """
        pts_src = np.array(runway_corners, dtype=np.float32)
        p_nl, p_nr, p_fr, p_fl = pts_src

        # Calculate visual foreshortening ratio using cross-ratio invariant
        # In projective geometry, the cross-ratio of 4 collinear points is preserved.
        # For a trapezoid ABCD on ground plane with parallel sides (threshold and far end),
        # Width near = ||p_nr - p_nl||, Width far = ||p_fr - p_fl||
        w_near = np.linalg.norm(p_nr - p_nl)
        w_far = np.linalg.norm(p_fr - p_fl)

        # Distance from threshold to vanishing point vs height
        vp = self.compute_vanishing_point(runway_corners)
        if vp is not None and w_far > 1.0:
            scale_ratio = w_near / w_far
            # Estimated perspective aspect ratio (L / W)
            # Length foreshortens quadratically with distance in perspective projection
            v_thresh = (p_nl[1] + p_nr[1]) / 2.0
            v_far = (p_fl[1] + p_fr[1]) / 2.0
            v_vp = vp[1]
            
            delta_far = max(0.5, abs(v_far - v_vp))
            delta_near = max(1.0, abs(v_thresh - v_vp))
            
            # Metric aspect ratio approximation via perspective depth scaling
            depth_near = self.fy / delta_near
            depth_far = self.fy / delta_far
            estimated_metric_length = abs(depth_far - depth_near)
            estimated_metric_width = (w_near * depth_near) / self.fx
            estimated_aspect_ratio = max(1.0, estimated_metric_length / max(0.1, estimated_metric_width))
        else:
            estimated_aspect_ratio = 10.0  # Fallback standard

        # Compute Homography for Top-Down Visual Inset
        out_w, out_h = output_size
        margin = 30
        pts_dst = np.array([
            [margin,         out_h - margin],  # near left
            [out_w - margin, out_h - margin],  # near right
            [out_w - margin, margin        ],  # far right
            [margin,         margin        ]   # far left
        ], dtype=np.float32)

        H, _ = cv2.findHomography(pts_src, pts_dst)
        return H, float(estimated_aspect_ratio)

    def extract_topdown_ortho_view(self, frame, H, output_size=(300, 600)):
        """Warps the perspective camera frame into an orthorectified top-down bird's-eye view."""
        if H is None:
            return np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
        top_down = cv2.warpPerspective(frame, H, output_size, flags=cv2.INTER_LINEAR)
        return top_down
