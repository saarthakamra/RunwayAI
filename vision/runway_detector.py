"""
Runway Detector:
Extracts candidate runway polygon, corners (near-left, near-right, far-right, far-left),
classifies surface type (paved asphalt vs mowed grass vs rough terrain),
and detects markings and foreign obstacles.
"""

import cv2
import numpy as np


class RunwayDetector:
    def __init__(self):
        pass

    def detect(self, frame):
        """
        Processes a monocular camera frame to identify:
        - 4 runway corners: [near_left, near_right, far_right, far_left]
        - Surface classification: 'asphalt', 'mowed_grass', 'unpaved_dirt'
        - Markings presence: boolean (has certified threshold/centerline markings)
        - Obstacle presence: boolean and bounding box
        """
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 1. Primary Strategy: Check for Paved Asphalt Runway
        # Asphalt is neutral gray: low saturation, moderate value
        asphalt_mask = cv2.inRange(hsv, np.array([0, 0, 40]), np.array([180, 40, 130]))
        # Markings are bright white
        white_mask = cv2.inRange(hsv, np.array([0, 0, 185]), np.array([180, 45, 255]))

        # Ground plane constraint: runways must be below horizon / upper sky
        ground_roi = np.zeros((h, w), dtype=np.uint8)
        ground_roi[int(h * 0.30):, :] = 255
        asphalt_mask = cv2.bitwise_and(asphalt_mask, asphalt_mask, mask=ground_roi)
        white_mask = cv2.bitwise_and(white_mask, white_mask, mask=ground_roi)

        target_contour = None
        surface_type = "unknown"

        # Only combine white markings if asphalt is genuinely present on the ground
        if np.count_nonzero(asphalt_mask) > 100:
            kernel_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            asphalt_zone = cv2.dilate(asphalt_mask, kernel_dil)
            white_markings = cv2.bitwise_and(white_mask, white_mask, mask=asphalt_zone)
            paved_mask = cv2.bitwise_or(asphalt_mask, white_markings)

            kernel_paved = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            paved_clean = cv2.morphologyEx(paved_mask, cv2.MORPH_CLOSE, kernel_paved, iterations=2)
            paved_cnts, _ = cv2.findContours(paved_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_paved = [c for c in paved_cnts if cv2.contourArea(c) > 60]

            if len(valid_paved) > 0:
                target_contour = max(valid_paved, key=cv2.contourArea)
                surface_type = "asphalt"

        # 2. Secondary Strategy: Check for Mowed Grass Strip (RC Airfield)
        if target_contour is None:
            grass_mask = cv2.inRange(hsv, np.array([40, 90, 85]), np.array([78, 255, 230]))
            grass_mask = cv2.bitwise_and(grass_mask, grass_mask, mask=ground_roi)

            kernel_grass = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            grass_clean = cv2.morphologyEx(grass_mask, cv2.MORPH_CLOSE, kernel_grass, iterations=2)
            grass_cnts, _ = cv2.findContours(grass_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_grass = [c for c in grass_cnts if cv2.contourArea(c) > 60]

            if len(valid_grass) > 0:
                target_contour = max(valid_grass, key=cv2.contourArea)
                surface_type = "mowed_grass"

        if target_contour is None:
            return None

        # 3. Extract Ordered Corners: [near_left, near_right, far_right, far_left]
        corners = self._extract_ordered_corners(target_contour)
        if corners is None:
            return None

        # 4. Create Mask of the Detected Polygon
        mask_poly = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask_poly, [corners.astype(np.int32)], 255)

        # 5. Markings Verification
        white_in_poly = cv2.bitwise_and(white_mask, white_mask, mask=mask_poly)
        white_pixel_ratio = np.count_nonzero(white_in_poly) / float(max(1, np.count_nonzero(mask_poly)))
        has_markings = (white_pixel_ratio > 0.015) and (surface_type == "asphalt")

        # 6. Foreign Obstacle Detection inside Runway Landing Corridor
        red_mask_1 = cv2.inRange(hsv, np.array([0, 70, 70]), np.array([15, 255, 255]))
        red_mask_2 = cv2.inRange(hsv, np.array([165, 70, 70]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)
        red_mask = cv2.bitwise_and(red_mask, red_mask, mask=ground_roi)

        # Check if hazard falls in lateral corridor between near-left and near-right
        x_min_corridor = min(corners[:, 0]) - 30
        x_max_corridor = max(corners[:, 0]) + 30
        y_min_corridor = min(corners[:, 1]) - 40
        y_max_corridor = max(corners[:, 1]) + 10

        corridor_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(corridor_mask, 
                      (int(max(0, x_min_corridor)), int(max(0, y_min_corridor))), 
                      (int(min(w, x_max_corridor)), int(min(h, y_max_corridor))), 255, -1)

        hazard_in_corridor = cv2.bitwise_and(red_mask, red_mask, mask=corridor_mask)

        has_obstacles = False
        obstacle_bbox = None
        red_count = np.count_nonzero(hazard_in_corridor)

        if red_count >= 5:  # Noticeable foreign obstacle in landing corridor
            has_obstacles = True
            obs_cnts, _ = cv2.findContours(hazard_in_corridor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(obs_cnts) > 0:
                biggest_obs = max(obs_cnts, key=cv2.contourArea)
                obstacle_bbox = cv2.boundingRect(biggest_obs)

        return {
            "corners": corners,
            "surface_type": surface_type,
            "has_markings": has_markings,
            "has_obstacles": has_obstacles,
            "obstacle_bbox": obstacle_bbox,
            "contour": target_contour
        }

    def _extract_ordered_corners(self, contour):
        """
        Orders the 4 corners:
        [0]: near-left (bottom-left)
        [1]: near-right (bottom-right)
        [2]: far-right (top-right)
        [3]: far-left (top-left)
        """
        pts = contour.reshape(-1, 2)
        if len(pts) < 4:
            return None

        y_vals = pts[:, 1]
        p80 = np.percentile(y_vals, 80)
        p20 = np.percentile(y_vals, 20)

        bottom_pts = pts[y_vals >= p80]
        top_pts = pts[y_vals <= p20]

        if len(bottom_pts) == 0 or len(top_pts) == 0:
            return None

        near_left = bottom_pts[bottom_pts[:, 0].argmin()]
        near_right = bottom_pts[bottom_pts[:, 0].argmax()]
        far_left = top_pts[top_pts[:, 0].argmin()]
        far_right = top_pts[top_pts[:, 0].argmax()]

        return np.array([near_left, near_right, far_right, far_left], dtype=np.float64)
