"""
Aviation HUD Renderer:
Renders professional flight HUD overlays:
- Artificial horizon line & pitch ladder
- Camera boresight vs. runway vanishing point reticle
- Runway polygon overlay with corner keypoints
- Orthorectified top-down IPM bird's-eye view inset
- Dual-category feasibility telemetry panels (Big Aircraft vs. RC Plane)
"""

import cv2
import numpy as np


class HUDRenderer:
    def __init__(self, camera_fx, camera_fy, cx, cy):
        self.fx = camera_fx
        self.fy = camera_fy
        self.cx = cx
        self.cy = cy

        # HUD Color Palette
        self.HUD_CYAN = (235, 206, 0)     # BGR: Tactical Cyan
        self.HUD_GREEN = (50, 255, 50)    # BGR: Bright Green
        self.HUD_YELLOW = (0, 215, 255)   # BGR: Alert Amber
        self.HUD_RED = (40, 40, 255)      # BGR: Danger Red
        self.HUD_WHITE = (245, 245, 245)
        self.HUD_DARK = (15, 15, 20)

    def draw_hud(self, frame, detection_res, vp, heading_err, aspect_ratio, ortho_view, feasibility):
        """Draws full HUD display on frame."""
        h, w = frame.shape[:2]
        hud_frame = frame.copy()

        # 1. Boresight Crosshair (Center of camera view)
        bx, by = int(self.cx), int(self.cy)
        cv2.line(hud_frame, (bx - 15, by), (bx + 15, by), self.HUD_CYAN, 1, cv2.LINE_AA)
        cv2.line(hud_frame, (bx, by - 15), (bx, by + 15), self.HUD_CYAN, 1, cv2.LINE_AA)
        cv2.circle(hud_frame, (bx, by), 6, self.HUD_CYAN, 1, cv2.LINE_AA)

        # 2. Runway Polygon and Keypoints
        if detection_res is not None and "corners" in detection_res:
            corners = detection_res["corners"].astype(np.int32)
            # Polyline around runway
            cv2.polylines(hud_frame, [corners], isClosed=True, color=self.HUD_CYAN, thickness=2, lineType=cv2.LINE_AA)

            # Draw corner markers
            labels = ["TH-L", "TH-R", "END-R", "END-L"]
            for i, pt in enumerate(corners):
                cv2.circle(hud_frame, tuple(pt), 4, self.HUD_GREEN, -1, cv2.LINE_AA)
                cv2.putText(hud_frame, labels[i], (pt[0] - 20, pt[1] + 18), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.HUD_WHITE, 1, cv2.LINE_AA)

            # Draw Obstacle Bounding Box if detected
            if detection_res.get("has_obstacles") and detection_res.get("obstacle_bbox") is not None:
                ox, oy, ow, oh = detection_res["obstacle_bbox"]
                cv2.rectangle(hud_frame, (ox, oy), (ox + ow, oy + oh), self.HUD_RED, 2, cv2.LINE_AA)
                cv2.putText(hud_frame, "! OBSTACLE DETECTED !", (ox - 30, oy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.HUD_RED, 2, cv2.LINE_AA)

        # 3. Vanishing Point (Runway Alignment Vector)
        if vp is not None:
            vpx, vpy = int(vp[0]), int(vp[1])
            # Check if within reasonable screen bounds
            if -w <= vpx <= 2 * w and -h <= vpy <= 2 * h:
                # Horizon line through vanishing point
                cv2.line(hud_frame, (0, vpy), (w, vpy), (100, 150, 100), 1, cv2.LINE_AA)
                # VP diamond reticle
                sz = 8
                pts_diamond = np.array([
                    [vpx, vpy - sz],
                    [vpx + sz, vpy],
                    [vpx, vpy + sz],
                    [vpx - sz, vpy]
                ], dtype=np.int32)
                cv2.polylines(hud_frame, [pts_diamond], isClosed=True, color=self.HUD_YELLOW, thickness=1, lineType=cv2.LINE_AA)
                cv2.putText(hud_frame, f"VP (ALIGN: {heading_err:+.1f} deg)", (vpx + 12, vpy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.HUD_YELLOW, 1, cv2.LINE_AA)

        # 4. Top-Right Inset: Inverse Perspective Mapping (IPM Top-Down View)
        if ortho_view is not None:
            inset_w, inset_h = 160, 220
            inset_resized = cv2.resize(ortho_view, (inset_w, inset_h))
            
            # Position at top right
            ix = w - inset_w - 20
            iy = 20

            # Draw backdrop & border
            cv2.rectangle(hud_frame, (ix - 4, iy - 22), (ix + inset_w + 4, iy + inset_h + 4), (20, 25, 30), -1)
            cv2.rectangle(hud_frame, (ix - 4, iy - 22), (ix + inset_w + 4, iy + inset_h + 4), self.HUD_CYAN, 1)
            cv2.putText(hud_frame, "TOP-DOWN (IPM)", (ix + 12, iy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.HUD_CYAN, 1, cv2.LINE_AA)
            hud_frame[iy:iy + inset_h, ix:ix + inset_w] = inset_resized

        # 5. Top Left Banner: System Mode & Pure Monocular Indicator
        self._draw_panel(hud_frame, 20, 20, 360, 80)
        cv2.putText(hud_frame, "RUNWAY AI -- DUAL-MODE FEASIBILITY", (30, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, self.HUD_CYAN, 1, cv2.LINE_AA)
        cv2.putText(hud_frame, "SENSOR: PURE MONOCULAR VISION (RGB)", (30, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.HUD_WHITE, 1, cv2.LINE_AA)
        
        surf_label = detection_res["surface_type"].upper() if detection_res else "SEARCHING..."
        mark_label = "YES" if (detection_res and detection_res["has_markings"]) else "NO"
        cv2.putText(hud_frame, f"SURFACE: {surf_label}  |  MARKINGS: {mark_label}  |  AR: {aspect_ratio:.1f}", 
                    (30, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 220, 255), 1, cv2.LINE_AA)

        # 6. Bottom Panels: Dual Feasibility Decision Cards
        card_w = 460
        card_h = 125
        margin_y = h - card_h - 20

        # --- Card 1: Big Aircraft Feasibility (Bottom Left) ---
        big_eval = feasibility["big_aircraft"]
        self._draw_panel(hud_frame, 20, margin_y, card_w, card_h)
        cv2.putText(hud_frame, "CATEGORY: BIG AIRCRAFT (JET / GA)", (30, margin_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.HUD_CYAN, 1, cv2.LINE_AA)
        
        # Status Pill
        cv2.putText(hud_frame, f"STATUS: {big_eval['status']}", (30, margin_y + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, big_eval["color"], 2, cv2.LINE_AA)
        
        # Explanatory lines
        for idx, reason in enumerate(big_eval["reasons"][:3]):
            cv2.putText(hud_frame, f"- {reason}", (30, margin_y + 72 + (idx * 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.HUD_WHITE, 1, cv2.LINE_AA)

        # --- Card 2: RC Plane / UAV Feasibility (Bottom Right) ---
        rc_eval = feasibility["rc_plane"]
        rc_x = w - card_w - 20
        self._draw_panel(hud_frame, rc_x, margin_y, card_w, card_h)
        cv2.putText(hud_frame, "CATEGORY: RC PLANE / FIXED-WING UAV", (rc_x + 10, margin_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.HUD_CYAN, 1, cv2.LINE_AA)
        
        # Status Pill
        cv2.putText(hud_frame, f"STATUS: {rc_eval['status']}", (rc_x + 10, margin_y + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, rc_eval["color"], 2, cv2.LINE_AA)

        # Explanatory lines
        for idx, reason in enumerate(rc_eval["reasons"][:3]):
            cv2.putText(hud_frame, f"- {reason}", (rc_x + 10, margin_y + 72 + (idx * 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.HUD_WHITE, 1, cv2.LINE_AA)

        return hud_frame

    def _draw_panel(self, frame, x, y, w, h):
        """Draws a semi-transparent HUD panel with a tactical glowing border."""
        sub_img = frame[y:y+h, x:x+w]
        black_rect = np.zeros(sub_img.shape, dtype=np.uint8)
        # 75% dark tint overlay
        tinted = cv2.addWeighted(sub_img, 0.25, black_rect, 0.75, 1.0)
        frame[y:y+h, x:x+w] = tinted
        cv2.rectangle(frame, (x, y), (x+w, y+h), (70, 90, 110), 1, cv2.LINE_AA)
        # Corner accent tabs
        t_len = 8
        cv2.line(frame, (x, y), (x + t_len, y), self.HUD_CYAN, 2)
        cv2.line(frame, (x, y), (x, y + t_len), self.HUD_CYAN, 2)
        cv2.line(frame, (x + w, y), (x + w - t_len, y), self.HUD_CYAN, 2)
        cv2.line(frame, (x + w, y), (x + w, y + t_len), self.HUD_CYAN, 2)
        cv2.line(frame, (x, y + h), (x + t_len, y + h), self.HUD_CYAN, 2)
        cv2.line(frame, (x, y + h), (x, y + h - t_len), self.HUD_CYAN, 2)
        cv2.line(frame, (x + w, y + h), (x + w - t_len, y + h), self.HUD_CYAN, 2)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - t_len), self.HUD_CYAN, 2)
