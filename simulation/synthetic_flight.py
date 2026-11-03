"""
Synthetic Flight Trajectory Generator:
Simulates realistic approach profiles for:
- Commercial / Big Aircraft (Standard 3° Glideslope, 140 kts, long approach path)
- RC Planes / Small Fixed-Wing UAVs (6° Glideslope, 18 m/s, short approach path)
"""

import numpy as np


class FlightTrajectory:
    @staticmethod
    def generate_commercial_approach(num_frames=60, lateral_drift_m=10.0, crab_angle_deg=2.5):
        """
        Generates 6-DoF trajectory for a commercial airliner landing approach:
        - Glideslope: ~3.0°
        - Distance to threshold: from -2400m to -200m
        - Altitude: from ~130m to ~15m
        - Heading / Crab: small crab angle due to simulated crosswind
        """
        x_positions = np.linspace(-2200.0, -150.0, num_frames)
        glideslope_rad = np.radians(3.0)
        
        trajectory = []
        for i, x in enumerate(x_positions):
            dist_to_threshold = abs(x)
            alt = max(10.0, dist_to_threshold * np.tan(glideslope_rad))
            
            # Simulated smooth lateral alignment correction
            progress = i / float(num_frames - 1)
            y = lateral_drift_m * (1.0 - progress)
            
            # Pilot decrabs as they get closer to threshold
            yaw = crab_angle_deg * (1.0 - 0.7 * progress)
            pitch = -3.0  # Nose pitched slightly down on glideslope
            roll = -1.2 * np.sin(progress * np.pi)  # Slight bank angle correction
            
            cam_pos = np.array([x, y, alt])
            cam_angles = np.array([yaw, pitch, roll])
            trajectory.append((cam_pos, cam_angles))
            
        return trajectory

    @staticmethod
    def generate_rc_approach(num_frames=60, lateral_drift_m=4.0, crab_angle_deg=4.0):
        """
        Generates 6-DoF trajectory for an RC plane / fixed-wing UAV:
        - Glideslope: ~6.0°
        - Distance to threshold: from -220m to -20m
        - Altitude: from ~23m to ~2.5m
        """
        x_positions = np.linspace(-220.0, -15.0, num_frames)
        glideslope_rad = np.radians(6.0)
        
        trajectory = []
        for i, x in enumerate(x_positions):
            dist_to_threshold = abs(x)
            alt = max(2.0, dist_to_threshold * np.tan(glideslope_rad))
            
            progress = i / float(num_frames - 1)
            y = lateral_drift_m * (1.0 - progress)
            
            yaw = crab_angle_deg * (1.0 - progress)
            pitch = -5.0
            roll = -2.0 * np.sin(progress * np.pi * 1.5)
            
            cam_pos = np.array([x, y, alt])
            cam_angles = np.array([yaw, pitch, roll])
            trajectory.append((cam_pos, cam_angles))
            
        return trajectory
