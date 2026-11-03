"""
Demonstration Scenarios:
Encapsulates test cases demonstrating multi-avenue feasibility:
1. Certified Commercial Runway
2. RC Plane Mowed Grass Strip
3. Obstructed Hazard Strip
"""

from simulation.camera_model import PinholeCamera
from simulation.world_generator import WorldGenerator
from simulation.synthetic_flight import FlightTrajectory


class DemoScenarios:
    def __init__(self, camera: PinholeCamera):
        self.camera = camera
        self.world = WorldGenerator(camera)

    def run_commercial_scenario(self, num_frames=60):
        """Scenario A: Certified Commercial Airport Approach."""
        traj = FlightTrajectory.generate_commercial_approach(num_frames=num_frames, lateral_drift_m=12.0, crab_angle_deg=2.5)
        frames_data = []
        for cam_pos, cam_angles in traj:
            frame, gt = self.world.render_commercial_runway(cam_pos, cam_angles)
            frames_data.append((frame, gt, cam_pos, cam_angles))
        return "Scenario_A_Certified_Airport", frames_data

    def run_rc_grass_scenario(self, num_frames=60):
        """Scenario B: Unpaved Grass Strip for RC / UAV."""
        traj = FlightTrajectory.generate_rc_approach(num_frames=num_frames, lateral_drift_m=3.5, crab_angle_deg=4.0)
        frames_data = []
        for cam_pos, cam_angles in traj:
            frame, gt = self.world.render_rc_grass_strip(cam_pos, cam_angles, has_obstacle=False)
            frames_data.append((frame, gt, cam_pos, cam_angles))
        return "Scenario_B_RC_Grass_Strip", frames_data

    def run_obstructed_scenario(self, num_frames=60):
        """Scenario C: Obstructed / Hazard Strip (Go-Around Trigger)."""
        traj = FlightTrajectory.generate_rc_approach(num_frames=num_frames, lateral_drift_m=2.0, crab_angle_deg=2.0)
        frames_data = []
        for cam_pos, cam_angles in traj:
            frame, gt = self.world.render_rc_grass_strip(cam_pos, cam_angles, has_obstacle=True)
            frames_data.append((frame, gt, cam_pos, cam_angles))
        return "Scenario_C_Obstructed_Hazard_Strip", frames_data
