"""
Feasibility Decision Engine:
Evaluates whether a candidate landing strip is feasible for:
1. Big Aircraft (Commercial Jet / General Aviation)
2. RC Plane / Small Tactical Fixed-Wing UAV
Based strictly on monocular vision observations (surface, markings, aspect ratio, heading, obstacles).
"""


class RunwayFeasibilityEvaluator:
    def __init__(self):
        # Specifications for Big Aircraft (e.g. B737 / A320 / Regional Jet)
        self.big_aircraft_reqs = {
            "allowed_surfaces": ["asphalt", "concrete"],
            "requires_markings": True,
            "min_aspect_ratio": 6.0,  # Minimum visible perspective aspect ratio
            "max_heading_error_deg": 6.0,
            "must_be_obstacle_free": True
        }

        # Specifications for RC Planes / Small Fixed-Wing UAVs (e.g., 1.5m - 3m wingspan)
        self.rc_plane_reqs = {
            "allowed_surfaces": ["asphalt", "concrete", "mowed_grass", "unpaved_dirt"],
            "requires_markings": False,
            "min_aspect_ratio": 3.0,
            "max_heading_error_deg": 15.0,
            "must_be_obstacle_free": True
        }

    def evaluate_big_aircraft(self, detection_res, aspect_ratio, heading_error_deg):
        """Evaluates feasibility for commercial / big aircraft."""
        surface = detection_res["surface_type"]
        has_markings = detection_res["has_markings"]
        has_obstacles = detection_res["has_obstacles"]

        reasons = []
        status = "FEASIBLE"
        color = (0, 255, 0)  # Green

        # 1. Obstacle Check
        if has_obstacles:
            return {
                "status": "ABORT / GO-AROUND",
                "color": (0, 0, 255),  # Red
                "reasons": ["CRITICAL: Runway incursion / foreign obstacle detected in touchdown zone!"]
            }

        # 2. Surface Check
        if surface not in self.big_aircraft_reqs["allowed_surfaces"]:
            reasons.append(f"Rejected: Unpaved surface '{surface}' cannot support heavy transport aircraft.")
            status = "INFEASIBLE"
            color = (0, 0, 255)

        # 3. Markings Check (Certified airport runway requirement)
        if self.big_aircraft_reqs["requires_markings"] and not has_markings:
            reasons.append("Non-certified: Missing standard ICAO threshold piano keys / centerline markings.")
            status = "INFEASIBLE"
            color = (0, 0, 255)

        # 4. Aspect Ratio (Length sufficiency)
        if aspect_ratio < self.big_aircraft_reqs["min_aspect_ratio"]:
            reasons.append(f"Length Alert: Visible aspect ratio ({aspect_ratio:.1f}) below minimum ({self.big_aircraft_reqs['min_aspect_ratio']}). Strip too short!")
            status = "INFEASIBLE"
            color = (0, 0, 255)

        # 5. Heading Alignment Check
        if abs(heading_error_deg) > self.big_aircraft_reqs["max_heading_error_deg"]:
            if status == "FEASIBLE":
                status = "CAUTION: ALIGNMENT"
                color = (0, 200, 255)  # Amber
            reasons.append(f"Crosswind crab: Heading error ({abs(heading_error_deg):.1f} deg) exceeds limits (+/-{self.big_aircraft_reqs['max_heading_error_deg']} deg).")

        if len(reasons) == 0:
            reasons.append(f"Approved: Paved {surface}, certified markings confirmed, nominal alignment ({abs(heading_error_deg):.1f} deg).")

        return {
            "status": status,
            "color": color,
            "reasons": reasons
        }

    def evaluate_rc_plane(self, detection_res, aspect_ratio, heading_error_deg):
        """Evaluates feasibility for RC plane / fixed-wing UAV."""
        surface = detection_res["surface_type"]
        has_obstacles = detection_res["has_obstacles"]

        reasons = []
        status = "FEASIBLE"
        color = (0, 255, 0)

        # 1. Obstacle Check
        if has_obstacles:
            return {
                "status": "ABORT / HAZARD",
                "color": (0, 0, 255),
                "reasons": ["Obstacle alert: Path blocked by hazard on strip! Initiating go-around."]
            }

        # 2. Surface Check
        if surface not in self.rc_plane_reqs["allowed_surfaces"]:
            reasons.append(f"Terrain alert: Rough surface '{surface}' unsafe for landing gear.")
            status = "INFEASIBLE"
            color = (0, 0, 255)

        # 3. Aspect Ratio Check
        if aspect_ratio < self.rc_plane_reqs["min_aspect_ratio"]:
            reasons.append(f"Corridor too short: Aspect ratio ({aspect_ratio:.1f}) < minimum ({self.rc_plane_reqs['min_aspect_ratio']}).")
            status = "INFEASIBLE"
            color = (0, 0, 255)

        # 4. Heading Alignment Check
        if abs(heading_error_deg) > self.rc_plane_reqs["max_heading_error_deg"]:
            if status == "FEASIBLE":
                status = "CAUTION: HEADING"
                color = (0, 200, 255)
            reasons.append(f"Heading offset: Alignment error ({abs(heading_error_deg):.1f} deg) requires bank turn.")

        if len(reasons) == 0:
            reasons.append(f"Approved: Clean {surface} strip, clear corridor, alignment within limits ({abs(heading_error_deg):.1f} deg).")

        return {
            "status": status,
            "color": color,
            "reasons": reasons
        }

    def evaluate_all(self, detection_res, aspect_ratio, heading_error_deg):
        """Returns dual feasibility evaluation for both aircraft classes."""
        big_eval = self.evaluate_big_aircraft(detection_res, aspect_ratio, heading_error_deg)
        rc_eval = self.evaluate_rc_plane(detection_res, aspect_ratio, heading_error_deg)
        return {
            "big_aircraft": big_eval,
            "rc_plane": rc_eval
        }
