from typing import Dict, List, Set, Tuple
import numpy as np
import supervision as sv


class DetectionsManager:
    def __init__(self) -> None:
        self.tracker_id_to_zone_id: Dict[int, int] = {}
        self.counts: Dict[int, Dict[int, Set[int]]] = {}
        self.previous_positions: Dict[int, Tuple[float, float]] = {}  # Tracker ID to (x, y)
        self.speeds: Dict[int, float] = {}  # Tracker ID to speed

    def update_positions(self, detections: sv.Detections):
        for tracker_id, bbox in zip(detections.tracker_id, detections.xyxy):
            x_center = (bbox[0] + bbox[2]) / 2
            y_center = (bbox[1] + bbox[3]) / 2
            self.previous_positions[tracker_id] = (x_center, y_center)

    def calculate_speed(self, tracker_id, new_position, frame_rate, scale):
        if tracker_id not in self.previous_positions:
            return 0
        old_position = self.previous_positions[tracker_id]
        distance_pixels = np.sqrt((new_position[0] - old_position[0])**2 + (new_position[1] - old_position[1])**2)
        distance_real = distance_pixels * scale  # Convert pixel distance to real-world distance
        self.speeds[tracker_id] = distance_real * frame_rate
        return self.speeds[tracker_id] 

    def update(
        self,
        detections_all: sv.Detections,
        detections_in_zones: List[sv.Detections],
        detections_out_zones: List[sv.Detections],
    ) -> sv.Detections:
        for zone_in_id, detections_in_zone in enumerate(detections_in_zones):
            for tracker_id in detections_in_zone.tracker_id:
                self.tracker_id_to_zone_id.setdefault(tracker_id, zone_in_id)

        for zone_out_id, detections_out_zone in enumerate(detections_out_zones):
            for tracker_id in detections_out_zone.tracker_id:
                if tracker_id in self.tracker_id_to_zone_id:
                    zone_in_id = self.tracker_id_to_zone_id[tracker_id]
                    self.counts.setdefault(zone_out_id, {})
                    self.counts[zone_out_id].setdefault(zone_in_id, set())
                    self.counts[zone_out_id][zone_in_id].add(tracker_id)

        detections_all.class_id = np.vectorize(
            lambda x: self.tracker_id_to_zone_id.get(x, -1)
        )(detections_all.tracker_id)
        return detections_all[detections_all.class_id != -1]

