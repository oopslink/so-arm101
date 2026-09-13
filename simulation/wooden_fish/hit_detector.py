"""One impact followed by sustained clearance; evaluated at physics frequency."""
from dataclasses import dataclass


@dataclass
class HitDetector:
    min_speed: float = 0.025
    max_speed: float = 0.7
    clearance: float = 0.025
    release_steps: int = 20  # 40 ms at 500 Hz
    count: int = 0
    released: int = 0
    touching: bool = False
    success: bool = False

    def update(self, contact: bool, in_region: bool, downward_speed: float, gap: float):
        new_hit = False
        repeated = False
        if contact and not self.touching:
            if self.count:
                repeated = True
            elif in_region and self.min_speed <= downward_speed <= self.max_speed:
                self.count = 1
                new_hit = True
        self.touching = contact
        if self.count and not contact and gap >= self.clearance:
            self.released += 1
        else:
            self.released = 0
        self.success = self.released >= self.release_steps
        return new_hit, repeated
