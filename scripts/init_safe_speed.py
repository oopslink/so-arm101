#!/usr/bin/env python3

import argparse

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus


MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synchronize SO-101 goal positions and set a safe servo speed limit."
    )
    parser.add_argument("--port", default="/dev/ttyACM0", help="Robot serial port")
    parser.add_argument(
        "--speed",
        type=int,
        default=150,
        help="Raw STS3215 Goal_Velocity value (1-4095; 0 means unlimited)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.speed <= 4095:
        raise ValueError("--speed must be between 1 and 4095; 0 means unlimited")

    motors = {
        name: Motor(
            motor_id,
            "sts3215",
            MotorNormMode.RANGE_0_100 if name == "gripper" else MotorNormMode.DEGREES,
        )
        for motor_id, name in enumerate(MOTOR_NAMES, 1)
    }

    bus = FeetechMotorsBus(port=args.port, motors=motors)
    bus.connect()

    try:
        bus.disable_torque(num_retry=10)

        present = bus.sync_read("Present_Position", normalize=False, num_retry=10)
        bus.sync_write("Goal_Position", present, normalize=False, num_retry=10)
        # Write this register one motor at a time so each servo returns an ACK.
        # A broadcast sync-write can return successfully even when the value was
        # not accepted, especially immediately after the controller boots.
        for name in MOTOR_NAMES:
            bus.write("Goal_Velocity", name, args.speed, normalize=False, num_retry=10)

        goal = bus.sync_read("Goal_Position", normalize=False, num_retry=10)
        velocity = bus.sync_read("Goal_Velocity", normalize=False, num_retry=10)

        print("Present_Position:", present)
        print("Goal_Position:   ", goal)
        print("Goal_Velocity:   ", velocity)

        if goal != present:
            raise RuntimeError("Goal position verification failed; torque will remain disabled")
        if any(value != args.speed for value in velocity.values()):
            raise RuntimeError("Goal velocity verification failed; torque will remain disabled")

        bus.enable_torque(num_retry=10)
        print("Initialization succeeded; torque is enabled")
    finally:
        # Preserve the torque state selected above and only close the serial port.
        bus.disconnect(False)


if __name__ == "__main__":
    main()
