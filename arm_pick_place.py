"""
A simple planar (2D) robotic arm with two links, simulated moving through
a pick-and-place sequence: HOME -> PICK -> PLACE -> HOME.

This demonstrates:
  - Forward kinematics  (angles -> tip position)
  - Inverse kinematics   (target position -> angles)
  - A basic state machine (sense -> decide -> act loop)
  - Animated visualization with matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

L1 = 3.0   # length of link 1 
L2 = 2.5   # length of link 2 


def forward_kinematics(theta1, theta2):
    
    elbow_x = L1 * np.cos(theta1)
    elbow_y = L1 * np.sin(theta1)

    tip_x = elbow_x + L2 * np.cos(theta1 + theta2)
    tip_y = elbow_y + L2 * np.sin(theta1 + theta2)

    return (elbow_x, elbow_y), (tip_x, tip_y)


def inverse_kinematics(x, y, elbow_up=True):
    
    d = np.sqrt(x**2 + y**2)  

    if d > (L1 + L2) or d < abs(L1 - L2):
        raise ValueError(f"Target ({x}, {y}) is out of reach for this arm.")


    cos_theta2 = (d**2 - L1**2 - L2**2) / (2 * L1 * L2)
    cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)  
    theta2 = np.arccos(cos_theta2)

    if elbow_up:
        theta2 = -theta2  
    k1 = L1 + L2 * np.cos(theta2)
    k2 = L2 * np.sin(theta2)
    theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)

    return theta1, theta2


HOME = (4.0, 1.0)
PICK = (2.0, 3.5)
PLACE = (-3.0, 2.0)

STATES = ["HOME", "TO_PICK", "GRIP_CLOSE", "TO_PLACE", "GRIP_OPEN", "TO_HOME"]

theta_home = inverse_kinematics(*HOME)
theta_pick = inverse_kinematics(*PICK)
theta_place = inverse_kinematics(*PLACE)

sequence = [
    (theta_home,  False, 5),   # start at home, gripper open
    (theta_pick,  False, 20),  # move to pick location
    (theta_pick,  True,  10),  # close gripper 
    (theta_place, True,  20),  # move to place location, holding object
    (theta_place, False, 10),  # open gripper, release
    (theta_home,  False, 20),  # return home
]


def interpolate_sequence(sequence, steps_per_leg=20):
    
    frames = []
    current = sequence[0][0]
    for target, gripper_closed, hold in sequence:
        for t in np.linspace(0, 1, hold if hold > 0 else steps_per_leg):
            th1 = current[0] + (target[0] - current[0]) * t
            th2 = current[1] + (target[1] - current[1]) * t
            frames.append((th1, th2, gripper_closed))
        current = target
    return frames


frames = interpolate_sequence(sequence)

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-6, 6)
ax.set_ylim(-1, 6)
ax.set_aspect("equal")
ax.grid(True, linestyle="--", alpha=0.4)
ax.set_title("2-Link Arm — Pick and Place")


for label, (px, py) in [("HOME", HOME), ("PICK", PICK), ("PLACE", PLACE)]:
    ax.plot(px, py, "kx")
    ax.annotate(label, (px, py), textcoords="offset points", xytext=(5, 5))

(arm_line,) = ax.plot([], [], "o-", lw=4, color="steelblue", markersize=8)
(gripper_marker,) = ax.plot([], [], "o", markersize=14, color="green")


def update(frame_idx):
    theta1, theta2, gripper_closed = frames[frame_idx]
    (elbow_x, elbow_y), (tip_x, tip_y) = forward_kinematics(theta1, theta2)

    xs = [0, elbow_x, tip_x]
    ys = [0, elbow_y, tip_y]
    arm_line.set_data(xs, ys)

    gripper_marker.set_data([tip_x], [tip_y])
    gripper_marker.set_color("red" if gripper_closed else "green")

    return arm_line, gripper_marker


ani = animation.FuncAnimation(
    fig, update, frames=len(frames), interval=40, blit=False, repeat=True
)

if __name__ == "__main__":
    plt.show()