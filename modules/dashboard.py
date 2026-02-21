"""Create and draw dashboard report panels."""
import cv2
import numpy as np


class Dashboard:
    """Holds all output panels and composition logic."""

    def __init__(self) -> None:
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.reset()

    def reset(self) -> None:
        """Allocate empty panels for the current frame."""
        self.blink_report = np.zeros((240, 427, 3), np.uint8)
        self.pose_report = np.zeros((240, 427, 3), np.uint8)
        self.hand_report = np.zeros((240, 427, 3), np.uint8)
        self.eye_pose_gray_report = np.zeros((120, 427, 3), np.uint8)
        self.eye_pose_mask_report = np.zeros((120, 427, 3), np.uint8)
        self.bpm_counter_frame = np.zeros((120, 427, 3), np.uint8)

    def compose(self, frame, fps: float):
        """Build the final dashboard output frame."""
        display_frame = cv2.resize(frame, None, fx=2 / 3, fy=2 / 3)
        cv2.putText(display_frame, f"FPS : {int(fps)}", (100, 50), self.font, 0.75, (0, 0, 255), 2)

        output = np.zeros((720, 1280, 3), np.uint8)
        output[0:480, 0:853] = display_frame
        output[0:240, 853:] = self.blink_report
        output[240:480, 853:] = self.pose_report
        output[480:720, 853:] = self.hand_report
        output[480:600, 0:426] = self.eye_pose_gray_report
        output[600:, 0:426] = self.eye_pose_mask_report
        output[480:720, 426:853] = self.bpm_counter_frame
        return output
