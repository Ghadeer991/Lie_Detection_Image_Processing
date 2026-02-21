"""Face landmarks, blink rate, pose, hand distance, and eye pose logic."""
import math
import cv2
import numpy as np

from .config import (
    LEFT_EYE_INTEREST_POINTS,
    LEFT_EYE_POINTS,
    RIGHT_EYE_INTEREST_POINTS,
    RIGHT_EYE_POINTS,
    HEAD_POINTS,
    HAND_FACE_DISTANCE_THRESHOLD,
)


def euclidean_distance(point_a, point_b):
    """Calculate 2D Euclidean distance between two points."""
    x1, y1 = point_a
    x2, y2 = point_b
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


class FaceAnalyzer:
    """Stateful analyzer for blink, pose, hand-face, and eye-gaze indicators."""

    def __init__(self, blink_config, eye_config, duration_seconds: float):
        self.blink_config = blink_config
        self.eye_config = eye_config
        self.duration_seconds = max(duration_seconds, 1.0)
        self.frame_counter = 0
        self.blink_counter = 0
        self.blink_freq = 0.0
        self.front_face = True

    def detect_landmarks(self, frame, mesh_results, pose_report):
        """Extract pixel face landmarks from MediaPipe output."""
        face_landmarks = []
        if mesh_results.multi_face_landmarks:
            for face in mesh_results.multi_face_landmarks:
                for lm in face.landmark:
                    x = int(lm.x * frame.shape[1])
                    y = int(lm.y * frame.shape[0])
                    face_landmarks.append([x, y])
                    cv2.circle(pose_report, (x // 3, y // 3), 1, (200, 255, 200), 1)
        return face_landmarks

    def _eye_landmarks(self, frame, mesh_results, eye_points, blink_report):
        """Extract landmarks for one eye."""
        eye_landmarks = []
        if mesh_results.multi_face_landmarks:
            for face in mesh_results.multi_face_landmarks:
                for idx, lm in enumerate(face.landmark):
                    if idx in eye_points:
                        x = int(lm.x * frame.shape[1])
                        y = int(lm.y * frame.shape[0])
                        eye_landmarks.append([x, y])
                        cv2.circle(blink_report, (x // 3, y // 3), 2, (200, 255, 0), 1)

        if len(eye_landmarks) >= 4:
            cv2.line(
                blink_report,
                (eye_landmarks[1][0] // 3, eye_landmarks[1][1] // 3),
                (eye_landmarks[0][0] // 3, eye_landmarks[0][1] // 3),
                (100, 50, 200),
                1,
            )
            cv2.line(
                blink_report,
                (eye_landmarks[2][0] // 3, eye_landmarks[2][1] // 3),
                (eye_landmarks[2][0] // 3, eye_landmarks[3][1] // 3),
                (100, 50, 0),
                1,
            )

        return eye_landmarks

    @staticmethod
    def _blink_ratio(right_eye_landmarks, left_eye_landmarks):
        """Compute average eye aspect ratio from both eyes."""
        right_horizontal = euclidean_distance(right_eye_landmarks[0], right_eye_landmarks[1])
        right_vertical = euclidean_distance(right_eye_landmarks[2], right_eye_landmarks[3])
        left_horizontal = euclidean_distance(left_eye_landmarks[0], left_eye_landmarks[1])
        left_vertical = euclidean_distance(left_eye_landmarks[2], left_eye_landmarks[3])

        right_ratio = right_horizontal / max(right_vertical, 1e-6)
        left_ratio = left_horizontal / max(left_vertical, 1e-6)
        return (right_ratio + left_ratio) / 2

    def analyze_blinks(self, frame, mesh_results, blink_report, font):
        """Update blink counters and draw blink report."""
        right = self._eye_landmarks(frame, mesh_results, RIGHT_EYE_INTEREST_POINTS, blink_report)
        left = self._eye_landmarks(frame, mesh_results, LEFT_EYE_INTEREST_POINTS, blink_report)
        if len(right) < 4 or len(left) < 4:
            return

        ear = self._blink_ratio(right, left)
        cv2.putText(blink_report, f"EAR: {ear:.2f}", (20, 60), font, 1, (0, 255, 0), 2)

        if ear > self.blink_config.ratio_threshold:
            self.frame_counter += 1
        else:
            if self.frame_counter >= self.blink_config.blink_frames:
                self.blink_counter += 1
            self.frame_counter = 0

        self.blink_freq = 60 * self.blink_counter / self.duration_seconds
        cv2.putText(blink_report, f"Blinks: {self.blink_counter}", (20, 30), font, 1, (255, 255, 0), 2)
        cv2.putText(blink_report, f"Freq: {self.blink_freq:.2f}", (20, 90), font, 1, (255, 100, 100), 2)
        if self.blink_freq >= 20:
            cv2.circle(blink_report, (350, 30), 15, (0, 0, 255), -1)
        else:
            cv2.circle(blink_report, (350, 30), 10, (100, 255, 0), -1)

    def analyze_head_pose(self, frame, mesh_results, pose_report, font):
        """Estimate head orientation and draw pose status."""
        face_2d = []
        face_3d = []

        if not mesh_results.multi_face_landmarks:
            return

        for face in mesh_results.multi_face_landmarks:
            for idx, lm in enumerate(face.landmark):
                if idx in HEAD_POINTS:
                    x = int(lm.x * frame.shape[1])
                    y = int(lm.y * frame.shape[0])
                    cv2.circle(pose_report, (x // 3, y // 3), 4, (200, 50, 255), -1)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

        if len(face_2d) < len(HEAD_POINTS):
            return

        face_2d = np.array(face_2d, np.float64)
        face_3d = np.array(face_3d, np.float64)
        focal_length = frame.shape[1]
        camera_matrix = np.array(
            [[focal_length, 0, frame.shape[0] / 2], [0, focal_length, frame.shape[1] / 2], [0, 0, 1]],
            dtype=np.float64,
        )
        dist_matrix = np.zeros((4, 1), np.float64)

        success, rotation_vector, _ = cv2.solvePnP(face_3d, face_2d, camera_matrix, dist_matrix)
        if not success:
            return

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        angles = cv2.RQDecomp3x3(rotation_matrix)[0]
        x_angle = angles[0] * 360
        y_angle = angles[1] * 360

        if y_angle < -5:
            text = "Looking Right"
            self.front_face = False
            cv2.circle(pose_report, (350, 30), 15, (0, 0, 255), -1)
        elif y_angle > 5:
            text = "Looking Left"
            self.front_face = False
            cv2.circle(pose_report, (350, 30), 15, (0, 0, 255), -1)
        elif x_angle < 2:
            text = "Looking Down"
            self.front_face = False
            cv2.circle(pose_report, (350, 30), 15, (0, 0, 255), -1)
        elif x_angle > 6:
            text = "Looking Up"
            self.front_face = False
            cv2.circle(pose_report, (350, 30), 15, (0, 0, 255), -1)
        else:
            text = "Forward"
            self.front_face = True
            cv2.circle(pose_report, (350, 30), 10, (100, 255, 0), -1)

        cv2.putText(pose_report, text, (20, 50), font, 1, (100, 0, 200), 2)

    def analyze_hand_face_distance(self, frame, holistic_results, hand_report, font):
        """Track right-hand distance from face and draw status."""
        if holistic_results.face_landmarks is None:
            return

        x_face = int(holistic_results.face_landmarks.landmark[4].x * frame.shape[1])
        y_face = int(holistic_results.face_landmarks.landmark[4].y * frame.shape[0])
        cv2.circle(hand_report, (x_face // 3, y_face // 3), 4, (255, 0, 0), -1)

        right_hand = holistic_results.right_hand_landmarks
        if right_hand is None:
            cv2.putText(hand_report, "No Hand", (20, 60), font, 1, (200, 100, 20), 2)
            cv2.circle(hand_report, (350, 30), 10, (100, 255, 0), -1)
            return

        x_hand = int(right_hand.landmark[8].x * frame.shape[1])
        y_hand = int(right_hand.landmark[8].y * frame.shape[0])
        distance = math.sqrt((x_hand - x_face) ** 2 + (y_hand - y_face) ** 2)

        cv2.putText(hand_report, f"R-Distance: {distance:.2f}", (20, 30), font, 1, (200, 100, 20), 2)
        cv2.circle(hand_report, (x_hand // 3, y_hand // 3), 4, (0, 255, 0), -1)

        if distance <= HAND_FACE_DISTANCE_THRESHOLD:
            cv2.putText(hand_report, "Hand On Face", (20, 60), font, 1, (200, 100, 20), 2)
            cv2.circle(hand_report, (350, 30), 15, (0, 0, 255), -1)
        else:
            cv2.putText(hand_report, "Hand Out Of Face", (20, 60), font, 1, (200, 100, 20), 2)
            cv2.circle(hand_report, (350, 30), 10, (100, 255, 0), -1)

    @staticmethod
    def _extract_eye(frame, eye_landmarks):
        """Crop right-eye grayscale image from polygon landmarks."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros(gray.shape, np.uint8)
        cv2.fillPoly(mask, [np.array(eye_landmarks, np.int32)], 255)
        eye = cv2.bitwise_and(gray, gray, mask=mask)

        max_x = max(eye_landmarks, key=lambda p: p[0])[0]
        min_x = min(eye_landmarks, key=lambda p: p[0])[0]
        max_y = max(eye_landmarks, key=lambda p: p[1])[1]
        min_y = min(eye_landmarks, key=lambda p: p[1])[1]

        cropped = eye[min_y:max_y, min_x:max_x]
        if cropped.size == 0:
            return None
        return cv2.resize(cropped, (426, 120))

    def _estimate_eye_position(self, cropped_eye):
        """Estimate gaze direction from thresholded iris region."""
        height, width = cropped_eye.shape
        blur = cv2.GaussianBlur(cropped_eye, (15, 15), 0)
        _, threshold_eye = cv2.threshold(blur, self.eye_config.iris_threshold, 255, cv2.THRESH_BINARY)

        part = width // 3
        right_piece = threshold_eye[:, 0:part]
        center_piece = threshold_eye[:, part : 2 * part]
        left_piece = threshold_eye[:, 2 * part : width]

        right_part = np.sum(right_piece == 0)
        center_part = np.sum(center_piece == 0)
        left_part = np.sum(left_piece == 0)
        eye_parts = [right_part, center_part, left_part]
        max_index = int(np.argmax(eye_parts))

        if not self.front_face:
            return cv2.resize(threshold_eye, (426, 120)), ""

        if max_index == 0:
            return cv2.resize(threshold_eye, (426, 120)), "RIGHT"
        if max_index == 1:
            return cv2.resize(threshold_eye, (426, 120)), "CENTER"
        if max_index == 2:
            return cv2.resize(threshold_eye, (426, 120)), "LEFT"
        return cv2.resize(threshold_eye, (426, 120)), "Closed"

    def analyze_eye_pose(self, frame, face_landmarks, dashboard, font):
        """Analyze eye pose and update eye reports."""
        if len(face_landmarks) <= max(RIGHT_EYE_POINTS):
            return

        right_eye_landmarks = [face_landmarks[p] for p in RIGHT_EYE_POINTS]
        cropped_eye = self._extract_eye(frame, right_eye_landmarks)
        if cropped_eye is None:
            return

        threshold_eye, eye_position = self._estimate_eye_position(cropped_eye)

        dashboard.eye_pose_gray_report = cv2.cvtColor(cropped_eye, cv2.COLOR_GRAY2BGR)
        dashboard.eye_pose_mask_report = cv2.cvtColor(threshold_eye, cv2.COLOR_GRAY2BGR)
        cv2.putText(dashboard.eye_pose_mask_report, eye_position, (10, 30), font, 1, (0, 150, 200), 2)

        if eye_position in ("RIGHT", "LEFT"):
            cv2.circle(dashboard.eye_pose_mask_report, (350, 30), 15, (0, 0, 255), -1)
        else:
            cv2.circle(dashboard.eye_pose_mask_report, (350, 30), 10, (100, 255, 0), -1)
