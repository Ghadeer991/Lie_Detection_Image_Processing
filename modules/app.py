"""Application orchestration for the lie detection dashboard."""
import cv2
import mediapipe as mp

from .config import load_config
from .dashboard import Dashboard
from .face_analysis import FaceAnalyzer
from .pulse_analysis import PulseAnalyzer


def run_app():
    """Run the complete lie-detection processing pipeline."""
    config = load_config()
    dashboard = Dashboard()

    mp_face_mesh = mp.solutions.face_mesh
    mp_holistic = mp.solutions.holistic

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5,
    )
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(config.video.input_path)
    duration_seconds = cap.get(cv2.CAP_PROP_FRAME_COUNT) // max(cap.get(cv2.CAP_PROP_FPS), 1)

    writer = cv2.VideoWriter()
    writer.open(
        config.video.output_path,
        cv2.VideoWriter_fourcc("j", "p", "e", "g"),
        config.video.output_fps,
        config.video.output_size,
        True,
    )

    face_analyzer = FaceAnalyzer(config.blink, config.eye, duration_seconds)
    pulse_analyzer = PulseAnalyzer(config.pulse)

    try:
        while True:
            dashboard.reset()
            timer = cv2.getTickCount()
            success, frame = cap.read()
            if not success:
                print("Check cap!")
                break

            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                mesh_results = face_mesh.process(rgb)
                holistic_results = holistic.process(rgb)
                rgb.flags.writeable = True

                face_landmarks = face_analyzer.detect_landmarks(frame, mesh_results, dashboard.pose_report)
                face_analyzer.analyze_blinks(frame, mesh_results, dashboard.blink_report, dashboard.font)
                face_analyzer.analyze_head_pose(frame, mesh_results, dashboard.pose_report, dashboard.font)
                face_analyzer.analyze_hand_face_distance(frame, holistic_results, dashboard.hand_report, dashboard.font)
                face_analyzer.analyze_eye_pose(frame, face_landmarks, dashboard, dashboard.font)

                pulse_analyzer.update(frame, dashboard.bpm_counter_frame, dashboard.font)
            except Exception:
                pass

            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            output = dashboard.compose(frame, fps)
            writer.write(output)
            cv2.imshow("Lie Detection", output)

            if cv2.waitKey(10) & 0xFF == 27:
                break
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
