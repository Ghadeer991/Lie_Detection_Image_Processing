"""Configuration for video input, thresholds, and UI settings."""
from dataclasses import dataclass


@dataclass
class VideoConfig:
    input_path: str = "b.mp4"
    output_path: str = "Demo.mov"
    output_size: tuple[int, int] = (1280, 720)
    output_fps: int = 10


@dataclass
class BlinkConfig:
    ratio_threshold: float = 3.8
    blink_frames: int = 2


@dataclass
class EyeConfig:
    iris_threshold: int = 60


@dataclass
class PulseConfig:
    real_width: int = 640
    real_height: int = 480
    video_width: int = 320
    video_height: int = 240
    video_channels: int = 3
    video_frame_rate: int = 15
    levels: int = 3
    alpha: int = 170
    min_frequency: float = 1.0
    max_frequency: float = 2.0
    buffer_size: int = 150
    bpm_calc_frequency: int = 15
    bpm_buffer_size: int = 5


@dataclass
class AppConfig:
    video: VideoConfig
    blink: BlinkConfig
    eye: EyeConfig
    pulse: PulseConfig


def load_config() -> AppConfig:
    """Create app configuration with default values."""
    return AppConfig(
        video=VideoConfig(),
        blink=BlinkConfig(),
        eye=EyeConfig(),
        pulse=PulseConfig(),
    )


LEFT_EYE_POINTS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_POINTS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_INTEREST_POINTS = [133, 33, 159, 145]
LEFT_EYE_INTEREST_POINTS = [263, 362, 386, 374]
HEAD_POINTS = [33, 263, 1, 61, 291, 199]
HAND_FACE_DISTANCE_THRESHOLD = 110
