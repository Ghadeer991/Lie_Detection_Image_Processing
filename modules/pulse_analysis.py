"""Remote photoplethysmography BPM estimation using Eulerian magnification."""
import cv2
import numpy as np


class PulseAnalyzer:
    """Maintains temporal buffers and returns BPM overlay frame."""

    def __init__(self, pulse_config):
        self.cfg = pulse_config
        self.frequencies = (
            (1.0 * self.cfg.video_frame_rate)
            * np.arange(self.cfg.buffer_size)
            / (1.0 * self.cfg.buffer_size)
        )
        self.mask = (self.frequencies >= self.cfg.min_frequency) & (self.frequencies <= self.cfg.max_frequency)
        self.bpm_buffer = np.zeros((self.cfg.bpm_buffer_size))
        self.bpm_buffer_index = 0
        self.buffer_index = 0
        self.iteration = 0

        first_frame = np.zeros((self.cfg.video_height, self.cfg.video_width, self.cfg.video_channels))
        first_gauss = self.build_gauss(first_frame, self.cfg.levels + 1)[self.cfg.levels]
        self.video_gauss = np.zeros(
            (
                self.cfg.buffer_size,
                first_gauss.shape[0],
                first_gauss.shape[1],
                self.cfg.video_channels,
            )
        )
        self.fourier_transform_avg = np.zeros((self.cfg.buffer_size))

    @staticmethod
    def build_gauss(frame, levels):
        """Build Gaussian pyramid with the requested level count."""
        pyramid = [frame]
        for _ in range(levels):
            frame = cv2.pyrDown(frame)
            pyramid.append(frame)
        return pyramid

    def reconstruct_frame(self, pyramid, index, levels):
        """Reconstruct filtered pyramid frame to ROI size."""
        filtered_frame = pyramid[index]
        for _ in range(levels):
            filtered_frame = cv2.pyrUp(filtered_frame)
        return filtered_frame[: self.cfg.video_height, : self.cfg.video_width]

    def update(self, source_frame, panel_frame, font):
        """Process frame ROI and draw BPM status on panel frame."""
        work = cv2.resize(source_frame, (self.cfg.real_width, self.cfg.real_height))
        detection = work[
            self.cfg.video_height // 2 : self.cfg.real_height - self.cfg.video_height // 2,
            self.cfg.video_width // 2 : self.cfg.real_width - self.cfg.video_width // 2,
            :,
        ]

        self.video_gauss[self.buffer_index] = self.build_gauss(detection, self.cfg.levels + 1)[self.cfg.levels]
        fourier_transform = np.fft.fft(self.video_gauss, axis=0)
        fourier_transform[self.mask == False] = 0

        if self.buffer_index % self.cfg.bpm_calc_frequency == 0:
            self.iteration += 1
            for buf in range(self.cfg.buffer_size):
                self.fourier_transform_avg[buf] = np.real(fourier_transform[buf]).mean()
            hz = self.frequencies[np.argmax(self.fourier_transform_avg)]
            bpm = 60.0 * hz
            self.bpm_buffer[self.bpm_buffer_index] = bpm
            self.bpm_buffer_index = (self.bpm_buffer_index + 1) % self.cfg.bpm_buffer_size

        filtered = np.real(np.fft.ifft(fourier_transform, axis=0))
        filtered *= self.cfg.alpha

        filtered_frame = self.reconstruct_frame(filtered, self.buffer_index, self.cfg.levels)
        output_frame = cv2.convertScaleAbs(detection + filtered_frame)

        self.buffer_index = (self.buffer_index + 1) % self.cfg.buffer_size
        work[
            self.cfg.video_height // 2 : self.cfg.real_height - self.cfg.video_height // 2,
            self.cfg.video_width // 2 : self.cfg.real_width - self.cfg.video_width // 2,
            :,
        ] = output_frame

        panel = cv2.resize(work, (427, 240))
        if self.iteration > self.cfg.bpm_buffer_size:
            bpm_mean = float(self.bpm_buffer.mean())
            cv2.putText(panel, f"BPM: {int(bpm_mean)}", (20, 30), font, 1, (0, 0, 250), 2)
            if bpm_mean >= 90:
                cv2.circle(panel, (350, 30), 15, (0, 0, 255), -1)
            else:
                cv2.circle(panel, (350, 30), 10, (100, 255, 0), -1)
        else:
            cv2.putText(panel, "Calculating BPM...", (20, 30), font, 1, (200, 0, 250), 2)

        panel_frame[:] = panel
