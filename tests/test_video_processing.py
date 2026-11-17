import unittest
from unittest.mock import patch, MagicMock
import cv2
from movie_barcodes import video_processing


class TestVideoProcessing(unittest.TestCase):
    """
    Test the video processing functions.
    """

    def setUp(self) -> None:
        """
        Set up the test case.
        :return: None
        """
        self.video_path = "dummy_video.mp4"
        self.start_frame = 0
        self.end_frame = 49
        self.target_frames = 5

    @staticmethod
    def mock_color_extractor(frame):
        """
        Mock color extractor function.
        :param frame:
        :return: frame
        """
        return frame

    @patch("movie_barcodes.video_processing.cv2.VideoCapture")
    def test_load_video_raises_error_on_file_not_open(self, mock_video: MagicMock) -> None:
        """
        Test load_video raises ValueError when the video file cannot be opened.
        :param mock_video: MagicMock object for cv2.VideoCapture
        :return: None
        """
        mock_video.return_value.isOpened.return_value = False
        with self.assertRaises(ValueError) as context:
            video_processing.load_video(self.video_path)
        self.assertIn(
            "Could not open the video file: " + self.video_path,
            str(context.exception),
        )

    @patch("movie_barcodes.video_processing.cv2.VideoCapture")
    def test_load_video_raises_error_on_no_frames(self, mock_video: MagicMock) -> None:
        """
        Test load_video raises ValueError when the video has no frames.
        :param mock_video: MagicMock object for cv2.VideoCapture
        :return: None
        """
        mock_video.return_value.isOpened.return_value = True
        mock_video.return_value.get.return_value = 0  # Mocking frame count as 0
        with self.assertRaises(ValueError) as context:
            video_processing.load_video(self.video_path)
        self.assertIn(
            "The video file " + self.video_path + " has no frames.",
            str(context.exception),
        )

    @patch("movie_barcodes.video_processing.cv2.VideoCapture")
    def test_load_video_raises_error_on_invalid_dimensions(self, mock_video: MagicMock) -> None:
        """
        Test load_video raises ValueError when video has invalid dimensions.
        :param mock_video: MagicMock object for cv2.VideoCapture
        :return: None
        """
        mock_video.return_value.isOpened.return_value = True
        mock_video.return_value.get.side_effect = [
            1,
            0,
            0,
        ]  # Mocking frame count as 1 and dimensions as 0
        with self.assertRaises(ValueError) as context:
            video_processing.load_video(self.video_path)
        self.assertIn(
            "The video file " + self.video_path + " has invalid dimensions.",
            str(context.exception),
        )

    @patch("movie_barcodes.video_processing.cv2.VideoCapture")
    def test_load_video_success(self, mock_video: MagicMock) -> None:
        """
        Test load_video successfully returns video properties.
        :param mock_video: MagicMock object for cv2.VideoCapture
        :return: None
        """
        mock_video.return_value.isOpened.return_value = True
        mock_video.return_value.get.side_effect = [
            100,
            1920,
            1080,
        ]  # Mocking frame count, width, and height
        video, frame_count, frame_width, frame_height = video_processing.load_video(self.video_path)
        self.assertEqual(frame_count, 100)
        self.assertEqual(frame_width, 1920)
        self.assertEqual(frame_height, 1080)

    @patch("cv2.VideoCapture")
    def test_extract_colors(self, mock_video: MagicMock) -> None:
        """
        Test extract_colors returns the expected colors.
        :param mock_video: MagicMock object for cv2.VideoCapture
        :return: None
        """
        # Mock the read method of VideoCapture
        mock_video_instance = mock_video.return_value
        mock_video_instance.read.side_effect = [(True, "frame_data")] * self.target_frames  # Simulate frames

        # Mock color_extractor to return a predictable color
        def special_color_extractor(frame):
            return "red" if frame == "frame_data" else "unknown"

        # Expected result
        expected_colors = ["red"] * self.target_frames

        # Run the test
        actual_colors = video_processing.extract_colors(
            self.video_path,
            self.start_frame,
            self.end_frame,
            special_color_extractor,
            self.target_frames,
        )

        # Verify the result
        self.assertEqual(actual_colors, expected_colors)

    @patch("cv2.VideoCapture")
    def test_frame_skipping_logic(self, mock_video: MagicMock) -> None:
        """
        Test that the frame skipping logic is working as expected.
        :param mock_video: MagicMock object for cv2.VideoCapture
        :return: None
        """
        mock_video_instance = mock_video.return_value

        # Initialize a counter to keep track of the current frame position
        current_frame = [0]  # Use a list so it can be modified within the side_effect functions

        # Define a side effect for the read method
        def read_side_effect():
            if current_frame[0] < self.end_frame:
                frame_data = (True, f"frame_data_{current_frame[0]}")
                current_frame[0] += 1
                return frame_data
            else:
                return False, None  # Simulate end of video

        # Define a side effect for the grab method to simply increment the counter
        def grab_side_effect():
            if current_frame[0] < self.end_frame:
                current_frame[0] += 1
            return True

        mock_video_instance.read.side_effect = read_side_effect
        mock_video_instance.grab.side_effect = grab_side_effect

        expected_colors = [f"frame_data_{i * 10}" for i in range(self.target_frames)]

        actual_colors = video_processing.extract_colors(
            self.video_path,
            self.start_frame,
            self.end_frame,
            self.mock_color_extractor,
            self.target_frames,
        )

        self.assertEqual(actual_colors, expected_colors)

    @patch("cv2.VideoCapture")
    def test_frame_skip_default(self, mock_video: MagicMock) -> None:
        """
        Test that the default frame_skip is 1.
        :param mock_video: MagicMock object for cv2.VideoCapture
        :return: None
        """
        # Setup mock to simulate frames in the video
        mock_video_instance = mock_video.return_value
        mock_video_instance.read.side_effect = [(True, f"frame_{i}") for i in range(self.end_frame)] + [(False, None)]

        # Call without specifying target_frames
        colors = video_processing.extract_colors(
            self.video_path, self.start_frame, self.end_frame, self.mock_color_extractor
        )

        # Assert frame_skip was effectively 1 by checking all frames were processed
        self.assertEqual(len(colors), self.end_frame)

    @patch("cv2.VideoCapture")
    def test_frame_skip_with_target_frames(self, mock_video: MagicMock) -> None:
        """
        Test that target_frames is respected when extracting colors.
        :param mock_video: MagicMock object for cv2.VideoCapture
        :return: None
        """
        # Setup mock to simulate frames in the video
        mock_video_instance = mock_video.return_value
        mock_video_instance.read.side_effect = [(True, f"frame_{i}") for i in range(self.end_frame)] + [(False, None)]

        colors = video_processing.extract_colors(
            self.video_path,
            self.start_frame,
            self.end_frame,
            self.mock_color_extractor,
            self.target_frames,
        )

        # Assert that we only processed the specified number of target_frames
        self.assertEqual(len(colors), self.target_frames)

    @patch("cv2.VideoCapture")
    def test_len_colors_equals_len_video_no_target_frames(self, mock_video: MagicMock) -> None:
        """
        Test that the number of colors extracted is equal to the number of frames in the video
         when target_frames is not given.
        :param mock_video: MagicMock object for cv2.VideoCapture
        :return: None
        """
        # Setup mock to simulate frames in the video
        mock_video_instance = mock_video.return_value
        mock_video_instance.read.side_effect = [(True, f"frame_{i}") for i in range(self.end_frame)] + [(False, None)]

        colors = video_processing.extract_colors(
            self.video_path, self.start_frame, self.end_frame, self.mock_color_extractor
        )

        # Assert that all frames were processed since target_frames was not given
        self.assertEqual(len(colors), self.end_frame)

    @patch("cv2.VideoCapture")
    def test_samples_spread_until_end_of_range(self, mock_video: MagicMock) -> None:
        """
        Samples are spread over the whole range: with 100 frames and 40 samples, an integer step of 2
        would stop at frame 78 and ignore the last 20% of the range.
        """
        mock_video_instance = mock_video.return_value
        current_frame = [0]

        def read_side_effect():
            frame = current_frame[0]
            current_frame[0] += 1
            return (True, frame) if frame < 100 else (False, None)

        def grab_side_effect():
            current_frame[0] += 1
            return True

        mock_video_instance.read.side_effect = read_side_effect
        mock_video_instance.grab.side_effect = grab_side_effect

        colors = video_processing.extract_colors(self.video_path, 0, 99, self.mock_color_extractor, 40)

        self.assertEqual(len(colors), 40)
        self.assertEqual(colors[0], 0)
        self.assertGreaterEqual(colors[-1], 97)
        self.assertEqual(colors, sorted(set(colors)))

    @staticmethod
    def _fake_capture(mock_video: MagicMock, real_frames: int, keyframe_interval: int = 1, overshoot: int = 0) -> list:
        """
        Make the mocked capture behave like a video of real_frames frames whose frames are their own index.
        A seek lands on the previous keyframe (plus overshoot) and get(POS_FRAMES) reports where it landed.
        Returns the list of seek targets.
        """
        instance = mock_video.return_value
        current = [0]
        seeks = []

        def set_side_effect(prop, value):
            seeks.append(value)
            current[0] = min(real_frames, value - value % keyframe_interval + overshoot)
            return True

        def get_side_effect(prop):
            return float(current[0])

        def grab_side_effect():
            if current[0] >= real_frames:
                return False
            current[0] += 1
            return True

        def read_side_effect():
            if current[0] >= real_frames:
                return False, None
            current[0] += 1
            return True, current[0] - 1

        instance.set.side_effect = set_side_effect
        instance.get.side_effect = get_side_effect
        instance.grab.side_effect = grab_side_effect
        instance.read.side_effect = read_side_effect
        return seeks

    @patch("cv2.VideoCapture")
    def test_seek_lands_on_earlier_keyframe(self, mock_video: MagicMock) -> None:
        """
        When the backend lands on an earlier keyframe, the worker grabs forward to its exact start frame.
        """
        self._fake_capture(mock_video, real_frames=1000, keyframe_interval=100)

        colors = video_processing.extract_colors(self.video_path, 537, 636, self.mock_color_extractor, 10)

        self.assertEqual(colors[0], 537)
        self.assertEqual(colors, [537 + i * 10 for i in range(10)])

    @patch("cv2.VideoCapture")
    def test_seek_overshoot_backs_off(self, mock_video: MagicMock) -> None:
        """
        When the backend lands past the requested frame, the worker seeks earlier and grabs forward.
        """
        seeks = self._fake_capture(mock_video, real_frames=1000, keyframe_interval=100, overshoot=33)

        colors = video_processing.extract_colors(self.video_path, 520, 529, self.mock_color_extractor)

        self.assertEqual(colors, list(range(520, 530)))
        self.assertEqual(seeks, [520, 520 - video_processing.SEEK_FROM_START_FRAMES])

    @patch("cv2.VideoCapture")
    def test_no_seek_near_start(self, mock_video: MagicMock) -> None:
        """
        Near the start of the video, the worker does not seek (unreliable on some streams) and grabs instead.
        """
        seeks = self._fake_capture(mock_video, real_frames=1000, overshoot=33)

        colors = video_processing.extract_colors(self.video_path, 5, 14, self.mock_color_extractor)

        self.assertEqual(seeks, [])
        self.assertEqual(colors, list(range(5, 15)))

    @patch("cv2.VideoCapture")
    def test_start_beyond_real_end_returns_empty(self, mock_video: MagicMock) -> None:
        """
        With an over-reported frame count, a worker whose range starts past the real end returns no colors.
        """
        self._fake_capture(mock_video, real_frames=300)

        colors = video_processing.extract_colors(self.video_path, 320, 392, self.mock_color_extractor, 10)

        self.assertEqual(colors, [])
        mock_video.return_value.release.assert_called_once()

    @patch("cv2.VideoCapture")
    def test_range_crossing_real_end_stops_at_end(self, mock_video: MagicMock) -> None:
        """
        With an over-reported frame count, a worker whose range crosses the real end keeps the frames before it.
        """
        self._fake_capture(mock_video, real_frames=300)

        colors = video_processing.extract_colors(self.video_path, 290, 392, self.mock_color_extractor)

        self.assertEqual(colors, list(range(290, 300)))


class FakeCapture:
    """
    Minimal stand-in for cv2.VideoCapture: read() returns the index of the frame it reads.
    `seek_error` shifts where a seek lands once frames have been read (negative: before the target,
    like a keyframe seek; positive: past it). `seek_ok=False` makes every seek fail.
    """

    def __init__(self, frame_count: int, seek_error: int = 0, seek_ok: bool = True) -> None:
        self.frame_count = frame_count
        self.seek_error = seek_error
        self.seek_ok = seek_ok
        self.pos = 0
        self.reads = 0
        self.seeks: list = []
        self.grabs = 0

    def set(self, prop: int, value: int) -> bool:
        assert prop == cv2.CAP_PROP_POS_FRAMES
        self.seeks.append(value)
        if not self.seek_ok:
            return False
        # The initial positioning on start_frame is exact; later seeks get seek_error applied
        self.pos = value + self.seek_error if self.reads else value
        return True

    def get(self, prop: int) -> float:
        assert prop == cv2.CAP_PROP_POS_FRAMES
        return float(self.pos)

    def grab(self) -> bool:
        self.grabs += 1
        self.pos += 1
        return self.pos <= self.frame_count

    def read(self) -> tuple:
        self.reads += 1
        frame = self.pos
        self.pos += 1
        return (True, frame) if frame < self.frame_count else (False, None)

    def release(self) -> None:
        pass


class TestFrameSkipping(unittest.TestCase):
    """
    Test how extract_colors moves between samples: grab() for small gaps, a seek for large ones.
    """

    # Sample spacing whose skip (spacing - 1 frames between two reads) is just over the threshold
    gap = video_processing.SEEK_THRESHOLD + 2
    # Start far enough in the stream for seeks to be allowed
    start = video_processing.SEEK_FROM_START_FRAMES

    def run_extract(self, capture: FakeCapture, start_frame: int, end_frame: int, target_frames: int) -> list:
        with patch("cv2.VideoCapture", return_value=capture):
            return video_processing.extract_colors("dummy.mp4", start_frame, end_frame, lambda f: f, target_frames)

    def run_four_samples(self, capture: FakeCapture, gap: int) -> list:
        return self.run_extract(capture, self.start, self.start + gap * 4 - 1, 4)

    def test_large_gaps_use_seek(self) -> None:
        capture = FakeCapture(frame_count=self.start + self.gap * 4)
        colors = self.run_four_samples(capture, self.gap)
        expected = [self.start + i * self.gap for i in range(4)]
        self.assertEqual(colors, expected)
        self.assertEqual(capture.seeks, expected)  # Initial positioning, then one absolute seek per sample
        self.assertEqual(capture.grabs, 0)

    def test_small_gaps_use_grab(self) -> None:
        gap = video_processing.SEEK_THRESHOLD + 1  # Skips exactly SEEK_THRESHOLD frames
        capture = FakeCapture(frame_count=self.start + gap * 4)
        colors = self.run_four_samples(capture, gap)
        self.assertEqual(colors, [self.start + i * gap for i in range(4)])
        self.assertEqual(capture.seeks, [self.start])  # Only the initial positioning
        self.assertEqual(capture.grabs, 3 * (gap - 1))

    def test_no_seek_near_start_of_stream(self) -> None:
        # Seeks are unreliable near the start of some streams: targets below SEEK_FROM_START_FRAMES are grabbed
        frame_count = video_processing.SEEK_FROM_START_FRAMES + self.gap * 2
        targets = list(range(0, frame_count, self.gap))
        capture = FakeCapture(frame_count=frame_count)
        colors = self.run_extract(capture, 0, len(targets) * self.gap - 1, len(targets))
        self.assertEqual(colors, targets)
        self.assertEqual(capture.seeks, [t for t in targets if t >= video_processing.SEEK_FROM_START_FRAMES])

    def test_seek_landing_before_target_is_corrected_with_grabs(self) -> None:
        capture = FakeCapture(frame_count=self.start + self.gap * 4, seek_error=-5)
        colors = self.run_four_samples(capture, self.gap)
        self.assertEqual(colors, [self.start + i * self.gap for i in range(4)])
        self.assertEqual(capture.grabs, 3 * 5)

    def test_seek_overshoot_keeps_order_and_count(self) -> None:
        capture = FakeCapture(frame_count=self.start + self.gap * 4, seek_error=2)
        colors = self.run_four_samples(capture, self.gap)
        targets = [self.start + i * self.gap for i in range(4)]
        self.assertEqual(len(colors), 4)
        self.assertEqual(colors, sorted(set(colors)))
        for color, target in zip(colors, targets):
            self.assertIn(color - target, (0, 2))  # Exact, or the frame the overshooting seek landed on

    def test_failed_seek_falls_back_to_grab(self) -> None:
        capture = FakeCapture(frame_count=self.start + self.gap * 4, seek_ok=False)
        colors = self.run_four_samples(capture, self.gap)
        self.assertEqual(colors, [self.start + i * self.gap for i in range(4)])

    def test_seek_past_end_of_video_stops_cleanly(self) -> None:
        # The container reports more frames than it can decode: samples past the real end are dropped
        capture = FakeCapture(frame_count=self.start + self.gap * 2)
        colors = self.run_four_samples(capture, self.gap)
        self.assertEqual(colors, [self.start, self.start + self.gap])

    def test_grab_past_end_of_video_stops_cleanly(self) -> None:
        gap = video_processing.SEEK_THRESHOLD + 1
        capture = FakeCapture(frame_count=self.start + gap + 10)
        colors = self.run_four_samples(capture, gap)
        self.assertEqual(colors, [self.start, self.start + gap])
        self.assertLessEqual(capture.grabs, 3 * gap)  # Grabbing stopped at the real end


if __name__ == "__main__":
    unittest.main()
