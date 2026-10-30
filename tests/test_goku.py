import unittest
from argparse import Namespace
from contextlib import ExitStack
from unittest.mock import Mock, patch

import numpy as np

from goku import DURATION, GestureLatch, forehead_pose, render_teleport, run


class GestureTests(unittest.TestCase):
    def test_hold_release_and_second_trigger(self):
        latch = GestureLatch()
        triggers = [latch.update(True, t / 10) for t in range(22)]
        self.assertEqual(sum(triggers), 1)
        for t in range(22, 27):
            self.assertFalse(latch.update(False, t / 10))
        triggers = [latch.update(True, t / 10) for t in range(27, 40)]
        self.assertEqual(sum(triggers), 1)

    def test_tracking_gap_does_not_count_as_held_pose(self):
        latch = GestureLatch()
        self.assertFalse(latch.update(True, 0))
        self.assertFalse(latch.update(True, 1.2))
        self.assertFalse(latch.update(True, 1.3))

    def test_interrupted_pose_restarts_countdown(self):
        latch = GestureLatch()
        for t in range(5):
            self.assertFalse(latch.update(True, t / 10))
        self.assertFalse(latch.update(False, 0.5))
        for t in range(6, 12):
            self.assertFalse(latch.update(True, t / 10))
        self.assertTrue(latch.update(True, 1.2))

    def points(self):
        points = np.zeros((21, 3))
        points[0, :2] = (0.5, 0.7)
        for start, x in ((5, 0.48), (9, 0.52)):
            for j, y in enumerate((0.55, 0.43, 0.34, 0.25)):
                points[start + j, :2] = (x, y)
        for start, x in ((13, 0.56), (17, 0.60)):
            for j, y in enumerate((0.57, 0.5, 0.56, 0.61)):
                points[start + j, :2] = (x, y)
        return points

    def test_forehead_pose_requires_face_and_two_close_fingers(self):
        points = self.points()
        box = (350, 110, 300, 300)
        self.assertTrue(forehead_pose(points, box, 1000, 600))
        self.assertFalse(forehead_pose(points, None, 1000, 600))
        self.assertFalse(forehead_pose(points, (0, 0, 100, 100), 1000, 600))
        points[12, 0] += 0.2
        self.assertFalse(forehead_pose(points, box, 1000, 600))

    def test_extended_ring_finger_rejected(self):
        points = self.points()
        points[16, 1] = 0.25
        self.assertFalse(forehead_pose(points, (350, 110, 300, 300), 1000, 600))


class RenderTests(unittest.TestCase):
    def test_animation_clock_starts_after_detection_and_segmentation(self):
        frame = np.zeros((24, 24, 3), dtype=np.uint8)
        camera = Mock()
        camera.read.return_value = (True, frame)
        hands, face = Mock(), Mock()
        hands.detect_for_video.return_value.hand_landmarks = []
        face.detect_for_video.return_value.detections = []
        args = Namespace(list_audio=False, audio_test=False, self_test=False,
                         preview=True, camera=0, virtual_audio=False, mute=True,
                         no_local_sound=False)
        with ExitStack() as stack:
            def mock(name, **kwargs):
                return stack.enter_context(patch("goku." + name, **kwargs))
            mock("download_models")
            mock("create_tasks", return_value=(Mock(), hands, face))
            mock("open_camera", return_value=camera)
            mock("time.monotonic", side_effect=[0, 0, 4.1, 5, 5.2, 5.21])
            mock("GestureLatch.update", return_value=True)
            mock("silhouette", return_value=np.ones((24, 24), dtype=np.float32))
            render = mock("render_teleport", return_value=frame)
            mock("cv2.resize", return_value=frame)
            for name in ("namedWindow", "resizeWindow", "imshow", "destroyAllWindows"):
                mock("cv2." + name)
            mock("cv2.waitKey", side_effect=[0, ord("q")])
            mock("cv2.getWindowProperty", return_value=1)
            run(args)
        self.assertAlmostEqual(render.call_args.args[3], 0.01 / DURATION)

    def test_endpoints_and_background_outside_silhouette(self):
        person = np.full((40, 60, 3), 200, dtype=np.uint8)
        background = np.full_like(person, 30)
        mask = np.zeros((40, 60), dtype=np.float32)
        mask[5:35, 20:40] = 1
        visible = background.copy()
        visible[mask == 1] = 200
        for returning in (False, True):
            for progress in (0, 0.5, 1):
                result = render_teleport(person, background, mask, progress, returning)
                self.assertEqual(result.dtype, np.uint8)
                np.testing.assert_array_equal(result[mask == 0], background[mask == 0])
                if progress in (0, 1):
                    expected = visible if (progress == 1) == returning else background
                    np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
