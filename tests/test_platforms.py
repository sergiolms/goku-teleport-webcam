import unittest
from unittest.mock import Mock, patch

import cv2

from goku import camera_backends, open_camera, virtual_backend
from goku_audio import LinuxAudioBridge, select_devices, select_linux_source


class VideoTests(unittest.TestCase):
    def test_native_camera_and_virtual_backends(self):
        self.assertEqual(camera_backends("Darwin"), (cv2.CAP_AVFOUNDATION,))
        self.assertEqual(camera_backends("Windows"), (cv2.CAP_DSHOW, cv2.CAP_MSMF))
        self.assertEqual(camera_backends("Linux"), (cv2.CAP_V4L2,))
        self.assertEqual(virtual_backend("Linux"), "v4l2loopback")
        for system in ("Darwin", "Windows"):
            self.assertEqual(virtual_backend(system), "obs")

    def test_windows_camera_releases_failed_backend_and_tries_next(self):
        failed = Mock()
        failed.isOpened.return_value = False
        working = Mock()
        working.isOpened.return_value = True
        with patch("goku.platform.system", return_value="Windows"), \
             patch("goku.cv2.VideoCapture", side_effect=[failed, working]) as capture:
            self.assertIs(open_camera(1), working)
            failed.release.assert_called_once()
            self.assertEqual(capture.call_args_list[1].args, (1, cv2.CAP_MSMF))


class WindowsAudioTests(unittest.TestCase):
    def devices(self):
        return [
            {"name": "CABLE Input", "hostapi": 0, "max_input_channels": 0, "max_output_channels": 2},
            {"name": "Microphone USB", "hostapi": 1, "max_input_channels": 1, "max_output_channels": 0},
            {"name": "CABLE Input", "hostapi": 1, "max_input_channels": 0, "max_output_channels": 2},
            {"name": "CABLE Output", "hostapi": 1, "max_input_channels": 2, "max_output_channels": 0},
        ]

    def test_input_and_cable_use_same_host_api(self):
        self.assertEqual(select_devices(self.devices(), 1, system="Windows"), (1, 2))

    def test_virtual_cable_cannot_be_microphone(self):
        with self.assertRaises(RuntimeError):
            select_devices(self.devices(), 1, "3", system="Windows")

    def test_mismatched_host_api_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "same audio API"):
            select_devices(self.devices()[:2], 1, system="Windows")


class LinuxAudioTests(unittest.TestCase):
    def sources(self):
        return [{"name": "alsa_input.usb", "description": "USB microphone", "monitor_of_sink": None},
                {"name": "speakers.monitor", "description": "Speakers", "monitor_of_sink": 0},
                {"name": "goku_microphone_1", "description": "Goku", "monitor_of_sink": None}]

    def test_monitor_and_previous_goku_source_cannot_loop_back(self):
        self.assertEqual(select_linux_source(self.sources(), "alsa_input.usb"), "alsa_input.usb")
        self.assertEqual(select_linux_source(self.sources(), "speakers.monitor", "USB"), "alsa_input.usb")
        for source in ("speakers.monitor", "goku_microphone_1"):
            with self.assertRaises(RuntimeError):
                select_linux_source(self.sources(), source)

    def test_modules_unloaded_in_reverse_order_without_changing_defaults(self):
        calls = []
        def pulse(*args):
            calls.append(args)
            if args[0] == "get-default-source":
                return "alsa_input.usb"
            if args[0] == "load-module":
                return str(sum(c[0] == "load-module" for c in calls))
            return ""
        with patch("goku_audio.shutil.which", return_value="/usr/bin/tool"), \
             patch("goku_audio.linux_sources", return_value=self.sources()), \
             patch("goku_audio.pulse", side_effect=pulse):
            bridge = LinuxAudioBridge("sound.wav").start()
            self.assertEqual(len(bridge.modules), 3)
            bridge.close()
        self.assertEqual([c for c in calls if c[0] == "unload-module"],
                         [("unload-module", "3"), ("unload-module", "2"), ("unload-module", "1")])
        self.assertFalse(any(c[0].startswith("set-default") for c in calls))

    def test_partial_setup_failure_unloads_created_modules(self):
        def pulse(*args):
            if args[0] == "get-default-source":
                return "alsa_input.usb"
            if args[:2] == ("load-module", "module-null-sink"):
                return "99"
            if args[0] == "load-module":
                raise RuntimeError("Cannot load module")
            return ""
        with patch("goku_audio.shutil.which", return_value="/usr/bin/tool"), \
             patch("goku_audio.linux_sources", return_value=self.sources()), \
             patch("goku_audio.pulse", side_effect=pulse) as command:
            with self.assertRaises(RuntimeError):
                LinuxAudioBridge("sound.wav").start()
            command.assert_any_call("unload-module", "99")


if __name__ == "__main__":
    unittest.main()
