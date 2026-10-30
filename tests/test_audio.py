from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import wave

import numpy as np

from goku_audio import AudioBridge, EffectMixer, LocalSound, load_effect, select_devices


def devices():
    return [
        {"name": "BlackHole 2ch", "max_input_channels": 2, "max_output_channels": 2},
        {"name": "MacBook Pro Microphone", "max_input_channels": 1, "max_output_channels": 0},
        {"name": "AirPods Pro", "max_input_channels": 1, "max_output_channels": 0},
        {"name": "AirPods Pro", "max_input_channels": 0, "max_output_channels": 2},
    ]


class RoutingTests(unittest.TestCase):
    def setUp(self):
        system = patch("goku_audio.platform.system", return_value="Darwin")
        system.start()
        self.addCleanup(system.stop)

    def test_default_microphone_preserved(self):
        self.assertEqual(select_devices(devices(), 2), (2, 0))

    def test_never_loops_blackhole_into_itself(self):
        self.assertEqual(select_devices(devices(), 0), (1, 0))
        with self.assertRaises(RuntimeError):
            select_devices(devices(), 0, "BlackHole")

    def test_microphone_can_be_selected_by_name_or_index(self):
        self.assertEqual(select_devices(devices(), 2, "MacBook"), (1, 0))
        self.assertEqual(select_devices(devices(), 2, "1"), (1, 0))
        self.assertEqual(select_devices(devices(), 1, "AirPods"), (2, 0))

    def test_missing_driver_and_invalid_device_fail_clearly(self):
        with self.assertRaisesRegex(RuntimeError, "audio-install.command"):
            select_devices(devices()[1:], 0)
        with self.assertRaisesRegex(RuntimeError, "microphone"):
            select_devices(devices(), 1, "999")

    def test_stream_routes_physical_input_to_blackhole_and_closes(self):
        with patch("goku_audio.audio_devices", return_value=devices()), \
             patch("goku_audio.sd.default.device", [2, 3]), \
             patch("goku_audio.load_effect", return_value=np.zeros(20)), \
             patch("goku_audio.sd.Stream") as stream:
            bridge = AudioBridge("unused.wav").start()
            self.assertEqual(stream.call_args.kwargs["device"], (2, 0))
            self.assertEqual(stream.call_args.kwargs["channels"], (1, 2))
            bridge.close()
            stream.return_value.stop.assert_called_once()
            stream.return_value.close.assert_called_once()

    def test_failed_start_closes_stream(self):
        with patch("goku_audio.audio_devices", return_value=devices()), \
             patch("goku_audio.sd.default.device", [2, 3]), \
             patch("goku_audio.load_effect", return_value=np.zeros(20)), \
             patch("goku_audio.sd.Stream") as stream:
            stream.return_value.start.side_effect = RuntimeError("Permission")
            with self.assertRaisesRegex(RuntimeError, "microphone"):
                AudioBridge("unused.wav").start()
            stream.return_value.close.assert_called_once()


class MixingTests(unittest.TestCase):
    def test_bundled_effect_starts_within_five_milliseconds(self):
        path = Path(__file__).resolve().parents[1] / "assets" / "goku-teleport-sound.wav"
        effect = load_effect(path, 48000)
        audible = np.flatnonzero(np.abs(effect) > 0.001)
        self.assertGreater(len(audible), 0)
        self.assertLess(audible[0] / 48000, 0.005)

    def test_voice_is_present_on_both_channels(self):
        mixer = EffectMixer(np.array([0.4, 0.3, 0.2], dtype=np.float32))
        output = np.empty((4, 2), dtype=np.float32)
        mixer.fill(np.full((4, 1), 0.1, dtype=np.float32), output)
        np.testing.assert_allclose(output, 0.1)

    def test_effect_continues_across_blocks_and_voice_continues_after(self):
        mixer = EffectMixer(np.array([0.4, 0.3, 0.2], dtype=np.float32))
        mixer.trigger()
        output = np.empty((2, 2), dtype=np.float32)
        microphone = np.full((2, 1), 0.1, dtype=np.float32)
        mixer.fill(microphone, output)
        np.testing.assert_allclose(output[:, 0], [0.5, 0.4])
        mixer.fill(microphone, output)
        np.testing.assert_allclose(output[:, 0], [0.3, 0.1])
        np.testing.assert_array_equal(output[:, 0], output[:, 1])
        mixer.trigger()
        mixer.fill(microphone, output)
        np.testing.assert_allclose(output[:, 0], [0.5, 0.4])

    def test_mix_clips_and_does_not_mutate_microphone(self):
        mixer = EffectMixer(np.array([0.8, -0.8], dtype=np.float32))
        microphone = np.array([[0.8], [-0.8]], dtype=np.float32)
        original = microphone.copy()
        output = np.empty((2, 2), dtype=np.float32)
        mixer.trigger()
        mixer.fill(microphone, output)
        np.testing.assert_allclose(output[:, 0], [1, -1])
        np.testing.assert_array_equal(microphone, original)

    def test_resample_preserves_duration(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "effect.wav"
            with wave.open(str(path), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(44100)
                audio.writeframes(np.zeros(4410, dtype="<i2").tobytes())
            effect = load_effect(path, 48000)
            self.assertEqual(len(effect), 4800)
            self.assertEqual(effect.dtype, np.float32)


class LocalSoundTests(unittest.TestCase):
    def test_preloaded_stream_restarts_effect_without_reopening_device(self):
        effect = np.array([0.4, 0.3, 0.2], dtype=np.float32)
        with patch("goku_audio.load_effect", return_value=effect) as load, \
             patch("goku_audio.sd.query_devices", return_value={"max_output_channels": 2}), \
             patch("goku_audio.sd.OutputStream") as stream:
            sound = LocalSound("sound.wav").start()
            output = np.empty((2, 2), dtype=np.float32)
            sound._callback(output, 2, None, None)
            np.testing.assert_array_equal(output, 0)
            sound.play()
            sound._callback(output, 2, None, None)
            np.testing.assert_allclose(output[:, 0], [0.4, 0.3])
            sound._callback(output, 2, None, None)
            np.testing.assert_allclose(output[:, 0], [0.2, 0])
            sound.play()
            sound._callback(output, 2, None, None)
            np.testing.assert_allclose(output[:, 0], [0.4, 0.3])
            np.testing.assert_array_equal(output[:, 0], output[:, 1])
            load.assert_called_once()
            stream.assert_called_once()
            stream.return_value.start.assert_called_once()
            sound.close()
            sound.close()
            stream.return_value.stop.assert_called_once()
            stream.return_value.close.assert_called_once()

    def test_failed_start_closes_output_stream(self):
        with patch("goku_audio.load_effect", return_value=np.zeros(20)), \
             patch("goku_audio.sd.query_devices", return_value={"max_output_channels": 2}), \
             patch("goku_audio.sd.OutputStream") as stream:
            stream.return_value.start.side_effect = RuntimeError("Device unavailable")
            with self.assertRaisesRegex(RuntimeError, "Device unavailable"):
                LocalSound("sound.wav").start()
            stream.return_value.close.assert_called_once()

    def test_mono_output_device_opens_mono_stream(self):
        effect = np.array([0.4, 0.3], dtype=np.float32)
        with patch("goku_audio.load_effect", return_value=effect), \
             patch("goku_audio.sd.query_devices", return_value={"max_output_channels": 1}), \
             patch("goku_audio.sd.OutputStream") as stream:
            sound = LocalSound("sound.wav").start()
            self.assertEqual(stream.call_args.kwargs["channels"], 1)
            output = np.empty((2, 1), dtype=np.float32)
            sound.play()
            sound._callback(output, 2, None, None)
            np.testing.assert_allclose(output[:, 0], [0.4, 0.3])


if __name__ == "__main__":
    unittest.main()
