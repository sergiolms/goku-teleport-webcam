"""Microphone and effect routing without recording system audio."""

import json
import os
import platform
from queue import Empty, SimpleQueue
import shutil
import subprocess
import time
import wave

import numpy as np
import sounddevice as sd


def audio_devices():
    return sd.query_devices()


def select_devices(devices, default_input, microphone=None, system=None):
    system = system or platform.system()
    target = "cable input" if system == "Windows" else "blackhole 2ch"
    outputs = [i for i, d in enumerate(devices)
               if target in d["name"].lower() and d["max_output_channels"] >= 2]
    if not outputs:
        if system == "Windows":
            raise RuntimeError("Install VB-CABLE and restart Windows. See README.")
        raise RuntimeError("BlackHole 2ch is unavailable. Run audio-install.command if it is not installed. If it is already installed, restart macOS.")

    def physical(i):
        return (0 <= i < len(devices) and devices[i]["max_input_channels"] > 0
                and not any(name in devices[i]["name"].lower()
                            for name in ("blackhole", "loopback", "vb-cable", "cable output", "cable input", "stereo mix", "monitor")))

    if microphone is not None:
        if str(microphone).isdigit():
            candidates = [int(microphone)]
        else:
            candidates = [i for i, d in enumerate(devices)
                          if str(microphone).lower() in d["name"].lower() and physical(i)]
            if system == "Windows" and 0 <= default_input < len(devices):
                candidates = [i for i in candidates if devices[i].get("hostapi") == devices[default_input].get("hostapi")]
        if len(candidates) != 1 or not physical(candidates[0]):
            raise RuntimeError("Invalid or ambiguous microphone. Use --list-audio to list names and --mic to select a physical microphone.")
        input_device = candidates[0]
    elif physical(default_input):
        input_device = default_input
    else:
        built_in = [i for i, d in enumerate(devices)
                    if physical(i) and "macbook" in d["name"].lower()]
        if len(built_in) != 1:
            raise RuntimeError("The default microphone is not a physical input. Choose one with --mic; use --list-audio to list available inputs.")
        input_device = built_in[0]
    matching = [i for i in outputs if devices[i].get("hostapi") == devices[input_device].get("hostapi")]
    if not matching:
        raise RuntimeError("Microphone and virtual cable must use the same audio API. Select a different --mic index.")
    return input_device, matching[0]


def load_effect(path, samplerate):
    with wave.open(str(path), "rb") as audio:
        if audio.getsampwidth() != 2:
            raise ValueError("The effect must be a 16-bit PCM WAV file.")
        samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
        samples = samples.reshape(-1, audio.getnchannels()).mean(axis=1) / 32768
        rate = audio.getframerate()
    count = round(len(samples) * samplerate / rate)
    if count == 0:
        raise ValueError("The sound file is empty.")
    return np.interp(np.arange(count) * rate / samplerate,
                     np.arange(len(samples)), samples).astype(np.float32)


class EffectMixer:
    def __init__(self, effect):
        self.effect = effect
        self.position = len(effect)
        self.commands = SimpleQueue()

    def trigger(self):
        self.commands.put(True)

    def fill(self, microphone, output):
        try:
            while self.commands.get_nowait():
                self.position = 0
        except Empty:
            pass
        output[:, 0] = microphone[:, 0]
        count = min(len(output), len(self.effect) - self.position)
        if count:
            output[:count, 0] += self.effect[self.position:self.position + count]
            self.position += count
        np.clip(output[:, 0], -1, 1, out=output[:, 0])
        output[:, 1:] = output[:, :1]


class AudioBridge:
    def __init__(self, sound_path, microphone=None):
        devices = audio_devices()
        self.input_device, self.output_device = select_devices(
            devices, int(sd.default.device[0]), microphone)
        self.microphone_name = devices[self.input_device]["name"]
        self.output_name = devices[self.output_device]["name"]
        self.call_microphone = "CABLE Output" if platform.system() == "Windows" else "BlackHole 2ch"
        self.samplerate = 48000
        self.mixer = EffectMixer(load_effect(sound_path, self.samplerate))
        self.last_status = None
        self.stream = None

    def _callback(self, input_data, output_data, frames, timing, status):
        if status:
            self.last_status = str(status)
        self.mixer.fill(input_data, output_data)

    def start(self):
        try:
            self.stream = sd.Stream(device=(self.input_device, self.output_device),
                samplerate=self.samplerate, channels=(1, 2), dtype="float32",
                latency="low", callback=self._callback)
            self.stream.start()
        except Exception as error:
            self.close()
            raise RuntimeError("Cannot open audio stream. Check microphone permissions and connection; try another --mic.\n" + str(error)) from error
        print(f"Audio: {self.microphone_name} + effect → {self.output_name}", flush=True)
        print(f"Meet / Zoom microphone: {self.call_microphone}. Speakers: your headphones.", flush=True)
        return self

    def check(self):
        if self.stream is None or not self.stream.active:
            raise RuntimeError("The audio mixer has stopped. Check that the microphone is still connected and reopen Goku.")

    def play(self):
        self.mixer.trigger()

    def close(self):
        if self.stream is not None:
            try:
                self.stream.stop()
            finally:
                self.stream.close()
                self.stream = None


def pulse(*args):
    try:
        return subprocess.run(["pactl", *args], check=True, capture_output=True,
                              text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("PulseAudio / PipeWire unavailable. Install pulseaudio-utils and enable PulseAudio or pipewire-pulse.") from error


def linux_sources():
    return json.loads(pulse("--format=json", "list", "sources"))


def select_linux_source(sources, default, microphone=None):
    physical = [s for s in sources if not s["name"].endswith(".monitor")
                and not s["name"].startswith("goku_")
                and s.get("monitor_of_sink") in (None, 4294967295, "4294967295")]
    if microphone is None:
        matches = [s for s in physical if s["name"] == default]
    else:
        exact = [s for s in physical if s["name"] == microphone]
        matches = exact or [s for s in physical
                            if microphone.lower() in s.get("description", "").lower()]
    if len(matches) != 1:
        raise RuntimeError("Choose a physical microphone with --mic; see --list-audio.")
    return matches[0]["name"]


class LinuxAudioBridge:
    def __init__(self, sound_path, microphone=None):
        if not shutil.which("paplay") or not shutil.which("pactl"):
            raise RuntimeError("Install pulseaudio-utils. See README.")
        self.source = select_linux_source(linux_sources(), pulse("get-default-source"), microphone)
        self.sink = f"goku_sink_{os.getpid()}"
        self.call_microphone = f"goku_microphone_{os.getpid()}"
        self.sound_path = sound_path
        self.modules = []
        self.process = None
        self.last_check = 0

    def start(self):
        try:
            self.modules.append(pulse("load-module", "module-null-sink",
                f"sink_name={self.sink}", "rate=48000", "channels=2",
                "sink_properties=device.description=Goku_Audio"))
            self.modules.append(pulse("load-module", "module-remap-source",
                f"master={self.sink}.monitor", f"source_name={self.call_microphone}",
                "source_properties=device.description=Goku_Microphone"))
            self.modules.append(pulse("load-module", "module-loopback",
                f"source={self.source}", f"sink={self.sink}", "latency_msec=30"))
        except Exception:
            self.close()
            raise
        print(f"Audio: {self.source} + effect → Goku Microphone", flush=True)
        print("Meet / Zoom microphone: Goku Microphone. Speakers: your headphones.", flush=True)
        return self

    def play(self):
        stop_process(self.process)
        self.process = subprocess.Popen(["paplay", "--latency-msec=10", f"--device={self.sink}", str(self.sound_path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def check(self):
        if self.process is not None and self.process.poll() not in (None, 0):
            raise RuntimeError("Linux effect playback failed.")
        now = time.monotonic()
        if now - self.last_check >= 2:
            self.last_check = now
            loaded = {line.split()[0] for line in pulse("list", "short", "modules").splitlines()}
            if not set(self.modules).issubset(loaded):
                raise RuntimeError("Virtual microphone disconnected; restart Goku.")

    def close(self):
        stop_process(self.process)
        for module in reversed(self.modules):
            try:
                pulse("unload-module", module)
            except RuntimeError:
                pass
        self.modules.clear()


def stop_process(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)


class LocalSound:
    def __init__(self, path):
        self.samplerate = 48000
        self.mixer = EffectMixer(load_effect(path, self.samplerate))
        self.stream = None

    def _callback(self, output, frames, timing, status):
        self.mixer.fill(np.zeros((frames, 1), dtype=np.float32), output)

    def start(self):
        try:
            # Bluetooth headsets in hands-free mode expose a mono output.
            channels = min(2, sd.query_devices(kind="output")["max_output_channels"])
            self.stream = sd.OutputStream(samplerate=self.samplerate,
                channels=channels, dtype="float32", latency="low", callback=self._callback)
            self.stream.start()
        except Exception:
            self.close()
            raise
        return self

    def play(self):
        self.mixer.trigger()

    def close(self):
        if self.stream is not None:
            try:
                self.stream.stop()
            finally:
                self.stream.close()
                self.stream = None


def create_bridge(sound_path, microphone=None):
    bridge = LinuxAudioBridge if platform.system() == "Linux" else AudioBridge
    return bridge(sound_path, microphone)


def list_inputs():
    if platform.system() == "Linux":
        for source in linux_sources():
            print(f'{source["name"]}: {source.get("description", "")}')
    else:
        for index, device in enumerate(audio_devices()):
            if device["max_input_channels"]:
                print(f'{index}: {device["name"]}')
