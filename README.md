# Goku Teleport
<img width="373" height="200" alt="dbz-dragon-ball-z" src="https://raw.githubusercontent.com/sergiolms/goku-teleport-webcam/refs/heads/main/assets/goku-teleport.gif" />
Teleport out of your webcam by holding two fingers together against your forehead.
Video, gesture detection and audio mixing run locally; nothing is recorded.

Requires **Python 3.13** (3.11–3.13 supported) and a desktop session. The first
launch installs dependencies and downloads three MediaPipe models.

| System | Virtual camera | Voice + effect | Run |
| --- | --- | --- | --- |
| macOS 13+, Apple Silicon | [OBS 30+](https://obsproject.com/) | [BlackHole 2ch](https://github.com/ExistentialAudio/BlackHole) | Double-click `Goku.command` |
| Windows 10/11, x64 | [OBS](https://obsproject.com/) | [VB-CABLE](https://vb-audio.com/Cable/) | Double-click `Goku.bat` |
| Linux x64, glibc 2.28+ | [v4l2loopback](https://github.com/v4l2loopback/v4l2loopback) | PulseAudio / PipeWire with `pipewire-pulse` | `bash goku.sh` |

- **Mac:** start OBS Virtual Camera once, allow its extension, stop it and close
  OBS. Install BlackHole with `audio-install.command` (Homebrew required) or its
  official installer. Restart if it is missing. Allow Camera/Microphone for Terminal.
- **Windows:** install OBS, VB-CABLE and Python 3.13 including the `py` launcher.
  Installing the audio driver requires administrator access and a restart.
  Keep OBS's virtual camera stopped while Goku produces video.
- **Linux Debian/Ubuntu:** use the commands below with Python 3.11–3.13. Load
  v4l2loopback after each reboot. Secure Boot may require signing the DKMS module.
  Choose another video number if `/dev/video10` is already occupied.

```bash
sudo apt install python3-venv libgl1 libglib2.0-0 libportaudio2 pulseaudio-utils v4l2loopback-dkms linux-headers-$(uname -r)
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label=Goku exclusive_caps=1
bash goku.sh --virtual-device /dev/video10
```

Leave the frame empty for four seconds at startup to capture the background.

Keep the camera still and use good light. **Space** recaptures the background;
**Q / Escape** quits with the preview window focused.

In Meet/Zoom, select **OBS Virtual Camera** (Mac/Windows) or **Goku** (Linux).
Select **BlackHole 2ch**, **CABLE Output**, or **Goku Microphone** as the microphone,
respectively. Keep your headphones as the speaker and leave Goku running.

Only your physical microphone and the effect are mixed; system audio is not captured.
Do not select the virtual cable/sink as your system or meeting speaker.
Disable meeting backgrounds; reduce noise suppression if it removes the effect.
Reopen Goku after changing headsets. When closing Goku, select your normal
microphone to continue the call. Linux audio modules are removed on normal exit;
system input/output defaults are never changed.

## Options

Append options to the launcher:
`Goku.bat --preview`, `bash goku.sh --preview`, `./Goku.command --preview` (Terminal).

| Option | Purpose |
| --- | --- |
| `--preview` | Preview without virtual drivers |
| `--no-virtual-audio` | Video only; use normal meeting mic |
| `--camera 1` | Another webcam |
| `--list-audio` | List mic names or indices |
| `--mic "NAME"` | Physical mic; Linux uses source names |
| `--audio-test` | Five seconds of voice + effect, no webcam |
| `--mute` | Mute effect, keep voice |
| `--no-local-sound` | Effect only in the call |
| `--setup` | Install and check models, no camera/mic |

## Repository

Validated on Apple Silicon macOS. Windows/Linux have automated platform tests,
but camera/audio hardware still needs testing there. These are source launchers,
not standalone installers.

Models: [MediaPipe](https://developers.google.com/edge/mediapipe/solutions/vision/face_detector).
