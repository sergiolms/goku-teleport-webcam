"""Create a local Python environment and run Goku from any working directory."""

import hashlib
from pathlib import Path
import subprocess
import sys
import venv


ROOT = Path(__file__).resolve().parent


def main():
    if not (3, 11) <= sys.version_info[:2] <= (3, 13):
        print("Use Python 3.11–3.13.")
        return 1
    environment = ROOT / ".venv"
    executable = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not executable.is_file():
        print("Preparing Python environment…", flush=True)
        venv.create(environment, with_pip=True)
    arguments = sys.argv[1:]
    setup = "--setup" in arguments
    if setup:
        arguments.remove("--setup")
    requirements = ROOT / "requirements.txt"
    digest = hashlib.sha256(requirements.read_bytes()).hexdigest()
    marker = environment / ".requirements-sha256"
    if setup or not marker.exists() or marker.read_text() != digest:
        subprocess.run([str(executable), "-m", "pip", "install", "-r", str(requirements)], check=True)
        marker.write_text(digest)
    if setup:
        return subprocess.call([str(executable), str(ROOT / "goku.py"), "--self-test"], cwd=ROOT)
    if not any(flag in arguments for flag in ("--preview", "--self-test", "--list-audio", "--audio-test")):
        arguments.insert(0, "--virtual-audio")
    return subprocess.call([str(executable), "-u", str(ROOT / "goku.py"), *arguments], cwd=ROOT)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"Setup failed: {error}")
        raise SystemExit(1)
