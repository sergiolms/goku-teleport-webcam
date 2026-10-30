#!/bin/zsh
cd -- "${0:A:h}" || exit 1
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
if [[ -x .venv/bin/python ]]; then
  .venv/bin/python -u launch.py "$@"
else
  python3 launch.py "$@"
fi
result=$?
if (( result != 0 )); then
  read '?Press Return to close.'
fi
exit "$result"
