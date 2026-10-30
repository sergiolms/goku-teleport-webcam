#!/bin/zsh
cd -- "${0:A:h}" || exit 1
./Goku.command --audio-test "$@"
result=$?
read '?Press Return to close.'
exit "$result"
