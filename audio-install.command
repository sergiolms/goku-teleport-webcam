#!/bin/zsh
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
print 'Installing BlackHole 2ch to send your voice and the effect to Meet / Zoom.'
print 'macOS may ask for your administrator password. It will not be displayed as you type.'
HOMEBREW_NO_AUTO_UPDATE=1 brew install --cask blackhole-2ch
result=$?
if (( result == 0 )); then
  print '\nInstallation complete. Close and reopen Goku, Meet or Zoom.'
  print 'Select BlackHole 2ch as the microphone and keep your headphones as the speaker.'
  print 'If BlackHole still does not appear, restart your Mac.'
else
  print '\nInstallation failed. Check the message above.'
fi
read '?Press Return to close.'
exit "$result"
