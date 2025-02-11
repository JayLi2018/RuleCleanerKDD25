#!/bin/bash

# Check if tmux is running
if ! tmux ls &> /dev/null; then
    echo "No active tmux session found. Starting a new one..."
    tmux new-session -d -s keep_alive
fi

# Prevent macOS from sleeping
echo "Keeping tmux alive. Close the lid safely."
caffeinate -dims &
CAFFEINATE_PID=$!

# Wait for tmux session to end
tmux attach

# Once tmux session ends, stop preventing sleep
kill $CAFFEINATE_PID
echo "tmux session ended. Sleep prevention disabled."
