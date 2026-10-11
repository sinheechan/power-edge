#!/bin/bash
P=$HOME/projects/power-edge

if [ "$1" = "stop" ]; then
    tmux kill-session -t power 2>/dev/null
    cd $P/server && docker compose down
    exit
fi

tmux has-session -t power 2>/dev/null && exec tmux attach -t power

if ! docker info >/dev/null 2>&1; then
    "/mnt/c/Program Files/Docker/Docker/Docker Desktop.exe" >/dev/null 2>&1 &
    echo "도커 켜는 중..."
    until docker info >/dev/null 2>&1; do sleep 3; done
fi
cd $P/server && docker compose up -d

tmux new-session -d -s power -c $P "cd $P/edge && source .venv/bin/activate && python simulator.py; bash"
sleep 3
tmux split-window -h -t power -c $P "cd $P/edge && source .venv/bin/activate && python controller.py; bash"
tmux split-window -v -t power:0.0 -c $P "sleep 15; cd $P/ml && source .venv/bin/activate && python serve.py; bash"
tmux split-window -v -t power:0.1 -c $P
tmux attach -t power