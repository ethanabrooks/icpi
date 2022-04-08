#! /usr/bin/env bash
mkdir -p logs ~/.cache/GPT ~/.cache/huggingface
name=$(basename "$PWD")_agent
docker build -t "$name" .
docker run --rm -it \
	--env-file .env \
	--gpus "$1" \
	-e HOST_MACHINE="$(hostname -s)" \
	-e TERM=xterm-256color \
	-v "$(pwd)/logs:/tmp/logs" \
	-v "$HOME/.cache/GPT/:/root/.cache/GPT" \
	-v "$HOME/.cache/data/:/root/.cache/data" \
	-v "$HOME/.cache/huggingface/:/root/.cache/huggingface" \
	"$name" "${@:2}"
