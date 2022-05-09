#! /usr/bin/env bash
mkdir -p logs completions
name=$(basename "$PWD")_run
docker build -f Dockerfile.cpu -t "$name" .
docker run --rm -it \
	--env-file .env \
	-h="$(hostname -s)" \
	-e TERM=xterm-256color \
	-v "$(pwd)/logs:/root/logs" \
	-v "$(pwd)/completions:/root/completions" \
	"$name" "${@:1}"
