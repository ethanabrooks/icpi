#! /usr/bin/env bash
name=$(basename "$PWD")_run
docker build -f Dockerfile -t "$name" .
docker run --rm -it \
	--env-file .env \
	--shm-size=10.24gb \
  --network="host" \
	-h="$(hostname -s)" \
	-e TERM=xterm-256color \
	-v "$(pwd)/logs:/root/logs" \
	-v "$(pwd)/completions:/root/completions" \
	-v "$HUGGINGFACE_CACHE_DIR:/root/.cache/huggingface/" \
	"$name" "${@:1}"
