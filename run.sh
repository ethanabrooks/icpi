#! /usr/bin/env bash
mkdir -p logs completions
name=$(basename "$PWD")_run
docker build -t "$name" .
docker run --rm -it \
  --gpus "\"device=$1\"" \
  --shm-size=5g \
	--env-file .env \
	-h="$(hostname -s)" \
	-e TERM=xterm-256color \
	-v "$(pwd)/logs:/root/logs" \
	-v "$(pwd)/completions:/root/completions"\
  --mount src="$HUGGINGFACE_CACHE_DIR",target="/root/.cache/huggingface/",type=bind \
  --mount src="$TORCH_EXTENSIONS_HOST_DIR",target="/root/.cache/torch_extensions/",type=bind \
	"$name" "${@:2}"
