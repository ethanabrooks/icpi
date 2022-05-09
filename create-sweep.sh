#! /usr/bin/env bash

docker-compose -f docker-compose.yml -f $1 down
docker-compose -f docker-compose.yml -f $1 build --pull
docker-compose -f docker-compose.yml -f $1 --env-file .env up --remove-orphans --force-recreate
