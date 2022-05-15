#! /usr/bin/env bash

docker-compose -p $2 -f docker-compose.yml -f $1 down
docker-compose -p $2 -f docker-compose.yml -f $1 build --pull
docker-compose -p $2 -f docker-compose.yml -f $1 --env-file .env up --remove-orphans --force-recreate
