#! /usr/bin/env bash

docker-compose build
docker-compose -f docker-compose.yml -f join-existing-sweep.yml --env-file .env up --remove-orphans --force-recreate
