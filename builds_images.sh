#!/bin/bash

echo "build the docker image CLIENT"
docker build -t rsna:client -f Dockerfile.client .

echo "build the docker image SERVER"
docker build -t rsna:server -f Dockerfile.server .

echo "build the docker image SEGM"
docker build -t rsna:segm -f Dockerfile.segm .

echo "build the docker image CLF"
docker build -t rsna:clf -f Dockerfile.clf .


