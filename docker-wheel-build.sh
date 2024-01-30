#!/bin/sh

docker build --platform=linux/amd64 -t apex-amd64 .

docker run -v "$(pwd):/mnt/host/" apex-amd64 sh -c "cp /mnt/build/*.whl /mnt/host/"
