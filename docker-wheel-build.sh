#!/bin/sh

docker build -t apex-builder .

docker run -v "$(pwd):/mnt/host/" apex-builder sh -c "cp /mnt/build/*.whl /mnt/host/"
