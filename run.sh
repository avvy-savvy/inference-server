#!/bin/bash
set -x

type=tf
host_uri="0.0.0.0"
host_port=5000

while [ "$#" -gt 0 ];
do case $1 in
  -t|--type) type="$2"
  shift;;
  -h|--host) host_uri="$2"
  shift;;
  -p|--port) host_port="$2"
  shift;;
  *) echo "Unknown parameter passed: $1";;
esac
shift
done

# use ubuntu-20.04 and python 3.8
# defaulting to tensorflow as it's image is 8GB whereas pytorch image is 20GB
if [[ "$type" == "pth" ]]
then
  base_image="nvcr.io/nvidia/pytorch:23.04-py3"
  echo "Currently only tensorflow is supported"
  echo "PyTorch model is not yet implemented"
  exit
else
  base_image="tensorflow/tensorflow:2.13.0-gpu"
fi

image_name="av-rest.${type}"

# remove existing images
existing_images=$(docker images "${image_name}" --format "{{.ID}}")
existing_images="${existing_images//$'\n'/ }"
existing_images=$(echo "${existing_images}" | xargs)
if [[ "${existing_images}" != "" ]]
then
  docker image rm "${existing_images}"
fi

# build new image
docker build --build-arg BASE_IMAGE="${base_image}" --build-arg HOST_URI="${host_uri}" --build-arg HOST_PORT="${host_port}" -t "${image_name}" .
# run container based on new image
docker run -p "${host_port}":"${host_port}" --gpus all -it --rm "${image_name}"
