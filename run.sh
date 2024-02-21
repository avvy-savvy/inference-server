#!/bin/bash
set -x

type=tf
while [ "$#" -gt 0 ];
do case $1 in
  -t|--type) type="$2"
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
docker build --build-arg BASE_IMAGE="${base_image}" -t "${image_name}" .
# run container based on new image
docker run --gpus all -it --rm "${image_name}"
