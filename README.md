# inference-server

## Starting server

Download additional data
from [drive/data.zip](https://drive.google.com/file/d/1JzG3UyNjoESkjWGsRxdcxH0lB1EeJrFK/view?usp=sharing) and extract
the contents to `data` folder in the root folder.

There are two base images supported by this project.

* Tensorflow: Start the server by running<br />
  ```bash
  sudo ./run.sh -t tf
  ```
* PyTorch: Start the server by running<br />
  ```bash
  sudo ./run.sh -t pth
  ```

Note:

* `sudo` lets us use gpus
* `-t [type_code]` is optional where the default option of `tf` is used as the base image because Tensorflow has a base
  image of 8GB whereas PyTorch has a base image of 20GB.

In both the types, the server is started on port `5000` and can be accessed by navigating to `127.0.0.1:5000`.

## Available APIs

Currently, this project expose 2 APIs.

| URL                | Parameters                                                                                                                                     | Description                                                                        |
|--------------------|------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| `/danger_scale`    | -                                                                                                                                              | Returns a list of classes in the scale.                                            |
| `/treeline_labels` | `lat`: Latitude of the location<br />`long`: Longitude of the location<br />`date`: (Optional) Prediction for a date, defaults to current date | Returns danger levels for _Above Treeline_, _Near Treeline_, and _Below Treeline_. |

## Deploying on AWS

Use instance type of `p3.2xlarge` and AMI containing tesla driver installed.

#### Install Docker

Install docker if it is not yet available.

```bash
sudo yum update -y
sudo amazon-linux-extras install docker

sudo service docker start
sudo usermod -a -G docker ec2-user
```

#### Install Nvidia-Container-Toolkit

Install Nvidia-container-toolkit if not yet available.

```bash

sudo yum-config-manager --disable amzn2-graphics
sudo yum install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
sudo yum-config-manager --enable amzn2-graphics
```

#### Pull image and start in background

```bash
docker pull public.ecr.aws/s8u8x2v9/avalanche-server  # optionally tag this image and use that name to start server
nohup docker run -p 5000:5000 --gpus all --rm avalanche-server &> inf.out &
```