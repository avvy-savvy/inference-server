# inference-server

## Starting server

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