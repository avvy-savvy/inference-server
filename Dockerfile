ARG BASE_IMAGE=tensorflow/tensorflow:2.13.0-gpu
FROM $BASE_IMAGE

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install any python packages you need
COPY requirements.txt /tmp/requirements.txt

# Install requirements
RUN python3 -m pip install -r /tmp/requirements.txt

# Copy source code
COPY app.py /app/app.py

# Set the working directory
WORKDIR /app

# Set flask environment labels
ENV FLASK_APP=app
ENV FLASK_ENV=development

# Expose port
EXPOSE 5000

# Set the entrypoint
ENTRYPOINT [ "flask", "run", "--host=0.0.0.0" ]