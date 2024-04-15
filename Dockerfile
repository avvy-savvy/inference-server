FROM tensorflow/tensorflow:2.13.0-gpu

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install any python packages you need
COPY requirements.txt /tmp/requirements.txt

# Install requirements
RUN python3 -m pip install -r /tmp/requirements.txt

# Copy source code
COPY app.py /app/app.py
COPY data /app/data

# Set the working directory
WORKDIR /app

# Set flask environment labels
ENV FLASK_APP=app
ENV FLASK_ENV=development
ARG HOST_URI=0.0.0.0
ARG HOST_PORT=5000
# Entrypoint are unable to use arg values and so converting to environment variables
ENV APP_URI=$HOST_URI
ENV APP_PORT=$HOST_PORT

# Expose port
EXPOSE ${HOST_PORT}

# Set the entrypoint
ENTRYPOINT flask run --host="${APP_URI}" -p "${APP_PORT}"
