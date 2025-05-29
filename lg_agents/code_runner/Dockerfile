FROM python:3.12.2-slim-bullseye
ARG DOCKER_REGISTRY
ENV DOCKER_REGISTRY=${DOCKER_REGISTRY}

# Security best practices
ENV PYTHONDONTWRITEBYTECODE=1                \
    PYTHONUNBUFFERED=1                       \
    PIP_NO_CACHE_DIR=off                     \
    PIP_DISABLE_PIP_VERSION_CHECK=on         \
    PIP_DEFAULT_TIMEOUT=100

# Create a non-root user with its home directory
RUN useradd --create-home -s /bin/bash pythonuser
SHELL ["/bin/bash", "-c"]
USER pythonuser
WORKDIR /home/pythonuser

# create artifacts and src folders.
RUN mkdir -p artifacts src

# copy necessary python requirements.
COPY --chown=pythonuser:pythonuser requirements.txt .

# install UV, create a virtual environment and install packages.
RUN pip install uv
ENV PATH="/home/pythonuser/.local/bin:$PATH"
RUN uv venv /home/pythonuser/.venv &&        \
    source /home/pythonuser/.venv/bin/activate && \
    uv pip install -r requirements.txt &&    \
    echo 'source /home/pythonuser/.venv/bin/activate' >> /home/pythonuser/.bashrc

# ensure we are using the right python version.
ENV PATH="/home/pythonuser/.venv/bin:$PATH"

# loop indefinitely.
CMD [ "tail", "-f", "/dev/null"]

