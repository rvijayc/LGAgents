FROM python:3.12.2-slim-bullseye

# Security best practices
ENV PYTHONDONTWRITEBYTECODE=1             \
    PYTHONUNBUFFERED=1                    \
    PIP_NO_CACHE_DIR=off                  \
    PIP_DISABLE_PIP_VERSION_CHECK=on      \
    PIP_DEFAULT_TIMEOUT=100

# Create a non-root user with its home directory
RUN useradd --create-home pythonuser
USER pythonuser
WORKDIR /code
RUN mkdir -p artifacts src

# copy necessary python requirements.
COPY requirements.txt .

# install UV, create a virtual environment and install packages.
RUN pip install uv
ENV PATH="/home/pythonuser/.local/bin:$PATH"
RUN uv venv /code/.venv &&                \
    . /code/.venv/bin/activate &&         \
    uv pip install -r requirements.txt && \
    echo 'source /code/.venv/bin/activate' >> /home/pythonuser/.bashrc

# ensure we are using the right python version.
ENV PATH="/code/.venv/bin:$PATH"

# loop indefinitely.
CMD [ "tail", "-f", "/dev/null"]

