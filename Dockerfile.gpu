# inspired by https://sourcery.ai/blog/python-docker/
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04 as base
ENV LC_ALL C.UTF-8

# no .pyc files
ENV PYTHONDONTWRITEBYTECODE 1

# traceback on segfault
ENV PYTHONFAULTHANDLER 1

# use ipdb for breakpoints
ENV PYTHONBREAKPOINT=ipdb.set_trace

# Update Nvidia GPG key & install common dependencies
WORKDIR /tmp
RUN rm /etc/apt/sources.list.d/cuda.list \
 && rm /etc/apt/sources.list.d/nvidia-ml.list \
 && apt-key del 7fa2af80 \
 && apt-get update -q && apt-get install -yq --no-install-recommends wget \
 && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb \
 && dpkg -i cuda-keyring_1.0-1_all.deb \
 && apt-get update -q \
 && DEBIAN_FRONTEND="noninteractive" \
    apt-get install -yq \
      # git-state
      git \

      # primary interpreter
      python3.9 \

      # for shelve
      python3.9-gdb \

      # for deepspeed models
      python3.9-dev \

      # redis-python
      redis \

 && apt-get clean

FROM base AS python-deps

# build dependencies
RUN apt-get update -q \
 && DEBIAN_FRONTEND="noninteractive" \
    apt-get install -yq \

      # required by poetry
      python3-pip \

      # required for redis
      gcc \

 && apt-get clean

WORKDIR "/deps"

COPY pyproject.toml poetry.lock /deps/
RUN pip3 install poetry && poetry install


FROM base AS runtime

WORKDIR "/root"
ENV VIRTUAL_ENV=/root/.cache/pypoetry/virtualenvs/gqn-K3BlsyQa-py3.9
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
COPY --from=python-deps $VIRTUAL_ENV $VIRTUAL_ENV
COPY . .
