# inspired by https://sourcery.ai/blog/python-docker/
FROM ubuntu:20.04 as base
ENV LC_ALL C.UTF-8

# no .pyc files
ENV PYTHONDONTWRITEBYTECODE 1

# traceback on segfau8t
ENV PYTHONFAULTHANDLER 1

# use ipdb for breakpoints
ENV PYTHONBREAKPOINT=ipdb.set_trace

# common dependencies
RUN apt-get update -q \
 && DEBIAN_FRONTEND="noninteractive" \
    apt-get install -yq \
      # git-state
      git \

      # primary interpreter
      python3.9 \

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

WORKDIR "/project"
ENV VIRTUAL_ENV=/root/.cache/pypoetry/virtualenvs/gql-K3BlsyQa-py3.9
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
COPY --from=python-deps $VIRTUAL_ENV $VIRTUAL_ENV
COPY . .

ENTRYPOINT ["python"]
