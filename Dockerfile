FROM python:3.9.13

WORKDIR /workspace

RUN apt update &&\
    rm -rf ~/.cache &&\
    apt clean all

# python
WORKDIR /workspace
RUN pip install --upgrade pip &&\
    rm -rf ~/.cache
RUN pip install poetry
COPY ./pyproject.toml /workspace/pyproject.toml
RUN poetry config virtualenvs.create false
RUN poetry install

# files
WORKDIR /workspace
COPY . /workspace

WORKDIR /workspace
VOLUME "/workspace"