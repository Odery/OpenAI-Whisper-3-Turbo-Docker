FROM ubuntu:latest
LABEL authors="serpe"

ENTRYPOINT ["top", "-b"]