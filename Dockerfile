#flywheel/inveon-pet-metadata

# Start with python 3.7
FROM python:3.7
MAINTAINER Flywheel <support@flywheel.io>

# Install pandas
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Flywheel spec (v0)
ENV FLYWHEEL /flywheel/v0
RUN mkdir -p ${FLYWHEEL}
COPY run.py ${FLYWHEEL}/run.py
COPY manifest.json ${FLYWHEEL}/manifest.json
RUN chmod +x ${FLYWHEEL}/run.py

# Configure entrypoint
ENTRYPOINT ["/flywheel/v0/run.py"]