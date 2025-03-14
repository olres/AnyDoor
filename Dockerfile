FROM continuumio/miniconda3

WORKDIR /usr/src/app

RUN apt-get -y update && apt-get -y upgrade \
    && apt-get install -y --no-install-recommends ffmpeg \ 
    && apt-get install libgl1

# Only copy the environment file
COPY environment.yaml .

# Create the environment - this step will be cached if environment.yaml doesn't change
RUN conda env create -f environment.yaml

SHELL ["conda", "run", "-n", "anydoor", "/bin/bash", "-c"]

# We'll run this through docker-compose command instead
CMD ["conda", "run", "--no-capture-output", "-n", "anydoor", "python", "run_inference.py"]



