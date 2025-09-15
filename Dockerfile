FROM --platform=linux/amd64 pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime
# Using a 'large' base container to show-case how to load pytorch and use the GPU (when enabled)

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED=1
ENV PYTHONWARNINGS="ignore"

# Install git so we can clone the nnunet repository
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev
RUN groupadd -r user && useradd -m --no-log-init -r -g user user

ENV PATH="/home/user/.local/bin:${PATH}"
USER user

COPY --chown=user:user requirements.txt /opt/app/

# You can add any Python dependencies to requirements.txt
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

### Clone nnUNet
COPY --chown=user:user ./resources/nnUNet/ /opt/resources/nnunet/
COPY --chown=user:user ./resources/nnUNet_results/ /opt/app/resources/nnUNet_results/
# Install a few dependencies that are not automatically installed
# RUN sudo apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev
# RUN pip3 install --no-deps -U "monai"

RUN pip3 install \
        -e /opt/resources/nnunet \
        graphviz \
        tensorboard \
        onnx \ 
        opencv-python-headless \
        SimpleITK && \
    rm -rf ~/.cache/pip

RUN pip3 install --no-deps -U monai && \
    pip3 install fire nibabel pyyaml tqdm einops

# USER root
# RUN conda install -c conda-forge pydensecrf
# USER user
COPY --chown=user:user ./resources/monai/ /opt/app/resources/monai/
WORKDIR /opt/app

COPY --chown=user:user inference.py /opt/app/
COPY --chown=user:user data_utils.py /opt/app/
COPY --chown=user:user process_hecktor.py /opt/app/
COPY --chown=user:user inference_monai.py /opt/app/
### Set environment variable defaults
ENV nnUNet_raw="/opt/app/resources/nnunet/nnUNet_raw" \
    nnUNet_preprocessed="/opt/app/resources/nnunet/nnUNet_preprocessed" \
    nnUNet_results="/opt/app/resources/nnunet/nnUNet_results"

ENTRYPOINT ["python", "inference.py"]
