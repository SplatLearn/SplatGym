FROM ghcr.io/nerfstudio-project/nerfstudio:1.1.3
ARG DEBIAN_FRONTEND=noninteractive

# get the source
COPY . /SplatGym

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip git build-essential \
    cmake libpcl-dev liboctomap-dev \
    libeigen3-dev pybind11-dev libfcl-dev \
    libccd-dev swig \
    curl wget && \
    apt-get clean

# Build the collision detector
RUN git clone https://github.com/SplatLearn/collision_detector.git && \
    mkdir -p collision_detector/build && \
    cd collision_detector/build && \
    cmake ../pcd2bt && make -j && \
    cp pybind_collision_detector.cpython*.so /SplatGym/src && \
    rm -rf /collision_detector

# Install the requirements
RUN pip3 --no-cache-dir install -r SplatGym/requirements.txt

# Set the environment variables
ENV PYTHONPATH="${PYTHONPATH}:/SplatGym/src"

# Test the installation
RUN python3 -c "from nerfgym.NeRFEnv import NeRFEnv"
