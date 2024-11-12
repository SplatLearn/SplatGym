FROM ubuntu:22.04
ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt update && apt install -y python3 python3-pip git build-essential cmake libpcl-dev liboctomap-dev libeigen3-dev pybind11-dev libfcl-dev libccd-dev


# Clone the repository
RUN git clone https://github.com/SplatLearn/SplatGym.git
RUN git clone https://github.com/SplatLearn/collision_detector.git

# Build the collision detector
RUN mkdir -p collision_detector/build && cd collision_detector/build && cmake ../pcd2bt && make -j

# Install the requirements
RUN pip3 install -r SplatGym/requirements.txt

# Test the installation
RUN python3 -c "from nerfgym.NeRFEnv import NeRFEnv"
