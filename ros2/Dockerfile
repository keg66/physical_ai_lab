FROM ros:jazzy-ros-base-noble

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y \
    ros-jazzy-ros-gz \
    ros-jazzy-demo-nodes-cpp \
    ros-jazzy-demo-nodes-py \
    python3-colcon-common-extensions \
    python3-rosdep \
    git \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# simulation
RUN apt-get update && apt-get install -y \
    ros-jazzy-turtlebot3 \
    ros-jazzy-turtlebot3-simulations \
    ros-jazzy-teleop-twist-keyboard \
    && rm -rf /var/lib/apt/lists/*

# viewer
RUN apt-get update && apt-get install -y \
    ros-jazzy-rviz2 \
    ros-jazzy-rqt-image-view \
    && rm -rf /var/lib/apt/lists/*

ENV TURTLEBOT3_MODEL=waffle_pi

RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc

WORKDIR /ws

CMD ["bash"]
