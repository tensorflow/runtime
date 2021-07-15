FROM ubuntu:20.04

RUN apt update &&\
    apt install -y lsb-release wget software-properties-common zip unzip

# clang
RUN wget https://apt.llvm.org/llvm.sh &&\
    chmod +x llvm.sh &&\
    ./llvm.sh 11 &&\
    rm llvm.sh &&\
    update-alternatives --install /usr/bin/clang clang /usr/bin/clang-11 11 &&\
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-11 11 &&\
    update-alternatives --install /usr/bin/python python /usr/bin/python3 3

# bazel
RUN wget https://github.com/bazelbuild/bazel/releases/download/4.0.0/bazel-4.0.0-installer-linux-x86_64.sh &&\
    chmod +x bazel-4.0.0-installer-linux-x86_64.sh &&\
    ./bazel-4.0.0-installer-linux-x86_64.sh &&\
    rm bazel-4.0.0-installer-linux-x86_64.sh

ARG UID
ARG GID
RUN groupadd -g $GID dummy && useradd -m -r --uid $UID -g dummy --groups "0" dummy

USER dummy


