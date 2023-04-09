FROM ubuntu:22.04 

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y sudo build-essential ca-certificates coreutils curl environment-modules \ 
    gfortran git gpg lsb-release python3 python3-distutils python3-venv python3-pip \
    ffmpeg libsm6 libxext6 \
    emacs unzip zip libopenmpi-dev && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash ubuntu \
    && echo "ubuntu:password" | chpasswd

# Grant sudo privileges to the user
RUN mkdir -p /etc/sudoers.d
RUN echo "ubuntu ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/ubuntu

USER ubuntu
RUN mkdir -p /home/ubuntu
COPY ./ /home/ubuntu/isciml/
RUN sudo chown -R ubuntu:ubuntu /home/ubuntu
RUN cd /home/ubuntu/isciml && \
    pip install -r requirements.txt && \
    cd /home/ubuntu/isciml/lib && \
    python -m numpy.f2py -c calc_and_mig_all_rx.f90 gtet.f90 gfacet.f90 ggfacet.f90 gzfacet.f90 check_divzero1.f90 check_divzero2.f90 -m calc_mig_all && \
    rm *.f90

ENV PYTHONPATH="${PYTHONPATH}:/home/ubuntu/isciml/lib"
WORKDIR /home/ubuntu/isciml
CMD ["/bin/bash"]


