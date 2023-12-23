FROM ubuntu:22.04 

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y sudo build-essential ca-certificates coreutils curl environment-modules \ 
    gfortran git gpg lsb-release python3 python3-distutils python3-venv python3-pip \
    ffmpeg libsm6 libxext6 \
    emacs unzip zip libopenmpi-dev && \
    rm -rf /var/lib/apt/lists/*


RUN pip install poetry numpy mpi4py
COPY ./ /isciml/
ENV POETRY_VIRTUALENVS_CREATE=false
RUN cd /isciml && \
    poetry install && \
    cd /isciml/lib && \
    python3 -m numpy.f2py -c calc_and_mig_all_rx.f90 gtet.f90 gfacet.f90 ggfacet.f90 gzfacet.f90 check_divzero1.f90 check_divzero2.f90 -m adjoint && \
    python3 -m numpy.f2py -c calc_all_rx_multi_k.f90 gtet.f90 gfacet.f90 ggfacet.f90 gzfacet.f90 check_divzero1.f90 check_divzero2.f90 -m forward && \
    rm *.f90 

ENV PYTHONPATH="${PYTHONPATH}:/isciml/lib"

CMD ["/bin/bash"]


