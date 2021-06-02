#FROM tensorflow/tensorflow:2.0.1-gpu
FROM tensorflow/tensorflow:2.3.0

RUN groupadd --system processors && useradd --system --create-home --shell /bin/bash --gid processors user && (echo user ; echo user) | passwd user

RUN pip install keras
RUN pip install pandas
RUN pip install -U scikit-learn
RUN pip install -U scikit-image
RUN pip install h5py

RUN pip install SimpleITK
RUN pip install pydicom
RUN pip install pypng
RUN pip install seaborn
RUN pip install tqdm



## COPY CODE AND EXPERIMENT DATA ##

USER root
WORKDIR /home
COPY process.py /home/user/
COPY preprocess.py /home/user/
COPY image_sorter.hdf5 /home/user/
COPY opencxr /home/user/opencxr

ENV PYTHONPATH "${PYTHONPATH}:/home/user/opencxr/"


USER user
WORKDIR /home/user

VOLUME /input
VOLUME /output

LABEL processor.cpus=2
LABEL processor.cpu.capabilities=
LABEL processor.memory=15G
LABEL processor.gpu_count=1
LABEL processor.gpu.compute_capability=
LABEL processor.gpu.memory=8G

ENTRYPOINT /bin/sh -c 'python  /home/user/process.py /input /output' 
