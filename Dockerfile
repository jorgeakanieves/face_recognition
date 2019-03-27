FROM debian:jessie
MAINTAINER Jorge Nieves

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

EXPOSE 8888

# Install
RUN apt-get update -y  && \
    DEBIAN_FRONTEND=noninteractive \
        apt-get install --no-install-recommends -y -q \
            build-essential \
            python2.7       \
            python2.7-dev   \
            python-pip        && \
    \
    \
    pip install --upgrade pip virtualenv  && \
    \
    \
    apt-get clean  && \
    rm -rf /var/lib/apt/lists/*


RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

#RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
#    wget --quiet https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh -O ~/anaconda.sh && \
#    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
#    rm ~/anaconda.sh

#RUN apt-get install -y curl grep sed dpkg && \
#    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
#    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
#    dpkg -i tini.deb && \
#    rm tini.deb && \
#    apt-get clean

#RUN apt-get -y install unzip && apt-get -y install build-essential

#ENV PATH /opt/conda/bin:$PATH

RUN mkdir /face-recognition
#ADD . /face-recognition

#RUN pip install --upgrade pip
RUN pip install text_unidecode \
&& pip install matplotlib==2.1.2 \
&& pip install numpy \
&& pip install pandas==0.22.0 \
&& pip install scipy==1.0.0 \
&& pip install pillow \
&& pip install scikit-learn==0.19.1 \
&& pip install tensorflow==1.4.0 \
#RUN pip install keras==2.1.3
#RUN pip install nltk==3.2.4
#RUN pip install distance==0.1.3
#RUN pip install unidecode==0.4.21
&& pip install opencv-contrib-python \
#RUN conda install -y py-xgboost
&& python -m pip install jupyter \
&& pip install ipython && python2 -m pip install ipykernel && python2 -m ipykernel install --user

#Installing dlib (version 19.9).
#RUN wget http://dlib.net/files/dlib-19.9.zip && unzip dlib-19.9.zip && cd dlib-19.9 && python setup.py install && cd ..

WORKDIR /face-recognition

# Create a new system user
RUN useradd -ms /bin/bash jupyter

# Change to this new user
USER jupyter

CMD ["sh", "-c", "jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root"]
#CMD []
