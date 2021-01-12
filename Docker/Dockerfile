FROM centos:centos7

#### ENV Variables For Packages ####
ENV PEGASUS_VERSION "pegasus-5.0.0"
ENV PEGASUS_VERSION_NUM "5.0.0"
ENV BOSCO_VERSION_NUM "1.2.12"

#### Update Packages ####
RUN yum -y update

#### Install basic packages ####
RUN yum -y install which java-1.8.0-openjdk-devel sudo mysql-devel postgresql-devel epel-release vim git python python3 openssh-clients libgomp rsync perl perl-Data-Dumper

RUN pip3 install --upgrade pip && pip3 install pyyaml gitpython

#### Add pegasus user ####
RUN useradd -s /bin/bash/ -m pegasus && chown -R pegasus:pegasus /home/pegasus

ENV LC_ALL "en_US.utf-8"
ENV LANG "en_US.utf-8"

#### Install Pegasus from tarball ####
RUN curl -o /opt/${PEGASUS_VERSION}.tar.gz http://download.pegasus.isi.edu/pegasus/${PEGASUS_VERSION_NUM}/pegasus-binary-${PEGASUS_VERSION_NUM}-x86_64_rhel_7.tar.gz && \
    tar -xzvf /opt/${PEGASUS_VERSION}.tar.gz -C /opt && \
    rm /opt/${PEGASUS_VERSION}.tar.gz && \
    chown pegasus:pegasus -R /opt/${PEGASUS_VERSION} &&\
    chmod 755 -R /opt/${PEGASUS_VERSION}

ENV PATH "/opt/${PEGASUS_VERSION}/bin:$PATH"
ENV PYTHONPATH "/opt/${PEGASUS_VERSION}/lib64/python3.6/site-packages:/opt/${PEGASUS_VERSION}/lib64/pegasus/externals/python:$PYTHONPATH"
ENV PERL5LIB "/opt/${PEGASUS_VERSION}/lib64/pegasus/perl:$PERL5LIB"

#### Install and configure BOSCO ####
RUN curl -o /opt/boscoinstaller.tar.gz ftp://ftp.cs.wisc.edu/condor/bosco/${BOSCO_VERSION_NUM}/boscoinstaller.tar.gz && \
    tar -xzvf /opt/boscoinstaller.tar.gz -C /opt && \
    /opt/boscoinstaller --prefix=/opt/bosco --owner=pegasus && \
    rm /opt/boscoinstaller && rm /opt/boscoinstaller.tar.gz && \
    chown pegasus:pegasus -R /opt/bosco

#### Copy Slurm cluster patch ####
COPY slurm_cluster_patch.tar.gz /opt/slurm_cluster_patch.tar.gz
RUN mkdir /opt/slurm_cluster_patch && \
    tar -C /opt/slurm_cluster_patch -xzvf /opt/slurm_cluster_patch.tar.gz && \
    rm /opt/slurm_cluster_patch.tar.gz && \
    chown pegasus:pegasus -R /opt/slurm_cluster_patch

#### Comment out copy of key to authorized keys ####
RUN for lnum in {829..834}; do sed -i "${lnum}s/\(.*\)/#\1/" /opt/bosco/bin/bosco_cluster; done

#### Bosco Default Config ####
RUN mv /opt/bosco/local.* /opt/bosco/local && \
    sed -i "s/^LOCAL_DIR.*/LOCAL_DIR = \/opt\/bosco\/local/" /opt/bosco/etc/condor_config && \
    sed -i "s/^LOCAL_CONFIG_FILE.*/LOCAL_CONFIG_FILE = \/opt\/bosco\/local\/condor_config.local/" /opt/bosco/etc/condor_config && \
    echo "HOSTALLOW_WRITE=*" >> /opt/bosco/local/condor_config.local

#### Create the entrypoint ####
RUN echo "#!/bin/bash" > /opt/entrypoint.sh && \
    echo "sed -i \"s/CONDOR_HOST.*/CONDOR_HOST = \$HOSTNAME/\" /opt/bosco/etc/condor_config" >> /opt/entrypoint.sh && \
    echo "sed -i \"s/COLLECTOR_NAME.*/COLLECTOR_NAME = Personal Condor at \$HOSTNAME/\" /opt/bosco/etc/condor_config" >> /opt/entrypoint.sh && \
    echo "source /opt/bosco/bosco_setenv" >> /opt/entrypoint.sh && \
    echo "bosco_start" >> /opt/entrypoint.sh && \
    echo "while true; do sleep 60; done" >> /opt/entrypoint.sh && \
    chown pegasus:pegasus /opt/entrypoint.sh && \
    chmod 755 /opt/entrypoint.sh

USER pegasus

ENV HOME "/home/pegasus"
WORKDIR /home/pegasus

ENTRYPOINT [ "/opt/entrypoint.sh" ]
