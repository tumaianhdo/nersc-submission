# A Docker recipe that runs Pegasus and HTCondor and targets NERSC as the execution site

This project prepares a Docker container that can run on your local machine and submit Pegasus workflows **remotely**, using HTCondor BOSCO, to NERSC's computing systems.

The container can be found at https://hub.docker.com/r/pegasus/nersc-remote-submission.

New versions of the container will be tagged with the version of Pegasus installed in the image (e.g., pegasus/nersc-remote-submission:pegasus-5.0.0).

## Basic scripts and files

_Docker/Dockerfile_. Dockerfile used to prepare a container with Pegasus and HTCondor BOSCO, ommiting Pegasus' R support.

_docker-compose.yml_. A Docker Compose file to automate the instantiation of the container.

_data/config/nersc.conf_. Contains envuironmental variables that are relevant for your account at NERSC.

_data/helpers/initialize-nersc.sh_. This script initializes your NERSC home to accept jobs using the HTCondor BOSCO method. It retrieves an SSH key for your account and installs the BOSCO binaries under your account.

_data/helpers/renew-nersc-key.sh_. The retrieved SSH key lasts for a limited time and this script can be used to renew it.

_data/workflows_. This fodler contains Pegasus 5.0 workflow examples that can be submitted directly to NERSC.

## Prerequisites

- Install Docker on your local machine (https://docs.docker.com/get-docker/)
- Install Docker Compose on your local machine (https://docs.docker.com/compose/install/)

Step 1: Update data/config/nersc.conf
-------------------------------------
In data/config/nersc.conf update the section "ENV Variables For NERSC" with your information.

More specifically replace:
- _NERSC\_SSH\_SCOPE_, with the ssh scope specified for your account by the NERSC admins (if any, otherwise leave empty)
- _NERSC\_PROJECT_, with your project name at NERSC
- _NERSC\_USER_, with your user name at NERSC

Step 2: Start the Docker container
----------------------------------

```
docker-compose up -d
```

Step 3: Get an interactive shell to the container
-------------------------------------------------
```
docker exec -it pegasus-nersc /bin/bash
```

Step 4a: Run the initialization script
--------------------------------------
This is required only once, the first time you bring up the container. This script will ask you to enter your NERSC pass + OTP two times.
(Hint: Wait for a new OTP the second time)
```
/home/pegasus/helpers/initialize-nersc.sh
```

Step 4b: Renew your NERSC SSH Key
---------------------------------
This will retrieve a new SSH key for your account. (Hint: Monitor the login messages for the expiration date)
```
/home/pegasus/helpers/renew-nersc-key.sh
```

Step 5: Run a workflow
----------------------

```
cd /home/pegasus/workflows/sns-namd-example
./workflow_generator_shifter_remote_staging.py
./plan.sh workflow.yml
```

Deleting the Docker container
-----------------------------

```
docker-compose down
```
