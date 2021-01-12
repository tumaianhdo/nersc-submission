#!/usr/bin/env bash

NERSC_SSH_PROXY=$HOME/helpers/sshproxy.sh

#### Create ssh key for BOSCO using sshproxy ####
if [ -z $NERSC_USER ]; then
   echo "NERSC user isn't specified in the config"
   exit
fi

if [ -f $NERSC_SSH_PROXY ]; then
   if [ -z $NERSC_SSH_SCOPE ]; then
      $NERSC_SSH_PROXY -u $NERSC_USER -o $HOME/.ssh/bosco_key.rsa > $HOME/.ssh/bosco_key.info
      cat $HOME/.ssh/bosco_key.info
   else
      $NERSC_SSH_PROXY -u $NERSC_USER -s $NERSC_SSH_SCOPE -o $HOME/.ssh/bosco_key.rsa > $HOME/.ssh/bosco_key.info
      cat $HOME/.ssh/bosco_key.info
   fi
else
   sftp $NERSC_USER@cori.nersc.gov:/project/projectdirs/mfa/NERSC-MFA/sshproxy.sh $NERSC_SSH_PROXY
   if [ -z $NERSC_SSH_SCOPE ]; then
      $NERSC_SSH_PROXY -u $NERSC_USER -o $HOME/.ssh/bosco_key.rsa > $HOME/.ssh/bosco_key.info
      cat $HOME/.ssh/bosco_key.info
   else
      $NERSC_SSH_PROXY -u $NERSC_USER -s $NERSC_SSH_SCOPE -o $HOME/.ssh/bosco_key.rsa > $HOME/.ssh/bosco_key.info
      cat $HOME/.ssh/bosco_key.info
   fi
fi

#### Check if bosco_key.rsa exists in $HOME/.ssh ####
if [ -f $HOME/.ssh/bosco_key.rsa ]; then
cat <<EOF >> $HOME/.ssh/config
Host cori*.nersc.gov
    User $NERSC_USER
    IdentityFile $HOME/.ssh/bosco_key.rsa
EOF
else
   echo "BOSCO ssh key is not available"
   exit
fi

#### start bosco ####
#bosco_start

#### register nersc endpoint ####
bosco_cluster --platform RH6 --add $NERSC_USER@cori.nersc.gov slurm

#### stop bosco ####
#bosco_stop

#### Install Pegasus glite attributes ####
#### Install openssl libraries ####
ssh -i $HOME/.ssh/bosco_key.rsa $NERSC_USER@cori.nersc.gov <<EOF
$NERSC_PEGASUS_HOME/bin/pegasus-configure-glite $NERSC_USER_HOME/bosco/glite
ln -s /global/common/software/m2187/shared_libraries/openssl/lib/libcrypto.so.1.0.0 $NERSC_USER_HOME/bosco/glite/lib/libcrypto.so.10
ln -s /global/common/software/m2187/shared_libraries/openssl/lib/libssl.so.1.0.0 $NERSC_USER_HOME/bosco/glite/lib/libssl.so.10
sed -i "s/supported_lrms=.*/supported_lrms=slurm/" $NERSC_USER_HOME/bosco/glite/etc/batch_gahp.config
EOF

#### Install edited glite scripts ####
sftp -i $HOME/.ssh/bosco_key.rsa $NERSC_USER@cori.nersc.gov <<EOF
put /opt/slurm_cluster_patch/* $NERSC_USER_HOME/bosco/glite/bin
EOF
