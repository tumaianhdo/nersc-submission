#!/usr/bin/env bash

NERSC_SSH_PROXY=$HOME/helpers/sshproxy.sh

#### Create ssh key for BOSCO using sshproxy ####
if [ -z $NERSC_USER ]; then
   echo "NERSC user isn't specified in the config"
   exit
fi

if [ -f $NERSC_SSH_PROXY ]; then
   if [ -z $NERSC_SSH_SCOPE ]; then
      $NERSC_SSH_PROXY -u $NERSC_USER -o $HOME/.ssh/bosco_key.rsa
   else
      $NERSC_SSH_PROXY -u $NERSC_USER -s $NERSC_SSH_SCOPE -o $HOME/.ssh/bosco_key.rsa
   fi
else
   sftp $NERSC_USER@cori.nersc.gov:/project/projectdirs/mfa/NERSC-MFA/sshproxy.sh $NERSC_SSH_PROXY
   if [ -z $NERSC_SSH_SCOPE ]; then
      $NERSC_SSH_PROXY -u $NERSC_USER -o $HOME/.ssh/bosco_key.rsa
   else
      $NERSC_SSH_PROXY -u $NERSC_USER -s $NERSC_SSH_SCOPE -o $HOME/.ssh/bosco_key.rsa
   fi
fi
