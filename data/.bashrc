source /opt/bosco/bosco_setenv
source $HOME/config/nersc.conf

if [ -f $HOME/.ssh/bosco_key.info ]; then
  expiration=$(cat $HOME/.ssh/bosco_key.info)
  expiration=${expiration##* }
  time_diff=$(( $(date '+%s' --date $expiration) - $(date '+%s') ))
  if [ "$time_diff" -lt "0" ]; then
    echo -e "###########################################################################################\n\
Your NERSC key expired on $expiration.\n\
Please retrieve a new one by executing $HOME/helpers/renew-nersc-key.sh.\n\
###########################################################################################\n"
  elif [ "$time_diff" -lt "172800" ]; then
    echo -e "###########################################################################################\n\
Your NERSC key is expiring on $expiration.\n\
You may want to retrieve a new one by executing $HOME/helpers/renew-nersc-key.sh.\n\
###########################################################################################\n"
  fi
else
  echo -e "############################################################################\n\
It seems that you haven't initialized NERSC to accept workflow submissions.\n\
Please execute $HOME/helpers/initialize-nersc.sh\n\
###########################################################################################\n"
fi
