#!/bin/bash

username="mazeller"
jupyter_port_client=8889

# Mount network path
# include uid, gid to grant write access to user, which wouldn't otherwise be allowed since mounting in /media
echo "Mounting network paths..."
echo "Reminder: are you connected to the VPN ?"
echo "$(findmnt -M "/media/epfl-nas/")"
if [[ -n $(findmnt -M "/media/epfl-nas/") ]]; then
  echo "not mounted"
  #sudo mount.cifs //files3.epfl.ch/data/${username} /media/epfl-nas/ -o user=${username},domain=INTRANET,uid=$(id -u),gid=$(id -g)
fi
if [[ -n $(findmnt -M "/media/cardio-project") ]]; then
  echo "not mounted"
  #sudo mount.cifs //sti1files.epfl.ch/cardio-project /media/cardio-project/ -o user=${username},domain=INTRANET
fi

# Enable port forwarding for Jupyter
echo "Enabling port forwarding ${jupyter_port_client}:8888"
# Find name of jupyter pod
jupyter_pod_name=$(kubectl get pods | grep jupyter | cut -d ' ' -f 1)
kubectl port-forward --namespace sti-cluster-lts4-${username} "${jupyter_pod_name}" ${jupyter_port_client}:8888
