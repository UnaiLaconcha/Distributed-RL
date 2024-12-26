#!/bin/bash

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Path to the hosts file in the project
HOSTS_FILE="$SCRIPT_DIR/hosts"

# Check if the script is run with sudo
if [ "$EUID" -ne 0 ]; then
  echo "Please run this script with sudo"
  exit 1
fi

# Backup the original /etc/hosts file
cp /etc/hosts /etc/hosts.bak

# Remove any existing entries for hadoop-master and hadoop-worker
sed -i '/hadoop-master/d' /etc/hosts
sed -i '/hadoop-worker/d' /etc/hosts

# Add the new entries from the project's hosts file
while read -r line; do
  if [[ $line =~ ^[0-9] ]]; then
    echo "$line" >> /etc/hosts
  fi
done < "$HOSTS_FILE"

echo "Hosts file updated successfully."