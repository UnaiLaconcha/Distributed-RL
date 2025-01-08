#!/bin/bash

# Solicitar los parámetros al usuario
read -p "Ingrese el tipo de instancia para el master (por defecto t3.large): " master_instance_type
master_instance_type=${master_instance_type:-t3.large}

read -p "Ingrese el tipo de instancia para el cliente (por defecto t3.large): " client_instance_type
client_instance_type=${client_instance_type:-t3.large}

read -p "Ingrese el tipo de instancia para los workers (por defecto t3.large): " worker_instance_type
worker_instance_type=${worker_instance_type:-t3.large}

read -p "Ingrese el número de nodos workers (por defecto 3): " num_workers
num_workers=${num_workers:-3}

echo "--------------------------"
echo "   CREANDO SERVIDOR"
echo "--------------------------"

if [ ! -d "../.ansible" ]; then
    echo "Creating virtual environment in ../.ansible"
    python -m venv ../.ansible
    source ../.ansible/bin/activate

    echo "Installing requirements"
    pip install -r requirements.txt

else
    echo "Activating ../.ansible"
    source ../.ansible/bin/activate
fi


# Create or overwrite the hosts.template file
echo "master-node-ip  hadoop-master" > hosts.template

# Add worker nodes
for ((i=0; i<num_workers; i++))
do
    echo "worker-$i-ip     hadoop-worker-$((i+1))" >> hosts.template
done

echo "hosts.template file has been created with $num_workers worker nodes."

# Update the workers file in hadoop-master directory
workers_file="hadoop-master/workers"
> "$workers_file"  # Clear the file

for ((i=1; i<=num_workers; i++))
do
    echo "hadoop-worker-$i" >> "$workers_file"
done

echo "hadoop-master/workers file has been updated with $num_workers worker nodes."


# Ejecutar el playbook de Ansible con los parámetros
ansible-playbook -i inventory.aws_ec2.yml --key-file=~/.ssh/vockey.pem --user ec2-user create-instances.yml \
    -e "master_instance_type=${master_instance_type}" \
    -e "worker_instance_type=${worker_instance_type}" \
    -e "client_instance_type=${client_instance_type}" \
    -e "num_workers=${num_workers}"

# Desactivar el entorno virtual
deactivate