# Ansible script for deploying Hadoop instaces at AWS EC2

1. On a root terminal install git and python3.11-venv: 

```
$ su root
$ apt update
$ apt install git python3.11-venv
```

1. Set AWS credentials at ~/.aws/credentials
2. Download SSH private key at ~/.ssh/vockey.pem
3. Set the proper permissions for the key: `chmod og-rwx ~/.ssh/vockey.pem`
4. Create a Python virtual environment: `python3 -m venv ~/ansible`
5. Activate the virtual environment: `source ~/ansible/bin/activate`
6. Clone the repository: `git clone https://github.com/memaldi/hadoop-ansible-ec2`
6. Access to `hadoop-ansible-ec2` directory: `cd hadoop-ansible-ec2`
4. Install requirements: `pip install -r requirements.txt`
6. Install Hadoop and Spark at AWS: `ansible-playbook -i inventory.aws_ec2.yml --key-file=~/.ssh/vockey.pem --user ec2-user create-instances.yml`
7. To access to HDFS NameNode UI, you must create another SSH tunnel: `ssh -i ~/.ssh/vockey.pem -N -L 50070:<master-node-public-name>:50070 ec2-user@<master-node-public-name>`, e.g., `ssh -i ~/.ssh/vockey.pem -N -L 50070:ec2-44-192-103-94.compute-1.amazonaws.com:50070 ec2-user@ec2-44-192-103-94.compute-1.amazonaws.com`.
8. To access to YARN ResourceManager UI, you must create a SSH tunnel: `ssh -i ~/.ssh/vockey.pem -N -L 8088:<master-node-public-name>:8088 ec2-user@<master-node-public-name>`, e.g., `ssh -i ~/.ssh/vockey.pem -N -L 8088:ec2-44-192-103-94.compute-1.amazonaws.com:8088 ec2-user@ec2-44-192-103-94.compute-1.amazonaws.com`.


## EN CASO DE NECESITAR AGREGAR MÁS NODOS

* Modificar workers en hadoop-master
* Modificar hosts.template