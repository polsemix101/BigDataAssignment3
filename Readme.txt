## Move Files to ECS System

Move part1.py, kdd.data, and SetupSparkClasspath.sh to
barretts@ecs.vuw.ac.nz
$ scp part1.py <username>@barretts.ecs.vuw.ac.nz:
etc..

## Access VUW Hadoop cluster

ssh into barretts using your ecs account
$ ssh <username>@barretts.ecs.vuw.ac.nz

ssh into one of the Hadoop nodes 
$ ssh co246a-5
(last number can be 1-8)

## Setup Hadoop and Spark

configure Hadoop and Spark
$ source SetupSparkClasspath.sh

create directory for input and output datasets
$ hadoop fs -mkdir /user/<username>/input /user/<username>/output

upload input data into hdfs
$ hadoop fs -put kdd.data /user/<username>/input/

## Run Spark Job

part1.py takes 2 inputs:
- path to input data
- path to output folder

$ spark-submit --master yarn --deploy-mode cluster part1.py
/user/<username>/input/kdd.data /user/<username>/output

## Retrieve Results

move from hdfs to ecs local
$ hadoop fs -copyToLocal /user/<username>/output
$ hadoop fs -rm -r /user/<username>/output

move from ECS system to desired path local pc
$ scp -r <username>@barretts.ecs.vuw.ac.nz:~/output ~/path/to/local
