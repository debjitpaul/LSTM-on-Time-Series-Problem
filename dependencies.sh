#!/bin/bash
##install python

sudo apt-get update
sudo apt-get install python3.6


##install pip

sudo apt-get install python-pip python-dev build-essential 
sudo pip install --upgrade pip 

##install numpy

sudo python3 -m pip install --upgrade numpy


##install keras

sudo python3 -m pip install --upgrade keras

##install tensorflow

sudo python3 -m pip install --upgrade tensorflow


