#!/bin/sh
dpkg -s "gcc" >/dev/null 2>&1 && {
    echo "gcc is installed."
} || {
  echo "gcc is not installed."
  sudo apt-get install gcc
}

dpkg -s "curl-config" >/dev/null 2>&1 && {
    echo "curl-config is installed."
} || {
  echo "curl-config is not installed."
  sudo apt-get install libcurl4-openssl-dev
}

file="/usr/include/expat.h"
if [ -f "$file" ]
then
	echo "$file found."
else
	echo "$file not found."
  sudo apt-get install libexpat1-dev
  #Install expat lib
fi

rm wfdb.tar.gz*

rm -rf wfdbsrc

wget https://www.physionet.org/physiotools/wfdb.tar.gz

mkdir wfdbsrc

tar xf wfdb.tar.gz -C wfdbsrc

cd wfdbsrc/wfdb*

./configure

make install

make check

#Install Packages for WDFC code
#While installation if it gives error of freetype package issue try "sudo apt-get install libfreetype6-dev" and re-run below mentioned command
#If any error related to scikit try pip install scikit-image
#if any error related to hdf5 try "sudo apt-get install libhdf5-dev"
# if any error related to libreadline try "sudo apt-get install libreadline-dev" and "sudo apt-get install lib32ncurses5-dev"
# if any error related to libxml2 try "sudo apt-get install -y python-lxml                                     "
#sudo pip install --upgrade -r requirements.txt
#pip install numpy
#pip install pandas
#pip install bs4
#pip install lxml
#pip install tables

pip install -r requirements.txt
