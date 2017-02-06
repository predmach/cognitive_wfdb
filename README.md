# Cognitive WFDB
Load the data from the Physionet MIT-BIH ECG samples and annotations, convert the data to CSV, move the CSV into HDF5, perform training of classifers.


**Install ubuntu python support libaries**
 
- sudo apt-get install python3-pip
- sudo apt-get install python3-numpy
- # sudo apt-get install python-pandas # old version 
- sudo pip3 install --upgrade pandas # installs version 0.18 (note I perform the prior step first so it was upgrade).
- sudo pip3 install scikit-neuralnetwork
 
 

**Install wfdb**    

- mkdir wfdb
- cd wfdb
- wget https://www.physionet.org/physiotools/wfdb.tar.gz
- tar xfvz wfdb.tar.gz
- cd wfdb-10.5.24/
- ./configure
- sudo make install # installs in /usr/local/bin
- make check

 
**Run**

- ./test.py

**Output**

             precision    recall  f1-score   support

        0.0       0.99      0.99      0.99       172
        1.0       0.99      0.99      0.99       192

avg / total       0.99      0.99      0.99       364

             precision    recall  f1-score   support

        0.0       1.00      1.00      1.00       736
        1.0       1.00      1.00      1.00       716

avg / total       1.00      1.00      1.00      1452




