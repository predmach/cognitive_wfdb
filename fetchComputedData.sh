#!/bin/sh
wget https://s3.amazonaws.com/helios-wfdb-precompute/data.tar.gz
tar -xzf data.tar.gz
wget https://s3.amazonaws.com/helios-wfdb-precompute/cached_eq_ml_data.hdf
