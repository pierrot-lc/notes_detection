#!/bin/sh

mkdir -p data
cd data  || exit  # Exit if `cd` fails.

curl https://zenodo.org/record/5120004/files/musicnet_midis.tar.gz -o musicnet_midis.tar.gz
curl https://zenodo.org/record/5120004/files/musicnet_metadata.csv -o musicnet_metadata.csv
curl --parallel https://zenodo.org/record/5120004/files/musicnet.tar.gz -o musicnet.tar.gz

tar -xf musicnet_midis.tar.gz
tar -xf musicnet.tar.gz
