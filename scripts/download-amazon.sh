#!/usr/bin/env bash

wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Video_Games_5.json.gz -O ../non_versioned/reviews_Video_Games_5.json.gz
cd ../non_versioned/
gunzip reviews_Video_Games_5.json.gz
