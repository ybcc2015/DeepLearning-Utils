#!/bin/bash

DIR=$1
find $DIR -name '*_gt.json' | wc -l

