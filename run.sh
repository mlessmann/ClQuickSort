#!/bin/bash

if [ $# -eq 0 ]
  then
    echo "Usage: run.sh <input-size> <lower rng-bound> <upper rng-bound>"
    echo "Running with default parameters."
fi

mkdir build &> /dev/null
cd build
echo "Running cmake..."
cmake ../ClQuickSort &> /dev/null
echo "Running make..."
make &> /dev/null
cp -s ../ClQuickSort/QuickSort.cl QuickSort.cl &> /dev/null
./Assignment $1 $2 $3
