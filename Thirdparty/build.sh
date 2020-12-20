#!/bin/bash

printf "Installing TEASER++:"
git clone https://github.com/MIT-SPARK/TEASER-plusplus.git
cd TEASER-plusplus && mkdir build && cd build
cmake -DTEASERPP_PYTHON_VERSION=3.7 .. && make teaserpp_python
cd python && pip install .
# cd .. && cd examples/teaser_cpp_ply && mkdir build && cd build
# cmake .. && make
cd ../../

printf "Installing Pyfbow:"
cd pyfbow/install/fbow
mkdir build && cd build
cmake -DBUILD_SHARED_LIBS=OFF \
      -DCMAKE_CXX_FLAGS="-fPIC" \
      -DCMAKE_C_FLAGS="-fPIC" ..

make && make install

cd ../../..

mkdir build
cd build

cmake ../src
make
