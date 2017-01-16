#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=examples/mnist_hdfs/lenet_solver.prototxt $@ \
    #--weights hdfs://192.168.184.39:8020/examples/mnist/lenet_iter_100.caffemodel
    #--weights examples/mnist/lenet_iter_20.caffemodel
