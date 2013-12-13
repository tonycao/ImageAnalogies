#!/bin/sh
export LIB_GCC=/usr/lib/x86_64-linux-gnu/gcc/x86_64-linux-gnu/4.5/
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/gcc/x86_64-linux-gnu/4.5/:/
export DYLD_INSERT_LIBRARIES=$LIB_GCC/libgfortran.so:$LIB_GCC/libgcc_s.so:$LIB_GCC/libstdc++.so:$LIB_GCC/libgomp.so
matlab $* -r "addpath('./build/'); addpath('./test_release'); setenv('MKL_NUM_THREADS','1'); setenv('MKL_SERIAL','YES');"
