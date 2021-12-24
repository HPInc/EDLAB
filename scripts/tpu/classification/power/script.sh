#!/bin/bash

# Copyright (c) HP-NTU Digital Manufacturing Corporate Lab, Nanyang Technological University, Singapore.
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.

sudo rfcomm bind /dev/rfcomm0 $1 
sudo chmod 666 /dev/rfcomm0
