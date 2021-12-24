
# Copyright (c) HP-NTU Digital Manufacturing Corporate Lab, Nanyang Technological University, Singapore.
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.

import serial
import codecs
import time
from collections import OrderedDict


class SerialBlueTooth:
    def __init__(self, port):
        self.port = port
        self.additional = ["description", "manufacturer", "product",
                           "serial_number", "vid", "pid"]
        self.serial = None

    def connect(self):
        if self.serial is None:
            self.serial = serial.Serial(port=self.port, baudrate=9600,
                                        timeout=3, write_timeout=0)

    def read(self):
        self.serial.write(bytes.fromhex("f0"))
        data_r = self.serial.read(130)

        if len(data_r) < 130:
            return None

        data_r = codecs.encode(data_r, "hex").decode("utf-8")
        result_r = OrderedDict()

        result_r = int("0x" + data_r[12] + data_r[13] + data_r[14] + data_r[15] +
                                data_r[16] + data_r[17] + data_r[18] + data_r[19], 0) / 1000

        return result_r

    def disconnect(self):
        if self.serial:
            self.serial.close()

