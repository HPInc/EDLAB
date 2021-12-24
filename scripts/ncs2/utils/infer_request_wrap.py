
# Copyright (c) HP-NTU Digital Manufacturing Corporate Lab, Nanyang Technological University, Singapore.
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.

from ctypes import *
from datetime import datetime
import threading
import logging as log

class InferReqWrap:
    def __init__(self, request, id, callbackQueue):
        self.id = id
        self.request = request
        self.request.set_completion_callback(self.callback, self.id)
        self.callbackQueue = callbackQueue

    def callback(self, statusCode, userdata):
        if (userdata != self.id):
            print("Request ID {} does not correspond to user data {}".format(self.id, userdata))
        elif statusCode != 0:
            print("Request {} failed with status code {}".format(self.id, statusCode))
        # log.info('Request {} completed!'.format(self.id))
        self.output.append(self.request.outputs)
        self.callbackQueue(self.id, self.request.latency)
        # return self.request.outputs[self.out_blob]

    def startAsync(self, input_data, output_data):
        self.request.async_infer(input_data)
        self.output = output_data 

    def infer(self, input_data):
        self.request.infer(input_data)
        # log.info('Request {} completed!'.format(self.id))
        self.callbackQueue(self.id, self.request.latency)
        return self.request.outputs

class InferRequestsQueue:
    def __init__(self, requests):
      self.idleIds = []
      self.requests = []
      self.times = []
      for id in range(0, len(requests)):
          self.requests.append(InferReqWrap(requests[id], id, self.putIdleRequest))
          self.idleIds.append(id)
      self.startTime = datetime.max
      self.endTime = datetime.min
      self.cv = threading.Condition()

    def resetTimes(self):
      self.times.clear()

    def getDurationInSeconds(self):
      return (self.endTime - self.startTime).total_seconds()

    def putIdleRequest(self, id, latency):
      self.cv.acquire()
      self.times.append(latency)
      self.idleIds.append(id)
      self.endTime = max(self.endTime, datetime.now())
      self.cv.notify()
      self.cv.release()

    def getIdleRequest(self):
        self.cv.acquire()
        while len(self.idleIds) == 0:
            self.cv.wait()
        id = self.idleIds.pop(0)
        self.startTime = min(datetime.now(), self.startTime)
        self.cv.release()
        return self.requests[id]

    def waitAll(self):
        self.cv.acquire()
        while len(self.idleIds) != len(self.requests):
            self.cv.wait()
        self.cv.release()
