from queue import Queue
from threading import Thread, Lock
from os import path
import gzip
import pickle
import os
import time
class InferenceOutputStreamer:
    def __init__(self, output_path, cache_size=10):
        self._cache_size = cache_size
        self._cache = list() # It will store `cache` number of elements
        self._closed = False
        self._initialized = False
        self._output_path = output_path
        if(not os.path.exists(self._output_path)):
            print('\n Creating test directory')
            os.mkdir(self._output_path)
        self._num_last_file = 0
        self._all_files = []

    def _worker(self):
        while True:
            output_cache = self._queue.get()
            file_path = path.join(self._output_path, 'data_'+ str(self._num_last_file)+ '.bin')
            with gzip.open(file_path, 'wb') as f:
                pickle.dump(output_cache, f)
            self._all_files.append(file_path)
            self._num_last_file += 1
            self._queue.task_done()

    def start_thread(self):
        self._queue = Queue() # When we are out of `cache` number of elements in cache, push it to queue, so it could be written
        self._t = Thread(target=self._worker)
        self._t.setDaemon(True)
        self._t.start()
        self._initialized=True

    def add(self, sample):
        if self._closed or (not self._initialized):
            RuntimeError("Attempting to use a closed or an unopened streamer")
        while(self._queue.qsize() > 20):
            time.sleep(0.05)
            #timer.sleep(0.01)

        self._cache.append(sample)
        n = len(self._cache)
        if n == self._cache_size:
            print('\nAdding to queue')
            self._queue.put(self._cache)
            self._cache = list()

    def close(self):
        self._queue.put(self._cache)
        self._cache = list()

        print("X", self._queue.qsize())

        self._queue.join()

        with open(path.join(self._output_path, 'inference_output_files.txt'), 'w') as f:
            for file_path in self._all_files:
                f.write("%s\n" % file_path)

        self._closed = True
