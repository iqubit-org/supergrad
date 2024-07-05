import argparse
import psutil
import time
import os
import sys
import signal
import subprocess
import platform
from threading import Thread


class RecordMaxMemoryUsage(Thread):
    def __init__(self, pid):
        super().__init__()
        self.pid = pid
        self.ps_process = psutil.Process(pid)
        self.max_mem = 0
        self.stop = False

    def run(self):
        while self.ps_process.is_running() and not self.stop:
            try:
                # Get the memory info
                mem_info = self.ps_process.memory_info()

                # Update max memory usage
                self.max_mem = max(self.max_mem, mem_info.rss)

                # Sleep for a while
                time.sleep(0.01)

            except psutil.NoSuchProcess:
                break

    def get_max_mem(self):
        return self.max_mem

    def set_stop(self):
        self.stop = True


def max_memory_usage(pid):
    t0 = RecordMaxMemoryUsage(pid=pid)
    t0.start()

    # Wait input
    input()

    t0.set_stop()
    t0.join()

    # Print the max memory usage
    print(t0.max_mem)


def trace_max_memory_usage(func, pid):

    def wrapper(*args, **kwargs):
        process = subprocess.Popen(
            ['python', __file__, str(pid)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)

        output = func(*args, **kwargs)
        # Stop the memory profiling process
        # Get the stdout and stderr
        stdout_data, stderr_data = process.communicate(input=b"stop\n")
        # delete \n in the end of the stdout
        stdout_data = stdout_data.decode('utf-8').strip()
        return output, int(stdout_data)

    return wrapper


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Memory profiling.')
    parser.add_argument('pid',
                        type=int,
                        help='the PID of the process to monitor')

    args = parser.parse_args()

    # Call the function with the PID from the command line
    max_memory_usage(args.pid)
