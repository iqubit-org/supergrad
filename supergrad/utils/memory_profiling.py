import argparse
import psutil
import time
import os
import sys
import signal
import subprocess


def max_memory_usage(pid):
    # Create a psutil process object
    ps_process = psutil.Process(pid)

    # Initialize max memory usage as 0
    max_mem = 0

    def handle_sigint(sig, frame):
        # Print the max memory usage
        print(max_mem)
        sys.exit(0)

    # Set the SIGINT handler
    signal.signal(signal.SIGINT, handle_sigint)

    # While the process is running
    while ps_process.is_running():
        try:
            # Get the memory info
            mem_info = ps_process.memory_info()

            # Update max memory usage
            max_mem = max(max_mem, mem_info.rss)

            # Sleep for a while
            time.sleep(0.01)

        except psutil.NoSuchProcess:
            break

    # Print the max memory usage
    print(max_mem)


def trace_max_memory_usage(func, pid):

    def wrapper(*args, **kwargs):
        process = subprocess.Popen(
            ['python', __file__, str(pid)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)

        output = func(*args, **kwargs)
        # Stop the memory profiling process
        os.kill(process.pid, signal.SIGINT)
        # Get the stdout and stderr
        stdout_data, stderr_data = process.communicate()
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
