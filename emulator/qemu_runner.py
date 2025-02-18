import subprocess
import threading
import time
import os
import signal
from queue import Queue, Empty


def _enqueue_output(stream, queue):
    """Helper function to enqueue lines from a stream."""
    try:
        while True:
            line = stream.readline()
            if line:
                queue.put(line)
            else:
                break
    except ValueError: # Raised if stream was closed or has no output.
        pass


def run_qemu_process_async(qemu_command, log_file='qemu_log.txt'):
    """
    Runs a QEMU process asynchronously using threads, capturing output, and returning
    process info and queues for output and errors.

    Args:
        qemu_command (list): The QEMU command as a list of strings.
        log_file (str): The path to the log file to save the output of QEMU.

    Returns:
        tuple: A tuple containing:
            - process (subprocess.Popen): The QEMU process object.
            - stdout_queue (queue.Queue): Queue for the standard output from QEMU.
            - stderr_queue (queue.Queue): Queue for the standard error output from QEMU.
    """

    stdout_queue = Queue()
    stderr_queue = Queue()
    process = None
    try:
        with open(log_file, 'w') as log:
          process = subprocess.Popen(qemu_command, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
          print(f"QEMU process started with PID: {process.pid}")

          stdout_thread = threading.Thread(target=_enqueue_output,
                                           args=(process.stdout, stdout_queue), daemon=True)
          stderr_thread = threading.Thread(target=_enqueue_output,
                                           args=(process.stderr, stderr_queue), daemon=True)
          stdout_thread.start()
          stderr_thread.start()

    except FileNotFoundError:
      print("File not found. Ensure that QEMU is installed and qemu-system-i386 is in path.")
    except Exception as e:
        print(f"An unexpected error has occurred: {e}")
    return process, stdout_queue, stderr_queue


def get_qemu_output(queue, timeout=0.1, max_lines=None):
    """
    Retrieves the output from a queue.

    Args:
        queue (queue.Queue): The queue to retrieve output from.
        timeout (float): The timeout for waiting for a new message.
        max_lines (int): The maximum lines that should be retrieved.

    Returns:
        str: The collected output from the queue.
    """
    output = ""
    lines_read = 0
    while True:
        try:
            line = queue.get(timeout=timeout)
            output += line
            lines_read += 1
            if max_lines and lines_read >= max_lines:
              break
        except Empty:
            break
    return output


def stop_qemu_process(process):
    """
    Stops the QEMU process gracefully, and if it doesn't stop after a timeout
    period, force stops the process

    Args:
      process: The QEMU process object.
    """
    try:
        if process is not None and process.poll() is None:
            print(f"Attempting to terminate QEMU process with PID {process.pid} gracefully.")
            os.kill(process.pid, signal.SIGINT) # sending a SIGINT to emulate Ctrl+C
            time_waited = 0
            while process.poll() is None and time_waited < 10:
              time.sleep(1)
              time_waited +=1
            if process.poll() is None:
              print(f"QEMU process with PID {process.pid} didn't terminate gracefully. Forcing stop")
              process.terminate()
              process.wait()
              print(f"QEMU process with PID {process.pid} terminated.")
        else:
            print(f"QEMU process either already stopped or not yet started.")
    except Exception as e:
        print(f"An error has occurred when trying to stop the process {e}")
