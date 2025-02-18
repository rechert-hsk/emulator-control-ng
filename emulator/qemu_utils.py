import socket
import time
import re
import select

def send_monitor_command(command, sock_path ="/tmp/qemu_monitor.sock"):
    """
    Send a command to QEMU monitor through Unix socket
    
    Args:
        command (str): The command to send to QEMU monitor
        sock_path (str): The path to the Unix socket
        
    Returns:
        str: Response from the QEMU monitor
    """
    try:
        
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(sock_path)
        command = (command + '\n').encode('utf-8')
        sock.send(command)
        time.sleep(0.1)
        response = sock.recv(4096).decode('utf-8')
        sock.close()
        
        return response
        
    except Exception as e:
        print(f"Error sending monitor command: {e}")
        return None


def create_snapshot(snapshot_name, sock_path ="/tmp/qemu_monitor.sock"):
    """
    Create a snapshot of the running VM with the given name
    """
    send_monitor_command(f"savevm {snapshot_name}", sock_path)
    print(f"Created snapshot: {snapshot_name}")

def restore_snapshot(snapshot_name, sock_path ="/tmp/qemu_monitor.sock"):
    """
    Restore the VM to a previously created snapshot
    """
    send_monitor_command(f"loadvm {snapshot_name}", sock_path)
    print(f"Restored snapshot: {snapshot_name}")

def list_snapshots(sock_path ="/tmp/qemu_monitor.sock"):
    """
    List all available snapshots
    """
    response = send_monitor_command("info snapshots")
    print("Available snapshots:")
    print(response)

def delete_snapshot(snapshot_name, sock_path ="/tmp/qemu_monitor.sock"):
    """
    Delete a snapshot with the given name
    """
    send_monitor_command(f"delvm {snapshot_name}", sock_path)
    print(f"Deleted snapshot: {snapshot_name}")


import subprocess
import os


def create_qcow2_disk_image(image_path, size_gb):
    """
    Creates an empty QCOW2 disk image using qemu-img.

    Args:
        image_path (str): The path where the disk image file will be created.
        size_gb (int): The size of the disk image in gigabytes.

    Returns:
        bool: True if the image was created successfully, False otherwise.
    """

    if os.path.exists(image_path):
        print(f"Error: File '{image_path}' already exists.")
        return False

    try:
        size_str = f"{size_gb}G"
        command = ["qemu-img", "create", "-f", "qcow2", image_path, size_str]
        subprocess.run(command, check=True, capture_output=True)
        print(f"Successfully created QCOW2 disk image at '{image_path}' of size {size_gb} GB")
        return True

    except FileNotFoundError:
      print("File not found. Ensure that QEMU is installed and qemu-img is in path.")
      return False
    except subprocess.CalledProcessError as e:
        print(f"Error creating QCOW2 disk image: {e.stderr.decode()}")
        return False
    except Exception as e:
      print(f"An unexpected error has occurred: {e}")
      return False
