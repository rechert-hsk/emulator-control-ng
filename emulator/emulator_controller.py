from emulator.qemu_runner import run_qemu_process_async, get_qemu_output, stop_qemu_process
import time
from emulator.qemu_utils import create_snapshot, restore_snapshot, delete_snapshot, list_snapshots
from emulator.computer import QemuComputer

class EmulatorController:
    def __init__(self, computer_config: QemuComputer):
        self.process = None
        self.stdout_queue = None
        self.stderr_queue = None
        self.computer_config = computer_config
        self.socket = computer_config.set_monitor_socket() # new random path for monitor socket

    def run(self):
        config = self.computer_config.to_qemu_args()
        self.process, self.stdout_queue, self.stderr_queue = run_qemu_process_async(config)
        print("QEMU is running in the background...")
        time.sleep(1)

    def stop(self):  
        # Collect remaining output
        stdout = get_qemu_output(self.stdout_queue)
        print("QEMU remaining standard output:")
        print(stdout)

        stderr = get_qemu_output(self.stderr_queue)
        print("QEMU remaining standard error:")
        print(stderr)

        if self.process:
            stop_qemu_process(self.process)

        return stdout, stderr
    
    def create_snapshot(self, snapshot_name):
        create_snapshot(snapshot_name, self.socket)

    def restore_snapshot(self, snapshot_name):
        restore_snapshot(snapshot_name, self.socket)

    def delete_snapshot(self, snapshot_name):    
        delete_snapshot(snapshot_name, self.socket)

    def list_snapshots(self):
        list_snapshots(self.socket)
    
   