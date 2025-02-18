from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum
import json
from pathlib import Path
import tempfile

class DriveType(Enum):
    HDD = "hdd"
    CDROM = "cdrom" 
    FLOPPY = "floppy"

class DriveInterface(Enum):
    IDE = "ide"
    FLOPPY = "floppy"

@dataclass
class Drive:
    path: Path
    drive_type: DriveType
    interface: DriveInterface = DriveInterface.IDE
    readonly: bool = False
    boot: bool = False
    
    def to_dict(self):
        return {
            "path": str(self.path),
            "drive_type": self.drive_type.value,
            "interface": self.interface.value,
            "readonly": self.readonly,
            "boot": self.boot
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            path=Path(data["path"]),
            drive_type=DriveType(data["drive_type"]),
            interface=DriveInterface(data["interface"]),
            readonly=data["readonly"],
            boot=data["boot"]
        )

    def to_qemu_args(self, index: int) -> List[str]:
        if self.drive_type == DriveType.FLOPPY:
            return ["-fda" if index == 0 else "-fdb", str(self.path)]
        
        if self.drive_type == DriveType.CDROM:
            return ["-cdrom", str(self.path)]
            
        # IDE hard drives
        device_letter = chr(ord('a') + index)
        return [f"-hd{device_letter}", str(self.path)]

@dataclass 
class QemuComputer:
    cpu: str = "pentium3"
    memory_mb: int = 1024
    drives: List[Drive] = field(default_factory=list)
    vga: str = "std"
    network_model: str = "e1000"
    use_tablet: bool = True
    enable_audio: bool = True
    vnc_display: int = 0
    monitor_socket: str = "/tmp/qemu_monitor.sock"

    def set_monitor_socket(self, socket_path: Optional[str] = None):
        if socket_path:
            self.monitor_socket = socket_path
        else:
            self.monitor_socket = tempfile.mktemp(suffix=".sock")

        return self.monitor_socket

    def add_drive(self, drive: Drive):
        # Validate drive configuration
        existing_boot = any(d.boot for d in self.drives)
        if drive.boot and existing_boot:
            raise ValueError("Only one drive can be marked as bootable")
            
        if drive.drive_type == DriveType.FLOPPY and \
           len([d for d in self.drives if d.drive_type == DriveType.FLOPPY]) >= 2:
            raise ValueError("Maximum 2 floppy drives supported")
            
        if drive.drive_type == DriveType.CDROM and \
           len([d for d in self.drives if d.drive_type == DriveType.CDROM]) >= 1:
            raise ValueError("Maximum 1 CDROM drive supported")
            
        if drive.drive_type == DriveType.HDD and \
           len([d for d in self.drives if d.drive_type == DriveType.HDD]) >= 4:
            raise ValueError("Maximum 4 IDE hard drives supported")
            
        self.drives.append(drive)

    def to_qemu_args(self) -> List[str]:
        args = [
            "qemu-system-i386",
            "-cpu", self.cpu,
            "-m", str(self.memory_mb),
            "-vga", self.vga,
            "-net", f"nic,model={self.network_model}"
        ]

        # Add drives with proper indexing per type
        hdd_index = 0
        floppy_index = 0
        
        for drive in self.drives:
            if drive.drive_type == DriveType.HDD:
                args.extend(drive.to_qemu_args(hdd_index))
                hdd_index += 1
            elif drive.drive_type == DriveType.FLOPPY:
                args.extend(drive.to_qemu_args(floppy_index))
                floppy_index += 1
            else:  # CDROM
                args.extend(drive.to_qemu_args(0))

        # Add boot order if specified
        boot_drive = next((d for d in self.drives if d.boot), None)
        if boot_drive:
            if boot_drive.drive_type == DriveType.CDROM:
                args.extend(["-boot", "d"])
            elif boot_drive.drive_type == DriveType.FLOPPY:
                args.extend(["-boot", "a"])
            elif boot_drive.drive_type == DriveType.HDD:
                args.extend(["-boot", "c"])

        if self.use_tablet:
            args.extend(["-usb", "-device", "usb-tablet"])

        if self.enable_audio:
            args.extend([
                "-device", "intel-hda",
                "-device", "hda-duplex"
            ])

        args.extend([
            "-vnc", f":{self.vnc_display}",
            "-monitor", f"unix:{self.monitor_socket},server,nowait"
        ])

        return args

    def to_json(self) -> str:
        data = {
            "cpu": self.cpu,
            "memory_mb": self.memory_mb,
            "drives": [d.to_dict() for d in self.drives],
            "vga": self.vga,
            "network_model": self.network_model,
            "use_tablet": self.use_tablet,
            "enable_audio": self.enable_audio,
            "vnc_display": self.vnc_display,
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'QemuComputer':
        data = json.loads(json_str)
        drives = [Drive.from_dict(d) for d in data.pop("drives")]
        computer = cls(**data)
        computer.drives = drives
        return computer

    def save_config(self, path: Path):
        path.write_text(self.to_json())

    @classmethod
    def load_config(cls, path: Path) -> 'QemuComputer':
        return cls.from_json(path.read_text())