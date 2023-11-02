from enum import IntEnum

class OpenCLDeviceType(IntEnum):
    Default = 1
    CPU = 2
    GPU = 4
    Accelerator = 8
    Custom = 16
    Host = 65536
