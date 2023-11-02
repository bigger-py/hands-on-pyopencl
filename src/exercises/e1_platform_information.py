"""
This code provides information about the platforms available from this
host device, including platform information and the devices available
on each platform. 

Directions:
- Import the Python OpenCL API
- Create a list of all the platform IDs (platforms)
- For each platform...
-   Print out some information about the platforms
-   Discover all devices
-   For each device...
-       Print out some information about it
-       Find the maximum dimensions of the work-groups
"""
# Import the Python OpenCL API.
import pyopencl as cl

from utils.enums import OpenCLDeviceType


# Create a list of of all the platforms.
platforms = cl.get_platforms()

for platform in platforms:
    # Print out some information about the platforms
    # Other info e.g. extensions and host_timer_resolution is available but quite verbose...
    print(f"{platform.name} | {platform.vendor} | {platform.version}")
    
    try: 
        v_start = platform.version.index("OpenCL ") + len("OpenCL ")
        numeric_version = float(platform.version[v_start:v_start+3])
    except ValueError as ve:
        numeric_version = 0.0
        print("\tCould not parse OpenCL version for this platform.")
    # Find the devices for this platform.
    devices = platform.get_devices()
    if not devices:
        print(f"\tNo devices for this platform.")
        continue
    
    # For each device, print out some information about it.
    for device in devices:

        print(f"\t{device.name}:")
        print(f"\t\tDevice type: {OpenCLDeviceType(device.type).name}")
        print(f"\t\tUnified Memory: {bool(device.host_unified_memory)}")
        print(f"\t\tMax 2D Shape: {device.image2d_max_height, device.image2d_max_width}")
        print(f"\t\tMax 3D Shape: {device.image3d_max_height, device.image3d_max_width, device.image3d_max_depth}")
        print(f"\t\t'Maximum' Clock Frequency (MHz): {device.max_clock_frequency}")
        print(f"\t\tMaximum Clock Frequency (MHz): {device.max_clock_frequency}")
        if numeric_version >= 3.0:
            pass

        if numeric_version >= 1.2:
            print(f"\t\tMax Image Buffer Size: {device.image_max_buffer_size}")
            print(f"\t\tMax Image Array Size: {device.image_max_array_size}")


# Discover all devices

# Investigate each device

# Print out some information about the devices

# Find the maximum dimensions of the work-groups