"""
This code adds together two vectors on a compute device.
"""
import time

import pyopencl as cl
import numpy as np

# We start by making an execution context and a command queue for it.
context: cl.Context = cl.create_some_context()
command_queue: cl.CommandQueue = cl.CommandQueue(context)

# Define the kernel function (will be compiled at runtime).
VECTOR_ADDITION = """
__kernel void vec_add(__global const float *a, __global const float *b, __global float *c)
{
    // Add together two vectors a and b and put the output into a vector c.
    int i = get_global_id(0);
    c[i] = a[i] + b[i];
}
"""

# Build the program and pull out just the kernel we want to call.
# NOTE Each time we call program.vec_add it resets any args that we
#      might have already attached to kernel via kernel.set_args()
#      So... easy to just add args when we call the kernel!

program = cl.Program(context, VECTOR_ADDITION).build()
vec_add: cl.Kernel = program.vec_add

# Define the host (h_) arrays we want to add together.
h_a = np.random.random((2048,)).astype(np.float32)
h_b = np.random.random(h_a.shape).astype(np.float32)
h_c = np.empty_like(h_a, dtype=np.float32)

# Define the global size of the problem we are solving.
global_size = h_a.shape

# Explicitly specify the no. of work groups (= global_size//local_size). 
local_size = (32,)

# Helper to reduce characters.
mf = cl.mem_flags

# Define the same-sized buffers on the device (d_).
# NOTE this copies a and b host arrays to the compute device.
d_start = time.perf_counter()
d_a = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_a)
d_b = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_b)
d_c = cl.Buffer(context, mf.WRITE_ONLY, h_c.nbytes)

# Queue the command.
vec_add(command_queue, global_size, local_size, d_a, d_b, d_c)

# Copy the data from the device to the host.
cl.enqueue_copy(command_queue, h_c, d_c)
d_dt = time.perf_counter() - d_start

# Compare the expected result with the received result.
h_start = time.perf_counter()
np.add(h_a, h_b)
h_dt = time.perf_counter() - h_start

print(f"Host computation: {h_dt*1000.0} ms.")
print(f"Device computation: {d_dt*1000.0} ms.")

try:
    assert np.allclose(h_c, h_a+h_b)
except AssertionError as e:
    print("Failed to successfully add the two vectors together...")
else:
    print("Successfully added the two vectors together.")
