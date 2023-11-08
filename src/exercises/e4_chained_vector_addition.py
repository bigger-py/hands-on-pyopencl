"""
This code extends the previous vector addition exercise (e2) by adding
multiple buffers and performing chains of vector addition, to 
demonstrate the host/device impact of adding more buffers and 
calculations to the kernel.

NOTE Much of the code is similar to exercise e2.

We will be solving the problem:
Find F === D + G, given A, B, E and G, where
D === C + E and C === A + B
"""
import time

import pyopencl as cl
import numpy as np

# We start by making an execution context and a command queue for it.
context: cl.Context = cl.create_some_context()
command_queue: cl.CommandQueue = cl.CommandQueue(context)

# Define a simple kernel function to add two vectors.
VECTOR_ADDITION = """
__kernel void vec_add_verbose(__global const float *a, __global const float *b, __global float *c)
{
    // Add together two vectors.
    int i = get_global_id(0);
    c[i] = a[i] + b[i];
}
"""

# Define the host (h_) arrays we want to add together.
h_a = np.random.random((2048,)).astype(np.float32)
h_b = np.random.random(h_a.shape).astype(np.float32)
h_e = np.random.random(h_a.shape).astype(np.float32)
h_g = np.random.random(h_a.shape).astype(np.float32)
h_c = np.empty_like(h_a, dtype=np.float32)
h_d = np.empty_like(h_a, dtype=np.float32)
h_f = np.empty_like(h_a, dtype=np.float32)

# Define the global size of the problem we are solving.
global_size = h_a.shape

# Explicitly specify the no. of work groups (= global_size//local_size). 
local_size = (32,)

# Helper to reduce characters.
mf = cl.mem_flags

program = cl.Program(context, VECTOR_ADDITION).build()
vec_add: cl.Kernel = program.vec_add_verbose

# Define the same-sized buffers on the device (d_).
# NOTE this copies host the various arrays to the compute device.
d_start = time.perf_counter()
d_a = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_a)
d_b = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_b)
d_e = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_e)
d_g = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_g)
d_c = cl.Buffer(context, mf.READ_WRITE, h_c.nbytes)
d_d = cl.Buffer(context, mf.READ_WRITE, h_d.nbytes)
d_f = cl.Buffer(context, mf.WRITE_ONLY, h_f.nbytes)

# Chain the vector addition commands in order.
vec_add(command_queue, global_size, local_size, d_a, d_b, d_c)
vec_add(command_queue, global_size, local_size, d_c, d_e, d_d)
vec_add(command_queue, global_size, local_size, d_d, d_g, d_f)

# Copy the data from the device back to the host.
cl.enqueue_copy(command_queue, h_f, d_f)
d_dt = time.perf_counter() - d_start

# Compare the expected result with the received result.
h_start = time.perf_counter()
h_f_host_calc = np.add(np.add(np.add(h_a, h_b), h_e), h_g)
h_dt = time.perf_counter() - h_start

print(f"Host computation: {h_dt*1000.0} ms.")
print(f"Device computation: {d_dt*1000.0} ms.")

try:
    assert np.allclose(h_f, h_f_host_calc)
except AssertionError as e:
    print("Failed to successfully add the vectors together...")
else:
    print("Successfully added the vectors together.")
