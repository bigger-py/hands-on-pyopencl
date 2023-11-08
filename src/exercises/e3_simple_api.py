"""
This code uses the built-in PyOpenCL APIs, which make it easier to
write simple parallel operations, to rewrite the vector addition
code from the previous exercise.

NOTE Much of the code is similar.
TODO Understand WHY the code is so much slower using the Elementwise Kernel.
"""
import time

import pyopencl as cl
import pyopencl.array
from pyopencl.elementwise import ElementwiseKernel
import numpy as np

# We start by making an execution context and a command queue for it.
context: cl.Context = cl.create_some_context()
command_queue: cl.CommandQueue = cl.CommandQueue(context)

vec_add = ElementwiseKernel(
    context,
    "const float *a, const float *b, float *c",
    "c[i] = a[i] + b[i]",
    "vec_add")

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

d_start = time.perf_counter()

# New way of allocating arrays on the device (more seamless?)
d_a = cl.array.to_device(command_queue, h_a)
d_b = cl.array.to_device(command_queue, h_b)
d_c = cl.array.empty_like(d_a)

# Queue the command.
vec_add(d_a, d_b, d_c, queue=command_queue)

# Alternative way of getting device data back - seems slow?
h_c = d_c.get()
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
