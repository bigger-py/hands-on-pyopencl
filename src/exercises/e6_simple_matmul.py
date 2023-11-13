"""
This code does matrix multiplication by taking the dot product of the
ith row and the jth column of two matrices.

We will be solving the problem:
Find C === A @ B where @ indicates a matrix multiplication.
We will assume tha the matrices are correctly formed (or 
at least let the host validate that.)
"""
import time

import pyopencl as cl
import numpy as np


REPEATS = 10

# We start by making an execution context and a command queue for it.
context: cl.Context = cl.create_some_context()
command_queue: cl.CommandQueue = cl.CommandQueue(context)

# Define a matrix-multiplication kernel.
MAT_MUL = """
__kernel void mat_mul(const int sharedSize, __global const float *left, __global const float *right, __global float *out)
{
    // n.b. Should only be called if leftWidth == rightHeight === sharedSize.
    int rightWidth = get_global_size(1);

    // Find the indices in the final matrix that we want to calculate for.
    int i = get_global_id(0);
    int j = get_global_id(1);

    // Initialize a running sum.
    float sum = 0.0f;

    // For the row and column that we are interested in, multiply each element of the left row and right column together, then sum them.
    for (int k = 0; k < sharedSize; k++)
    {
        // left[i,:] * right[:,j]
        sum = sum + left[k + sharedSize*i] * right[j + rightWidth*k];
    }
    
    // Output matrix shape is (leftHeight, rightWidth). Assign sum.
    out[j + i*rightWidth] = sum;
}
"""
N = 2048
# Define the host (h_) arrays we want to add together.
h_left = np.random.random((N,N)).astype(np.float32)
h_right = np.random.random((N,N)).astype(np.float32)

# Only a valid problem if the left width equals the right height.
assert h_left.shape[1] == h_right.shape[0]

# Define the global size of the problem we are solving.
global_size = (h_left.shape[0], h_right.shape[1],)

# Let OpenCL figure out the "correct" number of work groups.
local_size = None

# Helper to reduce characters.
mf = cl.mem_flags

program = cl.Program(context, MAT_MUL).build()
mat_mul: cl.Kernel = program.mat_mul
mat_mul.set_scalar_arg_dtypes([np.int32, None, None, None])

h_dts = []
d_dts = []
try:
    for i in range(REPEATS):
        h_out = np.empty(global_size, dtype=np.float32)

        # Define the same-sized buffers on the device (d_).
        # NOTE this copies host the various arrays to the compute device.
        d_left = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_left)
        d_right = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_right)
        d_out = cl.Buffer(context, mf.WRITE_ONLY, h_out.nbytes)

        # Submit the command to perform matrix multiplication.
        d_start = time.perf_counter()
        mat_mul(command_queue, global_size, local_size, h_left.shape[1], d_left, d_right, d_out)
        command_queue.finish()
        d_dt = time.perf_counter() - d_start

        # Copy the data from the device back to the host.
        cl.enqueue_copy(command_queue, h_out, d_out)

        # Compare the expected result with the received result.
        h_start = time.perf_counter()
        h_out_host_calc = h_left @ h_right
        h_dt = time.perf_counter() - h_start

        assert np.allclose(h_out, h_out_host_calc)

        h_dts.append(h_dt)
        d_dts.append(d_dt)

    av_h_dt = np.mean(h_dts)*1000.0
    av_d_dt = np.mean(d_dts)*1000.0
    std_h_dt = np.std(h_dts)*1000.0
    std_d_dt = np.std(d_dts)*1000.0

    h_mflops = ( 2.0 * (h_left.shape[1]*h_left.shape[0]*h_right.shape[0]) )/(1000000.0* av_h_dt)
    d_mflops = ( 2.0 * (h_left.shape[1]*h_left.shape[0]*h_right.shape[0]) )/(1000000.0* av_d_dt)
    print(f"Host computation: {av_h_dt}+-{std_h_dt} ms ({h_mflops} MFLOPS).")
    print(f"Device computation: {av_d_dt}+-{std_d_dt} ms ({d_mflops} MFLOPS).")
    
except AssertionError as e:
    print("Failed to successfully multiply the two matrices...")
else:
    print("Successfully multiplied the two matrices.")
