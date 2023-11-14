"""
This code continues with optimization of the matrix multiplication
problem, by using local memory which is shared within a work group.
"""
import time

import pyopencl as cl
import numpy as np


REPEATS = 3

def test_and_report(global_size: tuple[int], local_size: tuple[int] | None, print_host=False):

    h_dts = []
    d_dts = []

    for _ in range(REPEATS):

        # Time performance on device ...
        t_start = time.perf_counter()
        mat_mul(command_queue, global_size, local_size, *args)
        command_queue.finish()
        t_end = time.perf_counter()
        d_dt = t_end-t_start

        # ... and on host.
        t_start = time.perf_counter()
        h_T_host_calc = h_L @ h_R
        t_end = time.perf_counter()
        h_dt = t_end-t_start
        cl.enqueue_copy(command_queue, h_T, d_T)

        assert np.allclose(h_T, h_T_host_calc)

    d_dts.append(d_dt)
    h_dts.append(h_dt)

    d_dt_mean = np.mean(d_dts)
    d_dt_std = np.std(d_dts)

    h_dt_mean = np.mean(h_dts)
    h_dt_std = np.std(h_dts)

    complexity = 2*N*M*O

    d_mflops = complexity/(d_dt_mean)/1000000.
    h_mflops = complexity/(h_dt_mean)/1000000.

    if print_host:
        print(f"Host MFLOPS: {h_mflops}")
    print(f"Device MFLOPS: {d_mflops}")


# We start by making an execution context and a command queue for it.
context: cl.Context = cl.create_some_context()
command_queue: cl.CommandQueue = cl.CommandQueue(context)

# First define the problem. Let's support solving a general matrix
# multiplication of two matrices: L and R. The shapes of matrices L and
# R are (M,N) and (N,O), respectively. Do this on the host (h_).

M = 1024
N = 1024
O = 1024

h_L = np.random.random((M,N)).astype(np.float32)
h_R = np.random.random((N,O)).astype(np.float32)

# With the above configuration, we expect an output matrix T whose
# shape is (M,O). Regardless of exactly how we approach this problem,
# we will need to allocate an array of this shape on the device, as
# well as allocating device buffers that correspond to L, R and T.

h_T = np.empty((M,O), dtype=np.float32)    


# NOTE By allocating device memory here, we have essentially declared
# that we won't be taking memory allocation into account whenever we
# benchmark the various optimizations.
mf = cl.mem_flags
d_L = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_L)
d_R = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_R)
d_T = cl.Buffer(context, mf.WRITE_ONLY, h_T.nbytes)

# Our reference point is the best case from e7.
with open("src/kernels/e7/matmul_wi_row_private.cl", "r") as kernel_file:
    one_d_private_source = kernel_file.read()

global_size = (M,)
local_size = (32,)
program = cl.Program(context, one_d_private_source).build()
mat_mul: cl.Kernel = program.mat_mul
mat_mul.set_scalar_arg_dtypes([np.int32, np.int32, None, None, None])
args = (N, O, d_L, d_R, d_T)

print("Results for matrix multiplication (one work-item per row, 'private' memory):")
test_and_report(global_size, local_size)

# Notice that in Exercise 7, we gained some efficiency by using the
# fact that each work item repeatedly tries to access the same
# row of matrix L, once for each column of matrix R. By making a 
# local copy of that row at the start of the work item, we sped up 
# the memory access. 
# 
# Now consider how every work item in the work group needs access to 
# the same columns of matrix R - i.e. there is a shared memory space
# that is being accessed by each work _group_. 
# 
# IDEA: for each work item in a work group, store the column that all
# work items are about to use into a _local_ buffer (shared by all work
# items), then use it, then move onto the next column. This requires us
# to synchronize the threads before:
#   a) Trying to use the data in the local buffer.
#   b) Moving onto the next column.
# This is achieved by using memory fences which are a simple means of
# synchronization within the work group. 
# 
# Note that our host code has to change slightly because we must
# allocate the local buffer.
with open("src/kernels/e8/matmul_wi_row_private_wg_col_local.cl", "r") as kernel_file:
    local_column_source = kernel_file.read()

# NOTE local memory is not initialized in the same way as global!
d_R_col = cl.LocalMemory(np.float32().nbytes*N)
global_size = (M,)
local_size = (32,)
program = cl.Program(context, local_column_source).build()
mat_mul: cl.Kernel = program.mat_mul
mat_mul.set_scalar_arg_dtypes([np.int32, np.int32, None, None, None, None])
args = (N, O, d_L, d_R, d_R_col, d_T)

print("Results for matrix multiplication (one work-item per row in 'private' memory, work group sharing R column):")
test_and_report(global_size, local_size)

# Now run the optimal "blocked" approach written by someone 
# with much more experience than me!
with open("src/kernels/e8/matmul_blocked.cl", "r") as kernel_file:
    blocked_source = kernel_file.read()

global_size = (M,O)
local_size = (16,16)
program = cl.Program(context, blocked_source).build()
mat_mul: cl.Kernel = program.mmul
mat_mul.set_scalar_arg_dtypes([np.int32, None, None, None, None, None])

# Work-group computes a block of C. This size is also set
# in a #define inside the kernel function. Note this blocksize
# must evenly divide the matrix order
blocksize = 16

A_block = cl.LocalMemory(np.dtype(np.float32).itemsize * blocksize * blocksize)
B_block = cl.LocalMemory(np.dtype(np.float32).itemsize * blocksize * blocksize)
args = (N, d_L, d_R, d_T, A_block, B_block)

print("Results for blocked matrix multiplication:")
test_and_report(global_size, local_size)