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

# Now that we have allocated buffers and defined the problem, let's
# consider the different implementations of the solution approach.
# Regardless of the approach, we must define and build a kernel, and
# explicitly indicate how we would like the kernel to be distributed
# across the compute device. 
# 
# We have already seen one implementation of matrix multiplication
# where each kernel calculates a single element of the output array.
# Let's revisit this just to give an indication of performance with
# practically no optimization.
with open("src/kernels/e7/matmul_core.cl", "r") as kernel_file:
    core_source = kernel_file.read()

global_size = (M, O,)
local_size = None
program = cl.Program(context, core_source).build()
mat_mul: cl.Kernel = program.mat_mul
mat_mul.set_scalar_arg_dtypes([np.int32, None, None, None])
args = (N, d_L, d_R, d_T)

print("Results for naive matrix multiplication:")
test_and_report(global_size, local_size, print_host=True)
print("")

# Let's now try and optimize this matrix multiplication a bit. There is
# some overhead on the device to manage the work groups and work items.
# We know that our compute device is never (for now at least...) going
# to have 1024*1024 distinct processing elements, so perhaps it's not
# sensible to let the device manage that many total work items. Let's 
# therefore constrain each work item to do an entire row of the output
# matrix. 
with open("src/kernels/e7/matmul_wi_row.cl", "r") as kernel_file:
    one_d_source = kernel_file.read()

# Our global size is now just the number of rows in the output matrix.
global_size = (M,)
local_size = (32,)
program = cl.Program(context, one_d_source).build()
mat_mul: cl.Kernel = program.mat_mul
mat_mul.set_scalar_arg_dtypes([np.int32, np.int32, None, None, None])
args = (N, O, d_L, d_R, d_T)

print("Results for matrix multiplication (one work-item per row):")
test_and_report(global_size, local_size)
print("")

# Memory access is expensive -  we should try to 
# perform as many FLOPS as possible per memory access. Furthermore,
# some memory accesses are less costly than others. I think one way
# to think about it is that private memory is the fastest to access,
# but also the most scarce. There is more complexity to it though!
# Let's adjust the kernell so that we don't keep repeating the same
# memory accesses over and over again.
with open("src/kernels/e7/matmul_wi_row_private.cl", "r") as kernel_file:
    one_d_source = kernel_file.read()

global_size = (M,)
local_size = (32,)
program = cl.Program(context, one_d_source).build()
mat_mul: cl.Kernel = program.mat_mul
mat_mul.set_scalar_arg_dtypes([np.int32, np.int32, None, None, None])
args = (N, O, d_L, d_R, d_T)

print("Results for  matrix multiplication (one work-item per row, 'private' memory):")
test_and_report(global_size, local_size)

# NOTE For some reason, the device performance actually seems to be
# best with the naive approach when the problem size is 1024**3. 
# Increasing the problem size along M or N solves this.
# Either way, everything seems to be slower than the
# numpy approach of L @ R, which is probably because it uses a much
# more efficient algorithm.