"""
In this exercise, we look at how to 
"""
import time
import math

import pyopencl as cl
import numpy as np


REPEATS = 3

def test_and_report(global_size: tuple[int], local_size: tuple[int] | None, print_host=False):

    d_dts = []

    for _ in range(REPEATS):

        # Time performance on device ...
        t_start = time.perf_counter()
        get_pi(command_queue, global_size, local_size, *args)
        command_queue.finish()
        t_end = time.perf_counter()
        d_dt = t_end-t_start

        cl.enqueue_copy(command_queue, h_pi_to_sum, d_pi_to_sum)

    d_dts.append(d_dt)

    d_dt_mean = np.mean(d_dts)

    complexity = N

    d_mflops = complexity/(d_dt_mean)/1000000.

    print(f"π = {np.sum(h_pi_to_sum)}, Device MFLOPS: {d_mflops}")
    print(np.sum(h_pi_to_sum)/np.pi)


# We start by making an execution context and a command queue for it.
context: cl.Context = cl.create_some_context(answers=[0])
command_queue: cl.CommandQueue = cl.CommandQueue(context)

# Define the problem. In this case, we want to calculate π using an
# integral approach. It turns out that the integral of 4/(1+x**2) over
# the interval 0 to 1 evaluates to π. We can numerically integrate a
# function naively by discretizing the problem, viewing it as the
# summation of a set of evenly distributed rectangles whose height is 
# equal to the function evaluated at the midpoint of that rectangle.
# In the limit of the number of rectangles going to infinity, this is
# equivalent to algebraic integration. 
# All we need to do is set the number of rectangles to be used. More
# rectangles will be more computationally intensive, but more accurate.

N = 1024**3

# Done serially, this calculation requires O(1) memory - there is no
# buffer keep track of each step, we just overwrite the last-calculated
# rectangle area with the most-recently calculated rectangle area. 
# HOWEVER, each of the N rectangles can be calculated independently,
# so we can distribute the iteration across multiple work items.
# It'd be inefficient to get each work item to do one iteration - this
# would lead to more scheduling overhead and memory access than makes
# sense. Let's therefore say that we have M iterations per work item.
# Therefore we require N/M work items to complete the task. 

M = 1024*256
num_work_items = int(math.ceil(N/M))
global_size = (num_work_items,)

# If we have a work-group size of L (i.e. L work items per work group),
# then we need N/(M*L) work groups. Each work group has access to some
# shared (local) memory, which we can perform a reduction on.
# This leaves us with a scalar value per work group which we should put
# into global memory. 

L = 32
local_size = (L,)
d_work_group_memory = cl.LocalMemory(np.dtype(np.float32).itemsize * L)

mf = cl.mem_flags
num_work_groups = int(math.ceil(num_work_items/L))
print(num_work_items, num_work_groups)
d_pi_to_sum = cl.Buffer(context, mf.WRITE_ONLY, np.dtype(np.float32).itemsize * num_work_groups)
h_pi_to_sum = np.empty(num_work_groups, dtype=np.float32)
# We will perform a final reduction on the host side, summing each of
# the work group's generated scalars.

# Simplest kernel.
with open("src/kernels/e9/simple_pi.cl", "r") as kernel_file:
    simple_pi = kernel_file.read()

program = cl.Program(context, simple_pi).build()
get_pi: cl.Kernel = program.get_pi
get_pi.set_scalar_arg_dtypes([np.int32, np.int32, None, None])
args = (N, M, d_work_group_memory, d_pi_to_sum)

print("Results for finding pi (simplest approach):")
test_and_report(global_size, local_size)

# We could in principle implement a further on-device reduction but
# we would have to launch another kernel to reduce those values. We
# could probably also perform more complex reduction whereby the 
# work-group summation is done in parallel iteratively until only
# one value is left.