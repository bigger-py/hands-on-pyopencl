__kernel void get_pi(int N, int M, __local float *work_group_to_reduce, __global float *global_to_reduce)
{
    float sum = 0.0f;
    
    int global_thread_index = get_global_id(0);
    int wg_thread_index = get_local_id(0);
    float x;
    float step_size = 1.0f/N;

    // Iterate M times, starting at global_thread_index * M.
    for (int i = global_thread_index*M; i < (global_thread_index+1)*M; i++) {
        x = (i+0.5f)*step_size
;
        sum += 4.0f/(1.0f+x*x);
    }
    work_group_to_reduce[wg_thread_index] = sum;

    // Wait for all threads to finish iterating.
    barrier(CLK_LOCAL_MEM_FENCE);

    // Only the first thread in the work group is needed now.
    if (wg_thread_index != 0) return;
    
    // Now sum all the contributions from this work group.
    float wg_pi_contribution = 0.0f;
    int wg_size = get_local_size(0);
    for (int i = 0; i < wg_size; i++)
    {
        wg_pi_contribution += work_group_to_reduce[i];
    }

    // Put summed contributions into the global buffer to reduce again. 
    global_to_reduce[get_group_id(0)] = wg_pi_contribution * step_size;
}