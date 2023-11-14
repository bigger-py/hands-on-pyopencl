__kernel void mat_mul(const int sharedSize, const int rightWidth, __global const float *left, __global const float *right, __local float *rightColumn, __global float *out)
{
    // n.b. Should only be called if leftWidth == rightHeight === sharedSize.

    // Find the indices in the final matrix that we want to calculate for.
    int i = get_global_id(0);

    // Get index of this work item in the work group.
    int wi_index = get_local_id(0);

    // Get size of the work group.
    int wg_size = get_local_size(0);

    float left_memory[4096];
    // Store row into "private" memory for faster reuse.
    for (int k = 0; k < sharedSize; k++)
    {
        left_memory[k] = left[k + sharedSize*i];
    }

    // For each column in the output matrix...
    for (int j = 0; j < rightWidth; j++)
    {

        // Combine all wi effort to store the column of R into rightColumn buffer.
        // Sort of a grid stride?
        for (int l = wi_index; l < sharedSize; l+=wg_size)
        {
            rightColumn[l] = right[j + rightWidth*l];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Initialize a running sum.
        float sum = 0.0f;
        
        // For the index along the shared dimension...
        for (int k = 0; k < sharedSize; k++)
        {
            // left[i,:] * right[:,j]
            sum = sum + left_memory[k] * rightColumn[k];
        }
        
        // Output matrix shape is (leftHeight, rightWidth). Assign sum.
        out[j + i*rightWidth] = sum;

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}