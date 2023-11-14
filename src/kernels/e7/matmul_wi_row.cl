__kernel void mat_mul(const int sharedSize, const int rightWidth, __global const float *left, __global const float *right, __global float *out)
{
    // n.b. Should only be called if leftWidth == rightHeight === sharedSize.

    // Find the indices in the final matrix that we want to calculate for.
    int i = get_global_id(0);

    // For each column in the output matrix...
    for (int j = 0; j < rightWidth; j++)
    {
        // Initialize a running sum.
        float sum = 0.0f;
        
        // For the index along the shared dimension...
        for (int k = 0; k < sharedSize; k++)
        {
            // left[i,:] * right[:,j]
            sum = sum + left[k + sharedSize*i] * right[j + rightWidth*k];
        }
        
        // Output matrix shape is (leftHeight, rightWidth). Assign sum.
        out[j + i*rightWidth] = sum;
    }
}