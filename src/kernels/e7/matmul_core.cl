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