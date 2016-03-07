#ifdef _WIN32
#define __kernel
#define __global
#define __const
#define get_global_id
#endif

__kernel void VecAdd(__global const int* a, __global const int* b,
                     __global int* c, __const int numElements)
{
    int gid = get_global_id(0);
    if (gid < numElements)
        c[gid] = a[gid] + b[numElements - 1 - gid];
}