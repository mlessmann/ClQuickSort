#ifdef _WIN32
#define __kernel
#define __global
#define __const
#define __local
#define get_global_id
#define get_local_id
#define get_local_size
#define get_group_id
#define get_num_groups
#define printf
#define barrier
#define CLK_LOCAL_MEM_FENCE 1
#define uint unsigned int
#define int2 struct { int x, y; }
#endif

// Rotate the matrix CLOCKWISE

//naive implementation: move the elements of the matrix directly to their destinations
//this will cause unaligned memory accessed which - as we will see - should be avoided on the GPU

__kernel void MatrixRotNaive(__global const float* M, __global float* MR, uint SizeX, uint SizeY)
{
    int2 gid;
    gid.x = get_global_id(0);
    gid.y = get_global_id(1);

    if (gid.x < SizeX && gid.y < SizeY)
        MR[gid.x * SizeY + (SizeY - gid.y - 1)] = M[gid.y * SizeX + gid.x];
}

//this kernel does the same thing, however, the local memory is used to
//transform a small chunk of the matrix locally
//then write it back after synchronization in a coalesced access pattern

__kernel void MatrixRotOptimized(__global const float* M, __global float* MR, uint SizeX, uint SizeY,
                                 __local float* block)
{
    int2 gid = { get_global_id(0), get_global_id(1) };
    int2 lid = { get_local_id(0), get_local_id(1) };
    int2 bid = { get_group_id(0), get_group_id(1) };
    int2 localSize = { get_local_size(0), get_local_size(1) };

    int globalIndex = gid.y * SizeX + gid.x;
    int localIndex = lid.y * localSize.x + lid.x;

    if (gid.x < SizeX && gid.y < SizeY)
        block[localIndex] = M[globalIndex];

    barrier(CLK_LOCAL_MEM_FENCE);

    int2 localSheared = { localIndex % localSize.y, localIndex / localSize.y };
    int blockCountY = get_num_groups(1);
    int2 blockRotated = { blockCountY - bid.y - 1, bid.x };
    int2 globalRotated = { blockRotated.x * localSize.y + localSheared.x, blockRotated.y * localSize.x + localSheared.y };
    
    if (globalRotated.x >= 0 && globalRotated.x < SizeY && globalRotated.y >= 0 && globalRotated.y < SizeX)
    {
        int globalRotatedIndex = globalRotated.y * SizeY + globalRotated.x;
        int localShearedRotatedIndex = (localSize.y - localSheared.x - 1) * localSize.x + localSheared.y;
        MR[globalRotatedIndex] = block[localShearedRotatedIndex];
    }
}
 