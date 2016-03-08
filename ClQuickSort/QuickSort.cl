#ifdef _WIN32
#define __kernel
#define __global
#define __const
#define __local
#define get_global_id
#define get_local_id
#define get_group_id
#define atomic_inc
#define barrier(CLK_LOCAL_MEM_FENCE)
#define printf
#endif

__kernel void QuickSort1(__global const int* input, __global int* output,
	__const int startIndex, __const int count, __const int pivotIndexIn,
	__global int* leftCount, __global int* rightCount)
{
	__local int localLeftCount;
	__local int localRightCount;

	int gid = get_global_id(0);
	int lid = get_local_id(0);
	int grid = get_group_id(0);

	if (gid >= count)
		return;

	if (lid == 0)
	{
		localLeftCount = 0;
		localRightCount = 0;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	int pivot = input[pivotIndexIn];
	int element = input[startIndex + gid];
	if (element < pivot)
		atomic_inc(&localLeftCount);
	else if(element > pivot)
		atomic_inc(&localRightCount);

	barrier(CLK_LOCAL_MEM_FENCE);

	if (lid == 0)
	{
		leftCount[grid] = localLeftCount;
		rightCount[grid] = localRightCount;
	}
}