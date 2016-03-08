#ifdef _WIN32
#define __kernel
#define __global
#define __const
#define __local
#define get_global_id
#define get_local_id
#define get_group_id
#define get_local_size
#define atomic_inc
#define barrier(CLK_LOCAL_MEM_FENCE)
#define printf
#endif

// Bank conflicts
//#define AVOID_BANK_CONFLICTS
#ifdef AVOID_BANK_CONFLICTS
// TO DO: define your conflict-free macro here
#define MAPPED_INDEX(index) ((index) + ((index) / (NUM_BANKS)))
// Alternative macro which seems to be slower at least on Intel GPUs
//#define MAPPED_INDEX(index) ((index) + ((index) >> NUM_BANKS + (index) >> (2 * NUM_BANKS_LOG)))
#else
#define MAPPED_INDEX(index) (index)
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
		//printf("Grid %i: %i, %i\n", grid, localLeftCount, localRightCount);
	}
}

__kernel void Scan_Naive(const __global int* inArray, __global int* outArray, int N, int offset)
{
	int GID = get_global_id(0);
	if (GID >= N)
		return;


	if (GID < offset) {
		outArray[GID] = inArray[GID];
	}
	else {
		outArray[GID] = inArray[GID] + inArray[GID - offset];
	}
	//printf("%i: %i\n", GID, outArray[GID]);
}