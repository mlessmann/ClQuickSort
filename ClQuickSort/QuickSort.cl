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

__kernel void CountElements(__global const int* input,
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

__kernel void Scan_Naive(const __global int* inArray, __global int* outArray, int N, int offset)
{
	int GID = get_global_id(0);
	if (GID < N)
	{
		if (GID < offset) {
			outArray[GID] = inArray[GID];
		}
		else {
			outArray[GID] = inArray[GID] + inArray[GID - offset];
		}
	}
}

__kernel void DistributeElements(__global const int* input, __global int* output,
	__const int startIndex, __const int count, __global int* leftCountPrefix,
	__global int* rightCountPrefix, __const int pivotIndexIn)
{
	__local int leftIndex;
	__local int rightIndex;

	int gid = get_global_id(0);
	int lid = get_local_id(0);
	int grid = get_group_id(0);
	int nGroups = count / (get_local_size(0)) + (count % (get_local_size(0)) > 0);

	if (gid >= count)
		return;

	if (lid == 0) 
	{
		leftIndex = 0;
		rightIndex = 0;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	int pivot = input[pivotIndexIn];
	int element = input[startIndex + gid];
	int globalLeftCount = leftCountPrefix[nGroups - 1];
	int globalRightCount = rightCountPrefix[nGroups - 1];
	int globalPivotCount = count - globalRightCount - globalLeftCount;
	int globalLeftIndex = grid == 0 ? 0 : leftCountPrefix[grid - 1];
	int globalRightIndex = count - globalRightCount + (grid == 0 ? 0 : rightCountPrefix[grid - 1]);
	
	int newElementIndex = -1;

	if (element < pivot)
	{
		newElementIndex = globalLeftIndex + atomic_inc(&leftIndex);
	}
	else if (element > pivot)
	{
		newElementIndex = globalRightIndex + atomic_inc(&rightIndex);
	}

	if (newElementIndex >= 0) 
	{
		output[startIndex + newElementIndex] = element;
	}

	if (gid < globalPivotCount) 
	{
		output[startIndex + globalLeftCount + gid] = pivot;
	}
}