/******************************************************************************
GPU Computing / GPGPU Praktikum source code.

******************************************************************************/

#include "QuickSort.h"

#include "QuickSortTask.h"

#include <iostream>

using namespace std;

///////////////////////////////////////////////////////////////////////////////
// CAssignment1

bool QuickSort::DoCompute()
{
    size_t localWorkSize[3] = { 256, 1, 1 };
	QuickSortTask task(1048576);
    RunComputeTask(task, localWorkSize);

	return true;
}

///////////////////////////////////////////////////////////////////////////////
