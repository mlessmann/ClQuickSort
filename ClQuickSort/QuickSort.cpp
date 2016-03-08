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
	int elements = 1024 * 1024 * 64;
	cout << "Starting Quicksort with " << elements << " elements.";
	QuickSortTask task(elements);
    RunComputeTask(task, localWorkSize);

	return true;
}

///////////////////////////////////////////////////////////////////////////////
