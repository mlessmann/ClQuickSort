/******************************************************************************
GPU Computing / GPGPU Praktikum source code.

******************************************************************************/

#include "QuickSort.h"

#include "QuickSortTask.h"

#include <iostream>

using namespace std;

///////////////////////////////////////////////////////////////////////////////
// CAssignment1

bool QuickSort::DoCompute(int argc, char** argv)
{
    size_t localWorkSize[3] = { 256, 1, 1 };
	int elements = argc >= 2 ? atoi(argv[1]) : 1024 * 1024 * 64;
	int leftBound = argc >= 3 ? atoi(argv[2]) : 0;
	int rightBound = argc >= 4 ? atoi(argv[3]) : 1024 * 64;
	cout << "Starting Quicksort with " << elements << " elements and rng-bounds [" << leftBound << ";" << rightBound << "]." << endl;
	QuickSortTask task(elements, leftBound, rightBound);
    RunComputeTask(task, localWorkSize);

	return true;
}

///////////////////////////////////////////////////////////////////////////////
