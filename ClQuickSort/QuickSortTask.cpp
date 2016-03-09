/******************************************************************************
GPU Computing / GPGPU Praktikum source code.

******************************************************************************/

#include "QuickSortTask.h"

#include "../Common/CLUtil.h"
#include "../Common/CTimer.h"
#include <cstring>

using namespace std;

///////////////////////////////////////////////////////////////////////////////
// QuickSortTask

QuickSortTask::QuickSortTask(size_t size, int leftBound, int rightBound)
    :m_Size(size), m_hInput(NULL), m_hOutput(NULL), m_dInput(NULL),
    m_dOutput(NULL), m_hGPUResult(NULL), m_Program(NULL), m_KernelScan(NULL),
	m_KernelCountElements(NULL), m_KernelDistributeElements(NULL), m_dLeftCount(NULL),
	m_dRightCount(NULL), m_dScanPing(NULL), m_dScanPong(NULL), m_Rnd(NULL),
	m_LeftBound(leftBound), m_RightBound(rightBound)
{
}

QuickSortTask::~QuickSortTask()
{
    ReleaseResources();
}

size_t GetGroupCount(size_t size, size_t localWorkSize)
{
	return (size_t)ceil((double)size / (double)localWorkSize);
}

bool QuickSortTask::InitResources(cl_device_id Device, cl_context Context)
{
    //CPU resources
    m_hInput = new int[m_Size];
    m_hOutput = new int[m_Size];
    m_hGPUResult = new int[m_Size];

	//fill the array with random ints
	uniform_int_distribution<int> distribution(m_LeftBound, m_RightBound);
    for(unsigned int i = 0; i < m_Size; i++)
    {
		m_hInput[i] = distribution(m_Rnd);
    }
	//memset(m_hInput, 0, m_Size * sizeof(int));
	memcpy(m_hOutput, m_hInput, m_Size * sizeof(int));

    cl_int clError;

	//create buffers
    m_dInput = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(int) * m_Size, NULL, &clError);
    V_RETURN_FALSE_CL(clError, "Buffer allocation failed.");
	m_dOutput = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(int) * m_Size, NULL, &clError);
	V_RETURN_FALSE_CL(clError, "Buffer allocation failed.");

    size_t programSize = 0;
    string programCode;

    if (!CLUtil::LoadProgramSourceToMemory("QuickSort.cl", programCode))
        return false;

    m_Program = CLUtil::BuildCLProgramFromMemory(Device, Context, programCode);
    if (m_Program == nullptr)
        return false;

	m_KernelCountElements = clCreateKernel(m_Program, "CountElements", &clError);
	V_RETURN_FALSE_CL(clError, "Failed to create kernel: CountElements");
	m_KernelScan = clCreateKernel(m_Program, "Scan_Naive", &clError);
	V_RETURN_FALSE_CL(clError, "Failed to create kernel: Scan_Naive");
	m_KernelDistributeElements = clCreateKernel(m_Program, "DistributeElements", &clError);
	V_RETURN_FALSE_CL(clError, "Failed to create kernel: DistributeElements");

    return true;
}

void QuickSortTask::ReleaseResources()
{
    //CPU resources
    SAFE_DELETE_ARRAY(m_hInput);
    SAFE_DELETE_ARRAY(m_hOutput);
    SAFE_DELETE_ARRAY(m_hGPUResult);

    SAFE_RELEASE_MEMOBJECT(m_dInput);
    SAFE_RELEASE_MEMOBJECT(m_dOutput);
	SAFE_RELEASE_MEMOBJECT(m_dLeftCount);
	SAFE_RELEASE_MEMOBJECT(m_dRightCount);
	SAFE_RELEASE_MEMOBJECT(m_dScanPing);
	SAFE_RELEASE_MEMOBJECT(m_dScanPong);

	SAFE_RELEASE_KERNEL(m_KernelScan);
	SAFE_RELEASE_KERNEL(m_KernelCountElements);
	SAFE_RELEASE_KERNEL(m_KernelDistributeElements);
    SAFE_RELEASE_PROGRAM(m_Program);
}

void QuickSortTask::CountElements(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3],
	size_t startIndex, size_t count, size_t pivotIndex)
{
	cl_int clErr;
	size_t globalWorkSize = CLUtil::GetGlobalWorkSize(count, LocalWorkSize[0]);

	clErr = clSetKernelArg(m_KernelCountElements, 0, sizeof(cl_mem), (void*)&m_dInput);
	clErr |= clSetKernelArg(m_KernelCountElements, 1, sizeof(cl_int), (void*)&startIndex);
	clErr |= clSetKernelArg(m_KernelCountElements, 2, sizeof(cl_int), (void*)&count);
	clErr |= clSetKernelArg(m_KernelCountElements, 3, sizeof(cl_int), (void*)&pivotIndex);
	clErr |= clSetKernelArg(m_KernelCountElements, 4, sizeof(cl_mem), (void*)&m_dLeftCount);
	clErr |= clSetKernelArg(m_KernelCountElements, 5, sizeof(cl_mem), (void*)&m_dRightCount);
	V_RETURN_CL(clErr, "Failed to set kernel args for Kernel1.");

	clErr = clEnqueueNDRangeKernel(CommandQueue, m_KernelCountElements, 1, NULL, &globalWorkSize, LocalWorkSize, 0, NULL, NULL);
	V_RETURN_CL(clErr, "Failed to start Kernel1.");
}

void QuickSortTask::Scan(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3],
	size_t count, cl_mem input, int groupCount)
{
	cl_int clErr;

	clEnqueueCopyBuffer(CommandQueue, input, m_dScanPing, 0, 0, sizeof(cl_int) * groupCount, 0, NULL, NULL);

	for (unsigned int offset = 1; offset <= groupCount; offset *= 2)
	{
		size_t globalWorkSize = CLUtil::GetGlobalWorkSize(groupCount, LocalWorkSize[0]);

		clErr = clSetKernelArg(m_KernelScan, 0, sizeof(cl_mem), (void*)&m_dScanPing);
		clErr |= clSetKernelArg(m_KernelScan, 1, sizeof(cl_mem), (void*)&m_dScanPong);
		clErr |= clSetKernelArg(m_KernelScan, 2, sizeof(cl_int), (void*)&groupCount);
		clErr |= clSetKernelArg(m_KernelScan, 3, sizeof(cl_int), (void*)&offset);
		V_RETURN_CL(clErr, "Failed to set kernel args: KernelScan");

		clErr = clEnqueueNDRangeKernel(CommandQueue, m_KernelScan, 1, NULL, &globalWorkSize, LocalWorkSize, 0, NULL, NULL);
		V_RETURN_CL(clErr, "Failed to start KernelScan.");
		swap(m_dScanPing, m_dScanPong);
	}

	clEnqueueCopyBuffer(CommandQueue, m_dScanPing, input, 0, 0, sizeof(cl_int) * groupCount, 0, NULL, NULL);
}

void QuickSortTask::DistributeElements(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3],
	size_t startIndex, size_t count, size_t pivotIndex)
{
	cl_int clErr;
	size_t globalWorkSize = CLUtil::GetGlobalWorkSize(count, LocalWorkSize[0]);

	clErr = clSetKernelArg(m_KernelDistributeElements, 0, sizeof(cl_mem), &m_dInput);
	clErr |= clSetKernelArg(m_KernelDistributeElements, 1, sizeof(cl_mem), &m_dOutput);
	clErr |= clSetKernelArg(m_KernelDistributeElements, 2, sizeof(cl_int), &startIndex);
	clErr |= clSetKernelArg(m_KernelDistributeElements, 3, sizeof(cl_int), &count);
	clErr |= clSetKernelArg(m_KernelDistributeElements, 4, sizeof(cl_mem), &m_dLeftCount);
	clErr |= clSetKernelArg(m_KernelDistributeElements, 5, sizeof(cl_mem), &m_dRightCount);
	clErr |= clSetKernelArg(m_KernelDistributeElements, 6, sizeof(cl_int), &pivotIndex);
	V_RETURN_CL(clErr, "Failed to set kernel args: KernelDistributeElements");

	clErr = clEnqueueNDRangeKernel(CommandQueue, m_KernelDistributeElements, 1, NULL, &globalWorkSize, LocalWorkSize, 0, NULL, NULL);
	V_RETURN_CL(clErr, "Failed to start KernelDistributeElements.");
}

void QuickSortTask::Recurse(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3],
	size_t startIndex, size_t count)
{
	if (count <= 1)
	{
		return;
	}

	uniform_int_distribution<int> distribution(startIndex, startIndex + count - 1);
	int pivotIndex = distribution(m_Rnd);
	size_t groupCount = GetGroupCount(count, LocalWorkSize[0]);

	//calculate leftCount/rightCount per block
	CountElements(Context, CommandQueue, LocalWorkSize, startIndex, count, pivotIndex);
	//calculate inclusive prefix sums of leftCount/rightCount (in place)
	Scan(Context, CommandQueue, LocalWorkSize, count, m_dLeftCount, groupCount);
	Scan(Context, CommandQueue, LocalWorkSize, count, m_dRightCount, groupCount);

	int leftSize, rightSize;
	V_RETURN_CL(clEnqueueReadBuffer(CommandQueue, m_dLeftCount, CL_FALSE, (groupCount - 1) * sizeof(cl_int),
		sizeof(cl_int), &leftSize, 0, NULL, NULL), "Error reading data from device!");
	V_RETURN_CL(clEnqueueReadBuffer(CommandQueue, m_dRightCount, CL_FALSE, (groupCount - 1) * sizeof(cl_int),
		sizeof(cl_int), &rightSize, 0, NULL, NULL), "Error reading data from device!");

	//distribute elements around the pivot in the output buffer
	DistributeElements(Context, CommandQueue, LocalWorkSize, startIndex, count, pivotIndex);

	V_RETURN_CL(clEnqueueCopyBuffer(CommandQueue, m_dOutput, m_dInput, startIndex * sizeof(cl_int), startIndex * sizeof(cl_int),
		count * sizeof(cl_int), 0, NULL, NULL), "Error copying buffer.");

	Recurse(Context, CommandQueue, LocalWorkSize, startIndex, leftSize);
	Recurse(Context, CommandQueue, LocalWorkSize, startIndex + count - rightSize, rightSize);
}

void QuickSortTask::ComputeGPU(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3])
{
	CTimer timer;
	cl_int clErr;

	timer.Start();

	size_t groupCount = GetGroupCount(m_Size, LocalWorkSize[0]);
	m_dLeftCount = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(int) * groupCount, NULL, &clErr);
	V_RETURN_CL(clErr, "Buffer allocation failed.");
	m_dRightCount = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(int) * groupCount, NULL, &clErr);
	V_RETURN_CL(clErr, "Buffer allocation failed.");
	
	m_dScanPing = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(int) * groupCount, NULL, &clErr);
	V_RETURN_CL(clErr, "Buffer allocation failed.");
	m_dScanPong = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(int) * groupCount, NULL, &clErr);
	V_RETURN_CL(clErr, "Buffer allocation failed.");

	V_RETURN_CL(clEnqueueWriteBuffer(CommandQueue, m_dInput, CL_FALSE, 0, m_Size * sizeof(cl_int), m_hInput, 0, NULL, NULL),
		"Error copying data from host to device!");
	Recurse(Context, CommandQueue, LocalWorkSize, 0, m_Size);

	V_RETURN_CL(clEnqueueReadBuffer(CommandQueue, m_dOutput, CL_TRUE, 0, m_Size * sizeof(cl_int), m_hGPUResult, 0, NULL, NULL),
		"3Error reading data from device!");

	timer.Stop();
	cout << "GPU time: " << timer.GetElapsedMilliseconds() << "ms" << endl;
}

int cmpfunc(const void * a, const void * b)
{
	return (*(int*)a - *(int*)b);
}

void QuickSortTask::ComputeCPU()
{
	CTimer timer;
	timer.Start();
	qsort(m_hOutput, m_Size, sizeof(int), cmpfunc);
	timer.Stop();
	cout << "CPU time: " << timer.GetElapsedMilliseconds() << "ms" << endl;
}

bool QuickSortTask::ValidateResults()
{
    if(!(memcmp(m_hOutput, m_hGPUResult, m_Size * sizeof(int)) == 0))
    {
        cout<<"Results of the kernel are incorrect!"<<endl;
        return false;
    }
    return true;
}

///////////////////////////////////////////////////////////////////////////////
