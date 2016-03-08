/******************************************************************************
GPU Computing / GPGPU Praktikum source code.

******************************************************************************/

#include "QuickSortTask.h"

#include "../Common/CLUtil.h"

#include <string.h>

using namespace std;

///////////////////////////////////////////////////////////////////////////////
// QuickSortTask

QuickSortTask::QuickSortTask(size_t size)
    :m_Size(size), m_hInput(NULL), m_hOutput(NULL), m_dInput(NULL),
    m_dOutput(NULL), m_hGPUResult(NULL), m_Program(NULL), m_KernelScan(NULL),
	m_Kernel1(NULL), m_Kernel2(NULL), m_dLeftCount(NULL), m_dRightCount(NULL)
{
}

QuickSortTask::~QuickSortTask()
{
    ReleaseResources();
}

size_t GetGroupCount(size_t size, size_t localWorkSize)
{
	return ceil((double)size / (double)localWorkSize);
}

bool QuickSortTask::InitResources(cl_device_id Device, cl_context Context)
{
    //CPU resources
    m_hInput = new int[m_Size];
    m_hOutput = new int[m_Size];
    m_hGPUResult = new int[m_Size];

    //fill the array with random ints
    for(unsigned int i = 0; i < m_Size; i++)
    {
		m_hInput[i] = rand() & 15;
    }
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

	m_Kernel1 = clCreateKernel(m_Program, "QuickSort1", &clError);
	V_RETURN_FALSE_CL(clError, "Failed to create kernel: QuickSort");
	m_KernelScan = clCreateKernel(m_Program, "Scan_Naive", &clError);
	V_RETURN_FALSE_CL(clError, "Failed to create kernel: Scan_Naive");

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

	SAFE_RELEASE_KERNEL(m_KernelScan);
	SAFE_RELEASE_KERNEL(m_Kernel1);
	SAFE_RELEASE_KERNEL(m_Kernel2);
    SAFE_RELEASE_PROGRAM(m_Program);
}

void QuickSortTask::Kernel1(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3],
	size_t startIndex, size_t count)
{
	cl_int clErr;
	size_t globalWorkSize = CLUtil::GetGlobalWorkSize(count, LocalWorkSize[0]);

	int pivotIndex = startIndex + (rand() % count);

	clErr = clSetKernelArg(m_Kernel1, 0, sizeof(cl_mem), (void*)&m_dInput);
	clErr |= clSetKernelArg(m_Kernel1, 1, sizeof(cl_mem), (void*)&m_dOutput);
	clErr |= clSetKernelArg(m_Kernel1, 2, sizeof(cl_int), (void*)&startIndex);
	clErr |= clSetKernelArg(m_Kernel1, 3, sizeof(cl_int), (void*)&count);
	clErr |= clSetKernelArg(m_Kernel1, 4, sizeof(cl_int), (void*)&pivotIndex);
	clErr |= clSetKernelArg(m_Kernel1, 5, sizeof(cl_mem), (void*)&m_dLeftCount);
	clErr |= clSetKernelArg(m_Kernel1, 6, sizeof(cl_mem), (void*)&m_dRightCount);
	V_RETURN_CL(clErr, "Failed to set kernel args for Kernel1.");

	clErr = clEnqueueNDRangeKernel(CommandQueue, m_Kernel1, 1, NULL, &globalWorkSize, LocalWorkSize, 0, NULL, NULL);
	V_RETURN_CL(clErr, "Failed to start Kernel1.");
}

void QuickSortTask::Scan(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3],
	size_t count, cl_mem &input)
{
	cl_int clErr;
	size_t groupCount = GetGroupCount(count, LocalWorkSize[0]);

	cl_mem output = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(int) * groupCount, NULL, &clErr);
	V_RETURN_CL(clErr, "Buffer allocation failed.");

	for (unsigned int offset = 1; offset <= groupCount; offset *= 2)
	{
		size_t globalWorkSize = CLUtil::GetGlobalWorkSize(groupCount, LocalWorkSize[0]);

		clErr = clSetKernelArg(m_KernelScan, 0, sizeof(cl_mem), (void*)&input);
		clErr |= clSetKernelArg(m_KernelScan, 1, sizeof(cl_mem), (void*)&output);
		clErr |= clSetKernelArg(m_KernelScan, 2, sizeof(cl_int), (void*)&groupCount);
		clErr |= clSetKernelArg(m_KernelScan, 3, sizeof(cl_int), (void*)&offset);
		V_RETURN_CL(clErr, "Failed to set kernel args: KernelScan");

		clErr = clEnqueueNDRangeKernel(CommandQueue, m_KernelScan, 1, NULL, &globalWorkSize, LocalWorkSize, 0, NULL, NULL);
		V_RETURN_CL(clErr, "Failed to start KernelScan.");
		swap(input, output);
	}

	SAFE_RELEASE_MEMOBJECT(output);
}

void QuickSortTask::Recurse(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3],
	size_t startIndex, size_t count)
{
	//calculate leftCount/rightCount per block
	Kernel1(Context, CommandQueue, LocalWorkSize, startIndex, count);
	//calculate prefix sums of leftCount/rightCount (in place)
	Scan(Context, CommandQueue, LocalWorkSize, count, m_dLeftCount);
	Scan(Context, CommandQueue, LocalWorkSize, count, m_dRightCount);

	/*size_t groupCount = GetGroupCount(count, LocalWorkSize[0]);
	int* test = new int[groupCount];
	V_RETURN_CL(clEnqueueReadBuffer(CommandQueue, m_dLeftCount, CL_TRUE, 0, groupCount * sizeof(cl_int), test, 0, NULL, NULL),
		"Error reading data from device!");
	for (int j = 0; j < groupCount; j++) {
		cout << test[j] << " ";
	}
	cout << endl;*/

	//start kernel 2
	//read pivot index
	swap(m_dInput, m_dOutput);
	//recurse left, right
}

void QuickSortTask::ComputeGPU(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3])
{
	cl_int clErr;
	size_t groupCount = GetGroupCount(m_Size, LocalWorkSize[0]);
	m_dLeftCount = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(int) * groupCount, NULL, &clErr);
	V_RETURN_CL(clErr, "Buffer allocation failed.");
	m_dRightCount = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(int) * groupCount, NULL, &clErr);
	V_RETURN_CL(clErr, "Buffer allocation failed.");

	V_RETURN_CL(clEnqueueWriteBuffer(CommandQueue, m_dInput, CL_FALSE, 0, m_Size * sizeof(cl_int), m_hInput, 0, NULL, NULL),
		"Error copying data from host to device!");
	Recurse(Context, CommandQueue, LocalWorkSize, 0, m_Size);
}

int cmpfunc(const void * a, const void * b)
{
	return (*(int*)a - *(int*)b);
}

void QuickSortTask::ComputeCPU()
{
	qsort(m_hOutput, m_Size, sizeof(int), cmpfunc);
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
