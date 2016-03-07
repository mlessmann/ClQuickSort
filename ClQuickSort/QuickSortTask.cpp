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
    m_dOutput(NULL), m_hGPUResult(NULL), m_Program(NULL), m_Kernel(NULL)
{
}

QuickSortTask::~QuickSortTask()
{
    ReleaseResources();
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
		m_hInput[i] = rand();
    }
	memcpy(m_hOutput, m_hInput, m_Size * sizeof(int));

    cl_int clError;

	//TODO create buffers
    /*m_dM = clCreateBuffer(Context, CL_MEM_READ_ONLY, sizeof(float) * m_SizeX * m_SizeY, NULL, &clError);
    V_RETURN_FALSE_CL(clError, "Buffer allocation failed.");
    m_dMR = clCreateBuffer(Context, CL_MEM_WRITE_ONLY, sizeof(float) * m_SizeX * m_SizeY, NULL, &clError);
    V_RETURN_FALSE_CL(clError, "Buffer allocation failed.");*/

    size_t programSize = 0;
    string programCode;

    if (!CLUtil::LoadProgramSourceToMemory("QuickSort.cl", programCode))
        return false;

    m_Program = CLUtil::BuildCLProgramFromMemory(Device, Context, programCode);
    if (m_Program == nullptr)
        return false;

    m_Kernel = clCreateKernel(m_Program, "QuickSort", &clError);
    V_RETURN_FALSE_CL(clError, "Failed to create kernel: QuickSort");

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

    SAFE_RELEASE_KERNEL(m_Kernel);
    SAFE_RELEASE_PROGRAM(m_Program);
}

void QuickSortTask::ComputeGPU(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3])
{
	//TODO
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
