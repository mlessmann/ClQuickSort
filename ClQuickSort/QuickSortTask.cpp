/******************************************************************************
GPU Computing / GPGPU Praktikum source code.

******************************************************************************/

#include "QuickSortTask.h"

#include "../Common/CLUtil.h"

#include <string.h>

using namespace std;

///////////////////////////////////////////////////////////////////////////////
// CMatrixRotateTask

QuickSortTask::QuickSortTask(size_t SizeX, size_t SizeY)
    :m_SizeX(SizeX), m_SizeY(SizeY), m_hM(NULL), m_hMR(NULL), m_dM(NULL),
    m_dMR(NULL), m_hGPUResultNaive(NULL), m_hGPUResultOpt(NULL), m_Program(NULL),
    m_NaiveKernel(NULL), m_OptimizedKernel(NULL)
{
}

QuickSortTask::~QuickSortTask()
{
    ReleaseResources();
}

bool QuickSortTask::InitResources(cl_device_id Device, cl_context Context)
{
    //CPU resources
    m_hM = new float[m_SizeX * m_SizeY];
    m_hMR = new float[m_SizeX * m_SizeY];
    m_hGPUResultNaive = new float[m_SizeX * m_SizeY];
    m_hGPUResultOpt = new float[m_SizeX * m_SizeY];

    //fill the matrix with random floats
    for(unsigned int i = 0; i < m_SizeX * m_SizeY; i++)
    {
        m_hM[i] = float(rand()) / float(RAND_MAX);
    }

    // TO DO: allocate all device resources here
    cl_int clError;

    m_dM = clCreateBuffer(Context, CL_MEM_READ_ONLY, sizeof(float) * m_SizeX * m_SizeY, NULL, &clError);
    V_RETURN_FALSE_CL(clError, "Buffer allocation failed.");
    m_dMR = clCreateBuffer(Context, CL_MEM_WRITE_ONLY, sizeof(float) * m_SizeX * m_SizeY, NULL, &clError);
    V_RETURN_FALSE_CL(clError, "Buffer allocation failed.");

    size_t programSize = 0;
    string programCode;

    if (!CLUtil::LoadProgramSourceToMemory("MatrixRot.cl", programCode))
        return false;

    m_Program = CLUtil::BuildCLProgramFromMemory(Device, Context, programCode);
    if (m_Program == nullptr)
        return false;

    m_NaiveKernel = clCreateKernel(m_Program, "MatrixRotNaive", &clError);
    V_RETURN_FALSE_CL(clError, "Failed to create kernel: MatrixRotNaive");
    m_OptimizedKernel = clCreateKernel(m_Program, "MatrixRotOptimized", &clError);
    V_RETURN_FALSE_CL(clError, "Failed to create kernel: MatrixRotOptimized");

    return true;
}

void QuickSortTask::ReleaseResources()
{
    //CPU resources
    SAFE_DELETE_ARRAY(m_hM);
    SAFE_DELETE_ARRAY(m_hMR);
    SAFE_DELETE_ARRAY(m_hGPUResultNaive);
    SAFE_DELETE_ARRAY(m_hGPUResultOpt);

    SAFE_RELEASE_MEMOBJECT(m_dM);
    SAFE_RELEASE_MEMOBJECT(m_dMR);

    SAFE_RELEASE_KERNEL(m_NaiveKernel);
    SAFE_RELEASE_KERNEL(m_OptimizedKernel);
    SAFE_RELEASE_PROGRAM(m_Program);
}

void QuickSortTask::ComputeGPU(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3])
{
    V_RETURN_CL(
        clEnqueueWriteBuffer(CommandQueue, m_dM, CL_FALSE, 0, m_SizeX * m_SizeY * sizeof(float), m_hM, 0, NULL, NULL),
        "Error writing buffer M.");

    V_RETURN_CL(clSetKernelArg(m_NaiveKernel, 0, sizeof(cl_mem), (void*)&m_dM), "Failed to set kernel args: Naive");
    V_RETURN_CL(clSetKernelArg(m_NaiveKernel, 1, sizeof(cl_mem), (void*)&m_dMR), "Failed to set kernel args: Naive");
    V_RETURN_CL(clSetKernelArg(m_NaiveKernel, 2, sizeof(cl_uint), (void*)&m_SizeX), "Failed to set kernel args: Naive");
    V_RETURN_CL(clSetKernelArg(m_NaiveKernel, 3, sizeof(cl_uint), (void*)&m_SizeY), "Failed to set kernel args: Naive");

    //launch kernels

    size_t globalWorkSize[2];
    size_t nGroups[2];
    globalWorkSize[0] = CLUtil::GetGlobalWorkSize(m_SizeX, LocalWorkSize[0]);
    globalWorkSize[1] = CLUtil::GetGlobalWorkSize(m_SizeY, LocalWorkSize[1]);
    nGroups[0] = globalWorkSize[0] / LocalWorkSize[0];
    nGroups[1] = globalWorkSize[1] / LocalWorkSize[1];
    cout << "Executing (" << globalWorkSize[0] << " x " << globalWorkSize[1] << ") threads in ("
        << nGroups[0] << " x " << nGroups[1] << ") groups of size (" << LocalWorkSize[0]
        << " x " << LocalWorkSize[1] << ")." << endl;

    double time = 0;
    
    //naive kernel
    time = CLUtil::ProfileKernel(CommandQueue, m_NaiveKernel, 2, globalWorkSize, LocalWorkSize, 100);
    cout << "Executed naive kernel in " << time << " ms." << endl;
    
    //this command has to be blocking, since we want to check the valid data
    V_RETURN_CL(
        clEnqueueReadBuffer(CommandQueue, m_dMR, CL_TRUE, 0, m_SizeX * m_SizeY * sizeof(float), m_hGPUResultNaive, 0, NULL, NULL),
        "Error reading data.");

    //optimized kernel
    
    V_RETURN_CL(clSetKernelArg(m_OptimizedKernel, 0, sizeof(cl_mem), (void*)&m_dM), "Failed to set kernel args: optimized");
    V_RETURN_CL(clSetKernelArg(m_OptimizedKernel, 1, sizeof(cl_mem), (void*)&m_dMR), "Failed to set kernel args: optimized");
    V_RETURN_CL(clSetKernelArg(m_OptimizedKernel, 2, sizeof(cl_uint), (void*)&m_SizeX), "Failed to set kernel args: optimized");
    V_RETURN_CL(clSetKernelArg(m_OptimizedKernel, 3, sizeof(cl_uint), (void*)&m_SizeY), "Failed to set kernel args: optimized");
    V_RETURN_CL(clSetKernelArg(m_OptimizedKernel, 4, LocalWorkSize[0] * LocalWorkSize[1] * sizeof(float), NULL), "Failed to allocate shared memory.");

    // run kernel
    time = CLUtil::ProfileKernel(CommandQueue, m_OptimizedKernel, 2, globalWorkSize, LocalWorkSize, 100);
    cout << "Executed optimized kernel in " << time << " ms." << endl;

    V_RETURN_CL(
        clEnqueueReadBuffer(CommandQueue, m_dMR, CL_TRUE, 0, m_SizeX * m_SizeY * sizeof(float), m_hGPUResultOpt, 0, NULL, NULL),
        "Error reading data.");
}

void QuickSortTask::ComputeCPU()
{
    for(unsigned int x = 0; x < m_SizeX; x++)
    {
        for(unsigned int y = 0; y < m_SizeY; y++)
        {
            m_hMR[ x * m_SizeY + (m_SizeY - y - 1) ] = m_hM[ y * m_SizeX + x ];
        }
    }
}

bool QuickSortTask::ValidateResults()
{
    if(!(memcmp(m_hMR, m_hGPUResultNaive, m_SizeX * m_SizeY * sizeof(float)) == 0))
    {
        cout<<"Results of the naive kernel are incorrect!"<<endl;
        return false;
    }
    if(!(memcmp(m_hMR, m_hGPUResultOpt, m_SizeX * m_SizeY * sizeof(float)) == 0))
    {
        cout<<"Results of the optimized kernel are incorrect!"<<endl;
        return false;
    }
    return true;
}

///////////////////////////////////////////////////////////////////////////////
