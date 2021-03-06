/******************************************************************************
                         .88888.   888888ba  dP     dP 
                        d8'   `88  88    `8b 88     88 
                        88        a88aaaa8P' 88     88 
                        88   YP88  88        88     88 
                        Y8.   .88  88        Y8.   .8P 
                         `88888'   dP        `Y88888P' 
                                                       
                                                       
   a88888b.                                         dP   oo                   
  d8'   `88                                         88                        
  88        .d8888b. 88d8b.d8b. 88d888b. dP    dP d8888P dP 88d888b. .d8888b. 
  88        88'  `88 88'`88'`88 88'  `88 88    88   88   88 88'  `88 88'  `88 
  Y8.   .88 88.  .88 88  88  88 88.  .88 88.  .88   88   88 88    88 88.  .88 
   Y88888P' `88888P' dP  dP  dP 88Y888P' `88888P'   dP   dP dP    dP `8888P88 
                                88                                        .88 
                                dP                                    d8888P  
******************************************************************************/

#ifndef _QUICK_SORT_TASK
#define _QUICK_SORT_TASK

#include "../Common/IComputeTask.h"
#include <random>

class QuickSortTask : public IComputeTask
{
public:
	QuickSortTask(size_t size, int leftBound, int rightBound);
	virtual ~QuickSortTask();

	// IComputeTask
	virtual bool InitResources(cl_device_id Device, cl_context Context);
	
	virtual void ReleaseResources();

	void CountElements(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3], size_t startIndex, size_t count, size_t pivotIndex);

	void Scan(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3], size_t count, cl_mem input, int groupCount);

	void DistributeElements(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3], size_t startIndex, size_t count, size_t pivotIndex);

	void Recurse(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3], size_t startIndex, size_t count);

	virtual void ComputeGPU(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3]);

	virtual void ComputeCPU();

	virtual bool ValidateResults();

protected:
	//NOTE: we have two memory address spaces, so we mark pointers with a prefix
	//to avoid confusions: 'h' - host, 'd' - device

	unsigned int		m_Size;

	int				    *m_hInput, *m_hOutput;

	//pointers on the GPU
	cl_mem				m_dInput, m_dOutput, m_dLeftCount, m_dRightCount, m_dScanPing, m_dScanPong;
	//(..and a pointer to read back the result)
	int 				*m_hGPUResult;

	//OpenCL program and kernels
	cl_program			m_Program;
	cl_kernel			m_KernelScan;
	cl_kernel			m_KernelCountElements;
	cl_kernel			m_KernelDistributeElements;

	int					m_LeftBound, m_RightBound;
	std::default_random_engine m_Rnd;
};

#endif // _CMATRIX_ROTATE_TASK_H
