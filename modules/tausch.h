#ifndef TAUSCHBASE_H
#define TAUSCHBASE_H

#ifdef TAUSCH_OPENCL
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#endif

typedef double real_t;
typedef int Edge;

/*!
 * Virtual API, allowing runtime choice of 2D or 3D version.
 */
class Tausch {

public:
    virtual void setCpuData(real_t *data) = 0;
    virtual void setCpuStencil(real_t *stencil, int stencilNumPoints) = 0;
    virtual void postCpuDataReceives() = 0;
    virtual void postCpuStencilReceives() = 0;
    virtual void performCpuToCpuData() = 0;
    virtual void performCpuToCpuStencil() = 0;
    virtual void startCpuDataEdge(Edge edge) = 0;
    virtual void startCpuStencilEdge(Edge edge) = 0;
    virtual void completeCpuDataEdge(Edge edge) = 0;
    virtual void completeCpuStencilEdge(Edge edge) = 0;
    virtual MPI_Comm getMPICommunicator() = 0;
    virtual void info() = 0;

#ifdef TAUSCH_OPENCL
    virtual void enableOpenCL(int *gpuHaloWidth, bool blockingSyncCpuGpu = true, int clLocalWorkgroupSize = 64, bool giveOpenCLDeviceName = false) = 0;
    virtual void enableOpenCL(int gpuHaloWidth, bool blockingSyncCpuGpu, int clLocalWorkgroupSize, bool giveOpenCLDeviceName) = 0;
    virtual void enableOpenCL(cl::Device &cl_defaultDevice, cl::Context &cl_context, cl::CommandQueue &cl_queue,
                              int *gpuHaloWidth, bool blockingSyncCpuGpu, int clLocalWorkgroupSize) = 0;
    virtual void enableOpenCL(cl::Device &cl_defaultDevice, cl::Context &cl_context, cl::CommandQueue &cl_queue,
                              int gpuHaloWidth, bool blockingSyncCpuGpu, int clLocalWorkgroupSize) = 0;
    virtual void setGpuData(cl::Buffer &dat, int *gpuDim) = 0;
    virtual void setGpuStencil(cl::Buffer &stencil, int stencilNumPoints, int *stencilDim = nullptr) = 0;
    virtual void performCpuToGpuData() = 0;
    virtual void performCpuToGpuStencil() = 0;
    virtual void performCpuToCpuDataAndCpuToGpuData() = 0;
    virtual void performCpuToCpuStencilAndCpuToGpuStencil() = 0;
    virtual void performGpuToCpuData() = 0;
    virtual void performGpuToCpuStencil() = 0;
    virtual void startCpuToGpuData() = 0;
    virtual void startCpuToGpuStencil() = 0;
    virtual void startGpuToCpuData() = 0;
    virtual void startGpuToCpuStencil() = 0;
    virtual void completeCpuToGpuData() = 0;
    virtual void completeCpuToGpuStencil() = 0;
    virtual void completeGpuToCpuData() = 0;
    virtual void completeGpuToCpuStencil() = 0;
    virtual cl::Context getContext() = 0;
    virtual cl::CommandQueue getQueue() = 0;
#endif

};


#endif // TAUSCHBASE_H
