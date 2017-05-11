#ifndef CTAUSCH2D_H
#define CTAUSCH2D_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#ifdef TAUSCH_OPENCL
#include <CL/cl.h>
#endif

/*!
 * Use real_t in code to allow easier switch between double/float.
 */
typedef double real_t;

typedef void* CTausch2D;

enum Edge { TAUSCH_LEFT, TAUSCH_RIGHT, TAUSCH_TOP, TAUSCH_BOTTOM };

CTausch2D* tausch2d_new(int localDim[2], int mpiNum[2], int cpuHaloWidth[4], MPI_Comm comm);
void tausch2d_delete(CTausch2D *tC);

void tausch2d_getMPICommunicator(CTausch2D *tC, MPI_Comm *comm);

void tausch2d_postCpuDataReceives(CTausch2D *tC);
void tausch2d_postCpuStencilReceives(CTausch2D *tC);

void tausch2d_performCpuToCpuData(CTausch2D *tC);
void tausch2d_performCpuToCpuStencil(CTausch2D *tC);

void tausch2d_startCpuToGpu(CTausch2D *tC);

void tausch2d_startCpuDataEdge(CTausch2D *tC, enum Edge edge);
void tausch2d_startCpuStencilEdge(CTausch2D *tC, enum Edge edge);

void tausch2d_completeCpuDataEdge(CTausch2D *tC, enum Edge edge);
void tausch2d_completeCpuStencilEdge(CTausch2D *tC, enum Edge edge);

void tausch2d_setCPUData(CTausch2D *tC, real_t *dat);
void tausch2d_setCPUStencil(CTausch2D *tC, real_t *dat, int stencilNumPoints);

#ifdef TAUSCH_OPENCL
void tausch2d_enableOpenCL(CTausch2D *tC, int gpuHaloWidth[4], bool blockingSyncCpuGpu, int clLocalWorkgroupSize, bool giveOpenCLDeviceName);

void tausch2d_setOpenCLInfo(CTausch2D *tC, const cl_device_id *clDefaultDevice, const cl_context *clContext, const cl_command_queue *clQueue, int gpuHaloWidth[4], bool blockingSyncCpuGpu);

void tausch2d_performCpuToCpuDataAndCpuToGpuData(CTausch2D *tC);
void tausch2d_performCpuToCpuStencilAndCpuToGpuStencil(CTausch2D *tC);

void tausch2d_performCpuToGpuData(CTausch2D *tC);
void tausch2d_performCpuToGpuStencil(CTausch2D *tC);

void tausch2d_performGpuToCpuData(CTausch2D *tC);
void tausch2d_performGpuToCpuStencil(CTausch2D *tC);

void tausch2d_startGpuToCpuData(CTausch2D *tC);
void tausch2d_startGpuToCpuStencil(CTausch2D *tC);

void tausch2d_completeCpuToGpuData(CTausch2D *tC);
void tausch2d_completeCpuToGpuStencil(CTausch2D *tC);

void tausch2d_completeGpuToCpuData(CTausch2D *tC);
void tausch2d_completeGpuToCpuStencil(CTausch2D *tC);

void tausch2d_setGPUData(CTausch2D *tC, cl_mem dat, int gpuDimX, int gpuDimY);
void tausch2d_setGPUStencil(CTausch2D *tC, cl_mem stencil, int stencilNumPoints, int gpuDimX, int gpuDimY);

cl_context tausch2d_getContext(CTausch2D *tC);
cl_command_queue tausch2d_getQueue(CTausch2D *tC);

#endif

#ifdef __cplusplus
}
#endif


#endif // CTAUSCH2D_H
