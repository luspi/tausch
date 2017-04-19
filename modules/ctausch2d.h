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

CTausch2D* tausch2d_new(int localDimX, int localDimY, int mpiNumX, int mpiNumY, int haloWidth, MPI_Comm comm);
void tausch2d_delete(CTausch2D *tC);

void tausch2d_getMPICommunicator(CTausch2D *tC, MPI_Comm *comm);

void tausch2d_postCpuReceives(CTausch2D *tC);

void tausch2d_performCpuToCpu(CTausch2D *tC);

void tausch2d_startCpuToGpu(CTausch2D *tC);

void tausch2d_startCpuEdge(CTausch2D *tC, enum Edge edge);

void tausch2d_completeCpuEdge(CTausch2D *tC, enum Edge edge);

void tausch2d_setCPUData(CTausch2D *tC, real_t *dat);

#ifdef TAUSCH_OPENCL
void tausch2d_enableOpenCL(CTausch2D *tC, bool blockingSyncCpuGpu, int clLocalWorkgroupSize, bool giveOpenCLDeviceName);

void tausch2d_setOpenCLInfo(CTausch2D *tC, const cl_device_id *clDefaultDevice, const cl_context *clContext, const cl_command_queue *clQueue, bool blockingSyncCpuGpu);

void tausch2d_performCpuToCpuAndCpuToGpu(CTausch2D *tC);

void tausch2d_performCpuToGpu(CTausch2D *tC);

void tausch2d_performGpuToCpu(CTausch2D *tC);

void tausch2d_startGpuToCpu(CTausch2D *tC);

void tausch2d_completeCpuToGpu(CTausch2D *tC);
void tausch2d_completeGpuToCpu(CTausch2D *tC);

void tausch2d_setGPUData(CTausch2D *tC, cl_mem dat, int gpuDimX, int gpuDimY);

cl_context tausch2d_getContext(CTausch2D *tC);
cl_command_queue tausch2d_getQueue(CTausch2D *tC);

#endif

#ifdef __cplusplus
}
#endif


#endif // CTAUSCH2D_H
