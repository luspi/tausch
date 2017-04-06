#ifndef CTAUSCH_H
#define CTAUSCH_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#ifdef OPENCL
#include <CL/cl.h>
#endif

typedef double real_t;

typedef void* CTausch;

enum Edge { TauschLeft = 0, TauschRight, TauschTop, TauschBottom };

CTausch* tausch_new(int localDimX, int localDimY, int mpiNumX, int mpiNumY, int haloWidth, MPI_Comm comm);
void tausch_delete(CTausch *tC);

void tausch_getMPICommunicator(CTausch *tC, MPI_Comm *comm);

void tausch_postCpuReceives(CTausch *tC);

void tausch_performCpuToCpu(CTausch *tC);

void tausch_startCpuToGpu(CTausch *tC);

void tausch_startCpuEdge(CTausch *tC, enum Edge edge);

void tausch_completeCpuEdge(CTausch *tC, enum Edge edge);

void tausch_setCPUData(CTausch *tC, real_t *dat);

#ifdef TAUSCH_OPENCL
void tausch_enableOpenCL(CTausch *tC, bool blockingSyncCpuGpu, int clLocalWorkgroupSize, bool giveOpenCLDeviceName);

void tausch_setOpenCLInfo(CTausch *tC, const cl_device_id *clDefaultDevice, const cl_context *clContext, const cl_command_queue *clQueue, bool blockingSyncCpuGpu);

void tausch_performCpuToCpuAndCpuToGpu(CTausch *tC);

void tausch_performCpuToGpu(CTausch *tC);

void tausch_performGpuToCpu(CTausch *tC);

void tausch_startGpuToCpu(CTausch *tC);

void tausch_completeCpuToGpu(CTausch *tC);
void tausch_completeGpuToCpu(CTausch *tC);

void tausch_setGPUData(CTausch *tC, cl_mem dat, int gpuDimX, int gpuDimY);

cl_context tausch_getContext(CTausch *tC);
cl_command_queue tausch_getQueue(CTausch *tC);

#endif

#ifdef __cplusplus
}
#endif


#endif // CTAUSCH_H
