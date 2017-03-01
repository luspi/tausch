#ifndef CTAUSCH_H
#define CTAUSCH_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <CL/cl.h>

typedef void* CTausch;

enum { Left = 0, Right, Top, Bottom };
typedef int Edge;

CTausch* tausch_new(int localDimX, int localDimY, int mpiNumX, int mpiNumY, bool blockingSyncCpuGpu, bool withOpenCL, bool setupOpenCL, int clLocalWorkgroupSize, bool giveOpenCLDeviceName);
CTausch* tausch_newCpuAndGpu(int localDimX, int localDimY, int mpiNumX, int mpiNumY, bool blockingSyncCpuGpu, bool withOpenCL, bool setupOpenCL, int clLocalWorkgroupSize, bool giveOpenCLDeviceName);
CTausch* tausch_newCpu(int localDimX, int localDimY, int mpiNumX, int mpiNumY);
void tausch_delete(CTausch *tC);

void tausch_setOpenCLInfo(CTausch *tC, const cl_device_id *clDefaultDevice, const cl_context *clContext, const cl_command_queue *clQueue);

void tausch_postCpuReceives(CTausch *tC);

void tausch_performCpuToCpu(CTausch *tC);

void tausch_performCpuToCpuAndCpuToGpu(CTausch *tC);

void tausch_performCpuToGpu(CTausch *tC);

void tausch_performGpuToCpu(CTausch *tC);

void tausch_startCpuToGpu(CTausch *tC);
void tausch_startGpuToCpu(CTausch *tC);

void tausch_completeCpuToGpu(CTausch *tC);
void tausch_completeGpuToCpu(CTausch *tC);

void tausch_startCpuEdge(CTausch *tC, Edge edge);

void tausch_completeCpuEdge(CTausch *tC, Edge edge);

void tausch_syncCpuAndGpu(CTausch *tC, bool iAmTheCPU);


void tausch_setHaloWidth(CTausch *tC, int haloWidth);
void tausch_setCPUData(CTausch *tC, double *dat);
void tausch_setGPUData(CTausch *tC, cl_mem dat, int gpuDimX, int gpuDimY);
bool tausch_isGpuEnabled(CTausch *tC);

cl_context tausch_getContext(CTausch *tC);
cl_command_queue tausch_getQueue(CTausch *tC);

void tausch_checkOpenCLError(CTausch *tC, cl_int clErr, char *loc);

#ifdef __cplusplus
}
#endif


#endif // CTAUSCH_H
