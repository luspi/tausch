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

typedef void* CTausch3D;

enum Edge { TAUSCH_LEFT, TAUSCH_RIGHT, TAUSCH_TOP, TAUSCH_BOTTOM, TAUSCH_FRONT, TAUSCH_BACK };

CTausch3D* tausch3d_new(int *localDim, int *mpiNum, int *haloWidth, MPI_Comm comm);
void tausch3d_delete(CTausch3D *tC);

void tausch3d_getMPICommunicator(CTausch3D *tC, MPI_Comm *comm);

void tausch3d_postCpuReceives(CTausch3D *tC);

void tausch3d_performCpuToCpu(CTausch3D *tC);

void tausch3d_startCpuToGpu(CTausch3D *tC);

void tausch3d_startCpuEdge(CTausch3D *tC, enum Edge edge);

void tausch3d_completeCpuEdge(CTausch3D *tC, enum Edge edge);

void tausch3d_setCPUData(CTausch3D *tC, real_t *dat);

#ifdef TAUSCH_OPENCL
void tausch3d_enableOpenCL(CTausch3D *tC, int *gpuHaloWidth, bool blockingSyncCpuGpu, int clLocalWorkgroupSize, bool giveOpenCLDeviceName);

void tausch3d_setOpenCLInfo(CTausch3D *tC, const cl_device_id *clDefaultDevice, const cl_context *clContext, const cl_command_queue *clQueue, int *gpuHaloWidth, bool blockingSyncCpuGpu, int clLocalWorkgroupSize);

void tausch3d_performCpuToCpuAndCpuToGpu(CTausch3D *tC);

void tausch3d_performCpuToGpu(CTausch3D *tC);

void tausch3d_performGpuToCpu(CTausch3D *tC);

void tausch3d_startGpuToCpu(CTausch3D *tC);

void tausch3d_completeCpuToGpu(CTausch3D *tC);
void tausch3d_completeGpuToCpu(CTausch3D *tC);

void tausch3d_setGPUData(CTausch3D *tC, cl_mem dat, int *gpuDim);

cl_context tausch3d_getContext(CTausch3D *tC);
cl_command_queue tausch3d_getQueue(CTausch3D *tC);

#endif

#ifdef __cplusplus
}
#endif


#endif // CTAUSCH2D_H
