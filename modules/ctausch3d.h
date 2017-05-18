/*!
 * \file
 * \author  Lukas Spies <LSpies@illinois.edu>
 * \version 1.0
 */

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
typedef int Edge;

typedef void* CTausch3D;

enum Edges { TAUSCH_LEFT, TAUSCH_RIGHT, TAUSCH_TOP, TAUSCH_BOTTOM, TAUSCH_FRONT, TAUSCH_BACK };

CTausch3D* tausch3d_new(int *localDim, int *mpiNum, int *haloWidth, MPI_Comm comm);
void tausch3d_delete(CTausch3D *tC);

void tausch3d_getMPICommunicator(CTausch3D *tC, MPI_Comm *comm);

void tausch3d_postCpuDataReceives(CTausch3D *tC);
void tausch3d_postCpuStencilReceives(CTausch3D *tC);

void tausch3d_performCpuToCpuData(CTausch3D *tC);
void tausch3d_performCpuToCpuStencil(CTausch3D *tC);

void tausch3d_startCpuToGpuData(CTausch3D *tC);
void tausch3d_startCpuToGpuStencil(CTausch3D *tC);

void tausch3d_startCpuDataEdge(CTausch3D *tC, Edge edge);
void tausch3d_startCpuStencilEdge(CTausch3D *tC, Edge edge);

void tausch3d_completeCpuDataEdge(CTausch3D *tC, Edge edge);
void tausch3d_completeCpuStencilEdge(CTausch3D *tC, Edge edge);

void tausch3d_setCpuData(CTausch3D *tC, real_t *dat);
void tausch3d_setCpuStencil(CTausch3D *tC, real_t *dat, int stencilNumPoints);

#ifdef TAUSCH_OPENCL
void tausch3d_enableOpenCL(CTausch3D *tC, int *gpuHaloWidth, bool blockingSyncCpuGpu, int clLocalWorkgroupSize, bool giveOpenCLDeviceName);

void tausch3d_enableOpenCLWithInfo(CTausch3D *tC, const cl_device_id *clDefaultDevice, const cl_context *clContext, const cl_command_queue *clQueue,
                                   int *gpuHaloWidth, bool blockingSyncCpuGpu, int clLocalWorkgroupSize);

void tausch3d_performCpuToCpuDataAndCpuToGpuData(CTausch3D *tC);
void tausch3d_performCpuToCpuStencilAndCpuToGpuStencil(CTausch3D *tC);

void tausch3d_performCpuToGpuData(CTausch3D *tC);
void tausch3d_performCpuToGpuStencil(CTausch3D *tC);

void tausch3d_performGpuToCpuData(CTausch3D *tC);
void tausch3d_performGpuToCpuStencil(CTausch3D *tC);

void tausch3d_startGpuToCpuData(CTausch3D *tC);
void tausch3d_startGpuToCpuStencil(CTausch3D *tC);

void tausch3d_completeCpuToGpuData(CTausch3D *tC);
void tausch3d_completeCpuToGpuStencil(CTausch3D *tC);

void tausch3d_completeGpuToCpuData(CTausch3D *tC);
void tausch3d_completeGpuToCpuStencil(CTausch3D *tC);

void tausch3d_setGpuData(CTausch3D *tC, cl_mem dat, int *gpuDim);
void tausch3d_setGpuStencil(CTausch3D *tC, cl_mem stencil, int *gpuDim, int stencilNumPoints);

cl_context tausch3d_getContext(CTausch3D *tC);
cl_command_queue tausch3d_getQueue(CTausch3D *tC);

#endif

#ifdef __cplusplus
}
#endif


#endif // CTAUSCH2D_H
