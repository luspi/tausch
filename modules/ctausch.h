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

enum TAUSCH_VERSION {
    TAUSCH_1D_VERSION,
    TAUSCH_2D_VERSION,
    TAUSCH_3D_VERSION
};

typedef void* CTausch;

CTausch *tausch_new(int *localDim, int *haloWidth, int numBuffers, int valuesPerPoint, MPI_Comm comm, TAUSCH_VERSION version);
void tausch_delete(CTausch *tC);

void tausch_setCpuLocalHaloInfo(CTausch *tC, int numHaloParts, int **haloSpecs);
void tausch_setCpuRemoteHaloInfo(CTausch *tC, int numHaloParts, int **haloSpecs);
void tausch_postMpiReceives(CTausch *tC);
void tausch_packNextSendBuffer(CTausch *tC, int id, double *buf);
void tausch_send(CTausch *tC, int id);
void tausch_recv(CTausch *tC, int id);
void tausch_unpackNextRecvBuffer(CTausch *tC, int id, double *buf);
void tausch_packAndSend(CTausch *tC, int id, double *buf);
void tausch_recvAndUnpack(CTausch *tC, int id, double *buf);

#ifdef __cplusplus
}
#endif


#endif // CTAUSCH2D_H
