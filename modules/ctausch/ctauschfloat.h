/*!
 * \file
 * \author  Lukas Spies <LSpies@illinois.edu>
 * \version 1.0
 *
 * \brief
 *  C wrapper to C++ API, float datatype.
 *
 *  C wrapper to C++ API, float datatype. It provides a single interface for all three versions (1D, 2D, 3D). It is possible to choose at runtime
 *  which version to use (using enum).
 */

#ifndef CTAUSCHFLOAT_H
#define CTAUSCHFLOAT_H

#ifdef __cplusplus
extern "C" {
#endif

#include "../tausch.h"

#ifdef TAUSCH_OPENCL
#include <CL/cl.h>
#endif

#ifndef TAUSCHVERSIONDEF
#define TAUSCHVERSIONDEF
enum TauschVersion {
    TAUSCH_1D,
    TAUSCH_2D,
    TAUSCH_3D
};
#endif // TAUSCHVERSIONDEF

typedef void* CTauschFloat;

CTauschFloat *tausch_new_float(size_t numBuffers, size_t *valuesPerPointPerBuffer, MPI_Comm comm, TauschVersion version);

void tausch_delete_float(CTauschFloat *tC);
void tausch_setLocalHaloInfo_float(CTauschFloat *tC, TauschDeviceDirection flags, size_t numHaloParts, TauschHaloSpec *haloSpecs);
void tausch_setRemoteHaloInfo_float(CTauschFloat *tC, TauschDeviceDirection flags, size_t numHaloParts, TauschHaloSpec *haloSpecs);
void tausch_postReceive_float(CTauschFloat *tC, TauschDeviceDirection flags, size_t haloId, int mpitag);
void tausch_postAllReceives_float(CTauschFloat *tC, TauschDeviceDirection flags, int *mpitag);
void tausch_packSendBuffer_float(CTauschFloat *tC, TauschDeviceDirection flags, size_t haloId, size_t bufferId, float *buf, cl_mem *bufcl, TauschPackRegion region);
void tausch_send_float(CTauschFloat *tC, TauschDeviceDirection flags, size_t haloId, int mpitag);
void tausch_recv_float(CTauschFloat *tC, TauschDeviceDirection flags, size_t haloId);
void tausch_unpackNextRecvBuffer_float(CTauschFloat *tC, TauschDeviceDirection flags, size_t haloId, size_t bufferId, float *buf, cl_mem *bufcl, TauschPackRegion region);
void tausch_packAndSend_float(CTauschFloat *tC, TauschDeviceDirection flags, size_t haloId, float *buf, cl_mem *bufcl, TauschPackRegion region, int mpitag);
void tausch_recvAndUnpack_float(CTauschFloat *tC, TauschDeviceDirection flags, size_t haloId, float *buf, cl_mem *bufcl, TauschPackRegion region);

#ifdef __cplusplus
}
#endif


#endif // CTAUSCHFLOAT_H
