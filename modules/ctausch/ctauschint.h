/*!
 * \file
 * \author  Lukas Spies <LSpies@illinois.edu>
 * \version 1.0
 *
 * \brief
 *  C wrapper to C++ API, int datatype.
 *
 *  C wrapper to C++ API, int datatype. It provides a single interface for all three versions (1D, 2D, 3D). It is possible to choose at runtime
 *  which version to use (using enum).
 */

#ifndef CTAUSCHINT_H
#define CTAUSCHINT_H

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

typedef void* CTauschInt;

CTauschInt *tausch_new_int(size_t numBuffers, size_t *valuesPerPointPerBuffer, MPI_Comm comm, TauschVersion version);

void tausch_delete_int(CTauschInt *tC);
void tausch_setLocalHaloInfo_int(CTauschInt *tC, TauschDeviceDirection flags, size_t numHaloParts, TauschHaloSpec *haloSpecs);
void tausch_setRemoteHaloInfo_int(CTauschInt *tC, TauschDeviceDirection flags, size_t numHaloParts, TauschHaloSpec *haloSpecs);
void tausch_postReceive_int(CTauschInt *tC, TauschDeviceDirection flags, size_t haloId, int mpitag);
void tausch_postAllReceives_int(CTauschInt *tC, TauschDeviceDirection flags, int *mpitag);
#ifdef TAUSCH_OPENCL
void tausch_packSendBuffer_int(CTauschInt *tC, TauschDeviceDirection flags, size_t haloId, size_t bufferId, int *buf, cl_mem *bufcl, TauschPackRegion region);
#else
void tausch_packSendBuffer_int(CTauschInt *tC, TauschDeviceDirection flags, size_t haloId, size_t bufferId, int *buf, TauschPackRegion region);
#endif
void tausch_send_int(CTauschInt *tC, TauschDeviceDirection flags, size_t haloId, int mpitag);
void tausch_recv_int(CTauschInt *tC, TauschDeviceDirection flags, size_t haloId);
#ifdef TAUSCH_OPENCL
void tausch_unpackNextRecvBuffer_int(CTauschInt *tC, TauschDeviceDirection flags, size_t haloId, size_t bufferId, int *buf, cl_mem *bufcl, TauschPackRegion region);
void tausch_packAndSend_int(CTauschInt *tC, TauschDeviceDirection flags, size_t haloId, int *buf, cl_mem *bufcl, TauschPackRegion region, int mpitag);
void tausch_recvAndUnpack_int(CTauschInt *tC, TauschDeviceDirection flags, size_t haloId, int *buf, cl_mem *bufcl, TauschPackRegion region);
#else
void tausch_unpackNextRecvBuffer_int(CTauschInt *tC, TauschDeviceDirection flags, size_t haloId, size_t bufferId, int *buf, TauschPackRegion region);
void tausch_packAndSend_int(CTauschInt *tC, TauschDeviceDirection flags, size_t haloId, int *buf, TauschPackRegion region, int mpitag);
void tausch_recvAndUnpack_int(CTauschInt *tC, TauschDeviceDirection flags, size_t haloId, int *buf, TauschPackRegion region);
#endif

#ifdef __cplusplus
}
#endif


#endif // CTAUSCHINT_H
