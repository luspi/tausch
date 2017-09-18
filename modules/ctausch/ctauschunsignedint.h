/*!
 * \file
 * \author  Lukas Spies <LSpies@illinois.edu>
 * \version 1.0
 *
 * \brief
 *  C wrapper to C++ API, unsigned int datatype.
 *
 *  C wrapper to C++ API, unsigned int datatype. It provides a single interface for all three versions (1D, 2D, 3D). It is possible to choose at runtime
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

typedef void* CTauschUnsignedInt;

CTauschUnsignedInt *tausch_new_unsignedint(size_t numBuffers, size_t *valuesPerPointPerBuffer, MPI_Comm comm, TauschVersion version);

void tausch_delete_unsignedint(CTauschUnsignedInt *tC);
void tausch_setLocalHaloInfo_unsignedint(CTauschUnsignedInt *tC, TauschDeviceDirection flags, size_t numHaloParts, TauschHaloSpec *haloSpecs);
void tausch_setRemoteHaloInfo_unsignedint(CTauschUnsignedInt *tC, TauschDeviceDirection flags, size_t numHaloParts, TauschHaloSpec *haloSpecs);
void tausch_postReceive_unsignedint(CTauschUnsignedInt *tC, TauschDeviceDirection flags, size_t haloId, int mpitag);
void tausch_postAllReceives_unsignedint(CTauschUnsignedInt *tC, TauschDeviceDirection flags, int *mpitag);
void tausch_packSendBuffer_unsignedint(CTauschUnsignedInt *tC, TauschDeviceDirection flags, size_t haloId, size_t bufferId, unsigned int *buf, cl_mem *bufcl, TauschPackRegion region);
void tausch_send_unsignedint(CTauschUnsignedInt *tC, TauschDeviceDirection flags, size_t haloId, int mpitag);
void tausch_recv_unsignedint(CTauschUnsignedInt *tC, TauschDeviceDirection flags, size_t haloId);
void tausch_unpackNextRecvBuffer_unsignedint(CTauschUnsignedInt *tC, TauschDeviceDirection flags, size_t haloId, size_t bufferId, unsigned int *buf, cl_mem *bufcl, TauschPackRegion region);
void tausch_packAndSend_unsignedint(CTauschUnsignedInt *tC, TauschDeviceDirection flags, size_t haloId, unsigned int *buf, cl_mem *bufcl, TauschPackRegion region, int mpitag);
void tausch_recvAndUnpack_unsignedint(CTauschUnsignedInt *tC, TauschDeviceDirection flags, size_t haloId, unsigned int *buf, cl_mem *bufcl, TauschPackRegion region);

#ifdef __cplusplus
}
#endif


#endif // CTAUSCHFLOAT_H
