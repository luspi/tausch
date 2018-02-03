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
 *
 *  All C API versions for different data types are equivalent to this one, with different suffix, and are thus not documented.
 */

#ifndef CTAUSCHDOUBLE_H
#define CTAUSCHDOUBLE_H

#ifdef __cplusplus
extern "C" {
#endif

#include "../tausch.h"

#ifdef TAUSCH_OPENCL
#include <CL/cl.h>
#endif

/*!
 *
 * The object that is created by the C API is called CTauschUnsignedInt. After its creation it needs to be passed as parameter to any call to the API.
 *
 */
typedef void* CTauschUnsignedInt;

/*!
 *
 * Create and return a new CTauschUnsignedInt object using the datatype unsigned int.
 *
 * \param numBuffers
 *  The number of buffers that will be used. If more than one, they are all combined into one message. All buffers will have to use the same
 *  discretisation! Typical value: 1.
 * \param valuesPerPointPerBuffer
 *  How many values are stored consecutively per point in the same buffer. Each buffer can have different number of values stored per point. This
 *  is expected to be an array of the same size as the number of buffers. If set to NULL, all buffers are assumed to store 1 value per point.
 * \param comm
 *  The MPI Communictor to be used. %CTauschUnsignedInt will duplicate the communicator, thus it is safe to have multiple instances of %CTauschUnsignedInt working
 *  with the same communicator. By default, MPI_COMM_WORLD will be used.
 * \param version
 *  Which version of CTauschUnsignedInt to create. This depends on the dimensionality of the problem and can be any one of the enum TAUSCH_VERSION: TAUSCH_1D,
 *  TAUSCH_2D, or TAUSCH_3D.
 *
 * \return
 *  Return the CTauschUnsignedInt object created with the specified configuration.
 *
 */
CTauschUnsignedInt *tausch_new_unsignedint(size_t numBuffers, size_t *valuesPerPointPerBuffer, MPI_Comm comm);
void tausch_delete_unsignedint(CTauschUnsignedInt *tC);

/****************************************/
// setLocalHaloInfo*

void tausch_setLocalHaloInfo1D_CwC_unsignedint(CTauschUnsignedInt *tC, size_t numHaloParts, TauschHaloSpec *haloSpecs);
void tausch_setLocalHaloInfo2D_CwC_unsignedint(CTauschUnsignedInt *tC, size_t numHaloParts, TauschHaloSpec *haloSpecs);
void tausch_setLocalHaloInfo3D_CwC_unsignedint(CTauschUnsignedInt *tC, size_t numHaloParts, TauschHaloSpec *haloSpecs);

#ifdef TAUSCH_OPENCL

void tausch_setLocalHaloInfo1D_CwG_unsignedint(CTauschUnsignedInt *tC, size_t numHaloParts, TauschHaloSpec *haloSpecs);
void tausch_setLocalHaloInfo2D_CwG_unsignedint(CTauschUnsignedInt *tC, size_t numHaloParts, TauschHaloSpec *haloSpecs);
void tausch_setLocalHaloInfo3D_CwG_unsignedint(CTauschUnsignedInt *tC, size_t numHaloParts, TauschHaloSpec *haloSpecs);

void tausch_setLocalHaloInfo1D_GwC_unsignedint(CTauschUnsignedInt *tC, size_t numHaloParts, TauschHaloSpec *haloSpecs);
void tausch_setLocalHaloInfo2D_GwC_unsignedint(CTauschUnsignedInt *tC, size_t numHaloParts, TauschHaloSpec *haloSpecs);
void tausch_setLocalHaloInfo3D_GwC_unsignedint(CTauschUnsignedInt *tC, size_t numHaloParts, TauschHaloSpec *haloSpecs);

#endif


/****************************************/
// setRemoteHaloInfo*

void tausch_setRemoteHaloInfo1D_CwC_unsignedint(CTauschUnsignedInt *tC, size_t numHaloParts, TauschHaloSpec *haloSpecs);
void tausch_setRemoteHaloInfo2D_CwC_unsignedint(CTauschUnsignedInt *tC, size_t numHaloParts, TauschHaloSpec *haloSpecs);
void tausch_setRemoteHaloInfo3D_CwC_unsignedint(CTauschUnsignedInt *tC, size_t numHaloParts, TauschHaloSpec *haloSpecs);

#ifdef TAUSCH_OPENCL

void tausch_setRemoteHaloInfo1D_CwG_unsignedint(CTauschUnsignedInt *tC, size_t numHaloParts, TauschHaloSpec *haloSpecs);
void tausch_setRemoteHaloInfo2D_CwG_unsignedint(CTauschUnsignedInt *tC, size_t numHaloParts, TauschHaloSpec *haloSpecs);
void tausch_setRemoteHaloInfo3D_CwG_unsignedint(CTauschUnsignedInt *tC, size_t numHaloParts, TauschHaloSpec *haloSpecs);

void tausch_setRemoteHaloInfo1D_GwC_unsignedint(CTauschUnsignedInt *tC, size_t numHaloParts, TauschHaloSpec *haloSpecs);
void tausch_setRemoteHaloInfo2D_GwC_unsignedint(CTauschUnsignedInt *tC, size_t numHaloParts, TauschHaloSpec *haloSpecs);
void tausch_setRemoteHaloInfo3D_GwC_unsignedint(CTauschUnsignedInt *tC, size_t numHaloParts, TauschHaloSpec *haloSpecs);

#endif


/****************************************/
// postReceive*

void tausch_postReceive1D_CwC_unsignedint(CTauschUnsignedInt *tC, size_t haloId, int msgtag);
void tausch_postReceive2D_CwC_unsignedint(CTauschUnsignedInt *tC, size_t haloId, int msgtag);
void tausch_postReceive3D_CwC_unsignedint(CTauschUnsignedInt *tC, size_t haloId, int msgtag);

#ifdef TAUSCH_OPENCL

void tausch_postReceive1D_CwG_unsignedint(CTauschUnsignedInt *tC, size_t haloId, int msgtag);
void tausch_postReceive2D_CwG_unsignedint(CTauschUnsignedInt *tC, size_t haloId, int msgtag);
void tausch_postReceive3D_CwG_unsignedint(CTauschUnsignedInt *tC, size_t haloId, int msgtag);

void tausch_postReceive1D_GwC_unsignedint(CTauschUnsignedInt *tC, size_t haloId, int msgtag);
void tausch_postReceive2D_GwC_unsignedint(CTauschUnsignedInt *tC, size_t haloId, int msgtag);
void tausch_postReceive3D_GwC_unsignedint(CTauschUnsignedInt *tC, size_t haloId, int msgtag);

#endif


/****************************************/
// postAllReceives*

void tausch_postAllReceives1D_CwC_unsignedint(CTauschUnsignedInt *tC, int *msgtag);
void tausch_postAllReceives2D_CwC_unsignedint(CTauschUnsignedInt *tC, int *msgtag);
void tausch_postAllReceives3D_CwC_unsignedint(CTauschUnsignedInt *tC, int *msgtag);

#ifdef TAUSCH_OPENCL

void tausch_postAllReceives1D_CwG_unsignedint(CTauschUnsignedInt *tC, int *msgtag);
void tausch_postAllReceives2D_CwG_unsignedint(CTauschUnsignedInt *tC, int *msgtag);
void tausch_postAllReceives3D_CwG_unsignedint(CTauschUnsignedInt *tC, int *msgtag);

void tausch_postAllReceives1D_GwC_unsignedint(CTauschUnsignedInt *tC, int *msgtag);
void tausch_postAllReceives2D_GwC_unsignedint(CTauschUnsignedInt *tC, int *msgtag);
void tausch_postAllReceives3D_GwC_unsignedint(CTauschUnsignedInt *tC, int *msgtag);

#endif


/****************************************/
// postAllReceives*

void tausch_packSendBuffer1D_CwC_unsignedint(CTauschUnsignedInt *tC, size_t haloId, size_t bufferId, unsigned int *buf, TauschPackRegion region);
void tausch_packSendBuffer2D_CwC_unsignedint(CTauschUnsignedInt *tC, size_t haloId, size_t bufferId, unsigned int *buf, TauschPackRegion region);
void tausch_packSendBuffer3D_CwC_unsignedint(CTauschUnsignedInt *tC, size_t haloId, size_t bufferId, unsigned int *buf, TauschPackRegion region);

#ifdef TAUSCH_OPENCL

void tausch_packSendBuffer1D_CwG_unsignedint(CTauschUnsignedInt *tC, size_t haloId, size_t bufferId, unsigned int *buf, TauschPackRegion region);
void tausch_packSendBuffer2D_CwG_unsignedint(CTauschUnsignedInt *tC, size_t haloId, size_t bufferId, unsigned int *buf, TauschPackRegion region);
void tausch_packSendBuffer3D_CwG_unsignedint(CTauschUnsignedInt *tC, size_t haloId, size_t bufferId, unsigned int *buf, TauschPackRegion region);

void tausch_packSendBuffer1D_GwC_unsignedint(CTauschUnsignedInt *tC, size_t haloId, size_t bufferId, cl_mem *bufcl);
void tausch_packSendBuffer2D_GwC_unsignedint(CTauschUnsignedInt *tC, size_t haloId, size_t bufferId, cl_mem *bufcl);
void tausch_packSendBuffer3D_GwC_unsignedint(CTauschUnsignedInt *tC, size_t haloId, size_t bufferId, cl_mem *bufcl);

#endif


/****************************************/
// postAllReceives*

void tausch_send1D_CwC_unsignedint(CTauschUnsignedInt *tC, size_t haloId, int msgtag);
void tausch_send2D_CwC_unsignedint(CTauschUnsignedInt *tC, size_t haloId, int msgtag);
void tausch_send3D_CwC_unsignedint(CTauschUnsignedInt *tC, size_t haloId, int msgtag);

#ifdef TAUSCH_OPENCL

void tausch_send1D_CwG_unsignedint(CTauschUnsignedInt *tC, size_t haloId, int msgtag);
void tausch_send2D_CwG_unsignedint(CTauschUnsignedInt *tC, size_t haloId, int msgtag);
void tausch_send3D_CwG_unsignedint(CTauschUnsignedInt *tC, size_t haloId, int msgtag);

void tausch_send1D_GwC_unsignedint(CTauschUnsignedInt *tC, size_t haloId, int msgtag);
void tausch_send2D_GwC_unsignedint(CTauschUnsignedInt *tC, size_t haloId, int msgtag);
void tausch_send3D_GwC_unsignedint(CTauschUnsignedInt *tC, size_t haloId, int msgtag);

#endif


/****************************************/
// recv*

void tausch_recv1D_CwC_unsignedint(CTauschUnsignedInt *tC, size_t haloId);
void tausch_recv2D_CwC_unsignedint(CTauschUnsignedInt *tC, size_t haloId);
void tausch_recv3D_CwC_unsignedint(CTauschUnsignedInt *tC, size_t haloId);

#ifdef TAUSCH_OPENCL

void tausch_recv1D_CwG_unsignedint(CTauschUnsignedInt *tC, size_t haloId);
void tausch_recv2D_CwG_unsignedint(CTauschUnsignedInt *tC, size_t haloId);
void tausch_recv3D_CwG_unsignedint(CTauschUnsignedInt *tC, size_t haloId);

void tausch_recv1D_GwC_unsignedint(CTauschUnsignedInt *tC, size_t haloId);
void tausch_recv2D_GwC_unsignedint(CTauschUnsignedInt *tC, size_t haloId);
void tausch_recv3D_GwC_unsignedint(CTauschUnsignedInt *tC, size_t haloId);

#endif


/****************************************/
// unpackNextRecvBuffer*

void tausch_unpackNextRecvBuffer1D_CwC_unsignedint(CTauschUnsignedInt *tC, size_t haloId, size_t bufferId, unsigned int *buf, TauschPackRegion region);
void tausch_unpackNextRecvBuffer2D_CwC_unsignedint(CTauschUnsignedInt *tC, size_t haloId, size_t bufferId, unsigned int *buf, TauschPackRegion region);
void tausch_unpackNextRecvBuffer3D_CwC_unsignedint(CTauschUnsignedInt *tC, size_t haloId, size_t bufferId, unsigned int *buf, TauschPackRegion region);

#ifdef TAUSCH_OPENCL

void tausch_unpackNextRecvBuffer1D_CwG_unsignedint(CTauschUnsignedInt *tC, size_t haloId, size_t bufferId, unsigned int *buf, TauschPackRegion region);
void tausch_unpackNextRecvBuffer2D_CwG_unsignedint(CTauschUnsignedInt *tC, size_t haloId, size_t bufferId, unsigned int *buf, TauschPackRegion region);
void tausch_unpackNextRecvBuffer3D_CwG_unsignedint(CTauschUnsignedInt *tC, size_t haloId, size_t bufferId, unsigned int *buf, TauschPackRegion region);

void tausch_unpackNextRecvBuffer1D_GwC_unsignedint(CTauschUnsignedInt *tC, size_t haloId, size_t bufferId, cl_mem *bufcl);
void tausch_unpackNextRecvBuffer2D_GwC_unsignedint(CTauschUnsignedInt *tC, size_t haloId, size_t bufferId, cl_mem *bufcl);
void tausch_unpackNextRecvBuffer3D_GwC_unsignedint(CTauschUnsignedInt *tC, size_t haloId, size_t bufferId, cl_mem *bufcl);

#endif


/****************************************/
// packAndSend*

void tausch_packAndSend1D_CwC_unsignedint(CTauschUnsignedInt *tC, size_t haloId, unsigned int *buf, TauschPackRegion region, int msgtag);
void tausch_packAndSend2D_CwC_unsignedint(CTauschUnsignedInt *tC, size_t haloId, unsigned int *buf, TauschPackRegion region, int msgtag);
void tausch_packAndSend3D_CwC_unsignedint(CTauschUnsignedInt *tC, size_t haloId, unsigned int *buf, TauschPackRegion region, int msgtag);

#ifdef TAUSCH_OPENCL

void tausch_packAndSend1D_CwG_unsignedint(CTauschUnsignedInt *tC, size_t haloId, unsigned int *buf, TauschPackRegion region, int msgtag);
void tausch_packAndSend2D_CwG_unsignedint(CTauschUnsignedInt *tC, size_t haloId, unsigned int *buf, TauschPackRegion region, int msgtag);
void tausch_packAndSend3D_CwG_unsignedint(CTauschUnsignedInt *tC, size_t haloId, unsigned int *buf, TauschPackRegion region, int msgtag);

void tausch_packAndSend1D_GwC_unsignedint(CTauschUnsignedInt *tC, size_t haloId, cl_mem *bufcl, int msgtag);
void tausch_packAndSend2D_GwC_unsignedint(CTauschUnsignedInt *tC, size_t haloId, cl_mem *bufcl, int msgtag);
void tausch_packAndSend3D_GwC_unsignedint(CTauschUnsignedInt *tC, size_t haloId, cl_mem *bufcl, int msgtag);

#endif


/****************************************/
// packAndSend*

void tausch_recvAndUnpack1D_CwC_unsignedint(CTauschUnsignedInt *tC, size_t haloId, unsigned int *buf, TauschPackRegion region);
void tausch_recvAndUnpack2D_CwC_unsignedint(CTauschUnsignedInt *tC, size_t haloId, unsigned int *buf, TauschPackRegion region);
void tausch_recvAndUnpack3D_CwC_unsignedint(CTauschUnsignedInt *tC, size_t haloId, unsigned int *buf, TauschPackRegion region);

#ifdef TAUSCH_OPENCL

void tausch_recvAndUnpack1D_CwG_unsignedint(CTauschUnsignedInt *tC, size_t haloId, unsigned int *buf, TauschPackRegion region);
void tausch_recvAndUnpack2D_CwG_unsignedint(CTauschUnsignedInt *tC, size_t haloId, unsigned int *buf, TauschPackRegion region);
void tausch_recvAndUnpack3D_CwG_unsignedint(CTauschUnsignedInt *tC, size_t haloId, unsigned int *buf, TauschPackRegion region);

void tausch_recvAndUnpack1D_GwC_unsignedint(CTauschUnsignedInt *tC, size_t haloId, cl_mem *bufcl);
void tausch_recvAndUnpack2D_GwC_unsignedint(CTauschUnsignedInt *tC, size_t haloId, cl_mem *bufcl);
void tausch_recvAndUnpack3D_GwC_unsignedint(CTauschUnsignedInt *tC, size_t haloId, cl_mem *bufcl);

#endif

#ifdef __cplusplus
}
#endif


#endif // CTAUSCHDOUBLE_H
