/*!
 * \file
 * \author  Lukas Spies <LSpies@illinois.edu>
 * \version 1.0
 *
 * \brief
 *  C wrapper to C++ API, double datatype.
 *
 *  C wrapper to C++ API, double datatype. It provides a single interface for all three versions (1D, 2D, 3D). It is possible to choose at runtime
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

#ifndef TAUSCHVERSIONDEF
#define TAUSCHVERSIONDEF
/*!
 * An enum to choose at runtime which version of Tausch to use: 1D, 2D or 3D. This enum is only used for the C API!
 */
enum TauschVersion {
    TAUSCH_1D,
    TAUSCH_2D,
    TAUSCH_3D
};
#endif // TAUSCHVERSIONDEF

/*!
 *
 * The object that is created by the C API is called CTauschDouble. After its creation it needs to be passed as parameter to any call to the API.
 *
 */
typedef void* CTauschDouble;

/*!
 *
 * Create and return a new CTauschDouble object using the datatype double.
 *
 * \param numBuffers
 *  The number of buffers that will be used. If more than one, they are all combined into one message. All buffers will have to use the same
 *  discretisation! Typical value: 1.
 * \param valuesPerPointPerBuffer
 *  How many values are stored consecutively per point in the same buffer. Each buffer can have different number of values stored per point. This
 *  is expected to be an array of the same size as the number of buffers. If set to NULL, all buffers are assumed to store 1 value per point.
 * \param comm
 *  The MPI Communictor to be used. %CTauschDouble will duplicate the communicator, thus it is safe to have multiple instances of %CTauschDouble working
 *  with the same communicator. By default, MPI_COMM_WORLD will be used.
 * \param version
 *  Which version of CTauschDouble to create. This depends on the dimensionality of the problem and can be any one of the enum TAUSCH_VERSION: TAUSCH_1D,
 *  TAUSCH_2D, or TAUSCH_3D.
 *
 * \return
 *  Return the CTauschDouble object created with the specified configuration.
 *
 */
CTauschDouble *tausch_new_double(size_t numBuffers, size_t *valuesPerPointPerBuffer, MPI_Comm comm, TauschVersion version);

/*!
 *
 * Deletes the CTauschDouble object, cleaning up all the memory.
 *
 * \param tC
 *  The CTauschDouble object to be deleted.
 *
 */
void tausch_delete_double(CTauschDouble *tC);

/*!
 *
 * Set the info about all local halos that need to be sent off.
 *
 * \param tC
 *  The CTauschDouble object to operate on.
 * \param flags
 *  This is expected to be a bit wise combination of two flags. First choose one of TAUSCH_CPU or TAUSCH_GPU and combine it with either of
 *  TAUSCH_WITHCPU or TAUSCH_WITHGPU.
 * \param numHaloParts
 *  How many different parts there are to the halo
 * \param haloSpecs
 *  The specification of the different halo parts. This is expected to be a pointer of structs, and each halo region can be referenced later by its
 *  index in this pointer. Each struct must have specified some of the following variables:
 *  variable | description
 *  :------: | -----------
 *   haloX | The starting x coordinate of the halo region
 *   haloY | The starting y coordinate of the halo region (if present)
 *   haloZ | The starting z coordinate of the halo region (if present)
 *   haloWidth | The width of the halo region
 *   haloHeight | The height of the halo region (if present)
 *   haloDepth | The height of the halo region (if present)
 *   bufferWidth | The width of the underlying buffer
 *   bufferHeight | The height of the underlying buffer (if present)
 *   bufferDepth | The depth of the underlying buffer (if present)
 *   remoteMpiRank | The receiving processor
 *
 */
void tausch_setLocalHaloInfo_double(CTauschDouble *tC, TauschDeviceDirection flags, size_t numHaloParts, TauschHaloSpec *haloSpecs);

/*!
 *
 * Set the info about all remote halos that are needed by this partition.
 *
 * \param tC
 *  The CTauschDouble object to operate on.
 * \param flags
 *  This is expected to be a bit wise combination of two flags. First choose one of TAUSCH_CPU or TAUSCH_GPU and combine it with either of
 *  TAUSCH_WITHCPU or TAUSCH_WITHGPU.
 * \param numHaloParts
 *  How many different parts there are to the halo
 * \param haloSpecs
 *  The specification of the different halo parts. This is expected to be a pointer of structs, and each halo region can be referenced later by its
 *  index in this pointer. Each struct must have specified some of the following variables:
 *   variable | description
 *  :------: | -----------
 *   haloX | The starting x coordinate of the halo region
 *   haloY | The starting y coordinate of the halo region (if present)
 *   haloZ | The starting z coordinate of the halo region (if present)
 *   haloWidth | The width of the halo region
 *   haloHeight | The height of the halo region (if present)
 *   haloDepth | The height of the halo region (if present)
 *   bufferWidth | The width of the underlying buffer
 *   bufferHeight | The height of the underlying buffer (if present)
 *   bufferDepth | The depth of the underlying buffer (if present)
 *   remoteMpiRank | The sending processor
 *
 */
void tausch_setRemoteHaloInfo_double(CTauschDouble *tC, TauschDeviceDirection flags, size_t numHaloParts, TauschHaloSpec *haloSpecs);

/*!
 *
 * Post the receive for the specified remote halo region of the current rank.
 *
 * \param tC
 *  The CTauschUnsignedInt object to operate on.
 * \param flags
 *  This is expected to be a bit wise combination of two flags. First choose one of TAUSCH_CPU or TAUSCH_GPU and combine it with either of
 *  TAUSCH_WITHCPU or TAUSCH_WITHGPU.
 * \param haloId
 *  The id of the halo region. This is the index of this halo region in the remote halo specification provided with setRemoteHaloInfo().
 * \param msgtag
 *  The message tag to be used for this receive.
 *
 */
void tausch_postReceive_double(CTauschDouble *tC, TauschDeviceDirection flags, size_t haloId, int msgtag);

/*!
 *
 * Post all receives for the current rank for each remote halo region.
 *
 * \param tC
 *  The CTauschUnsignedInt object to operate on.
 * \param flags
 *  This is expected to be a bit wise combination of two flags. First choose one of TAUSCH_CPU or TAUSCH_GPU and combine it with either of
 *  TAUSCH_WITHCPU or TAUSCH_WITHGPU.
 * \param msgtag
 *  An array containing the message tags, one for each halo region.
 *
 */
void tausch_postAllReceives_double(CTauschDouble *tC, TauschDeviceDirection flags, int *msgtag);

/*!
 *
 * This packs the next buffer for a send. This has to be called for all buffers before sending the message.
 *
 * \param tC
 *  The CTauschDouble object to operate on.
 * \param flags
 *  This is expected to be a bit wise combination of two flags. First choose one of TAUSCH_CPU or TAUSCH_GPU and combine it with either of
 *  TAUSCH_WITHCPU or TAUSCH_WITHGPU.
 * \param haloId
 *  The id of the halo region. This is the index of this halo region in the local halo specification provided with setLocalHaloInfo().
 * \param bufferId
 *  The id of the buffer. The order of the buffers will be preserved, i.e., packing buffer with id 1 required unpacking that buffer with id 1.
 *  The numbering of the buffers has to start with 0!
 * \param buf
 *  The buffer from which the data is to be extracted according to the local halo specification. Set to NULL if OpenCL buffer is used!
 * \param bufcl
 *  The buffer from which the data is to be extracted according to the local halo specification. Set to NULL if "normal" C buffer is used!
 * \param region
 *  The specification of the region of the halo parts to be packed. This is expected to be a pointer of structs, and each halo region can be
 *  referenced later by its index in this pointer. Each struct must have specified the following 3, 5, or 7 entries:
 *  variable | description
 *  :-------: | -------
 *   startX | The starting x coordinate of the region to be packed
 *   startY | The starting y coordinate of the region to be packed (if present)
 *   startZ | The starting z coordinate of the region to be packed (if present)
 *   width | The width of the region to be packed
 *   height | The height of the region to be packed (if present)
 *   depth | The depth of the region to be packed (if present)
 *
 */
#ifdef TAUSCH_OPENCL
void tausch_packSendBuffer_double(CTauschDouble *tC, TauschDeviceDirection flags, size_t haloId, size_t bufferId, double *buf, cl_mem *bufcl, TauschPackRegion region);
#else
void tausch_packSendBuffer_double(CTauschDouble *tC, TauschDeviceDirection flags, size_t haloId, size_t bufferId, double *buf, TauschPackRegion region);
#endif

/*!
 *
 * Sends off the send buffer for the specified halo region.
 *
 * \param tC
 *  The CTauschDouble object to operate on.
 * \param flags
 *  This is expected to be a bit wise combination of two flags. First choose one of TAUSCH_CPU or TAUSCH_GPU and combine it with either of
 *  TAUSCH_WITHCPU or TAUSCH_WITHGPU.
 * \param haloId
 *  The id of the halo region. This is the index of this halo region in the local halo specification provided with setLocalHaloInfo().
 * \param msgtag
 *  The message tag to be used for this send.
 *
 */
void tausch_send_double(CTauschDouble *tC, TauschDeviceDirection flags, size_t haloId, int msgtag);

/*!
 *
 * Makes sure the message for the specified halo is received by this buffer. It does not do anything with that message!
 *
 * \param tC
 *  The CTauschDouble object to operate on.
 * \param flags
 *  This is expected to be a bit wise combination of two flags. First choose one of TAUSCH_CPU or TAUSCH_GPU and combine it with either of
 *  TAUSCH_WITHCPU or TAUSCH_WITHGPU.
 * \param haloId
 *  The id of the halo region. This is the index of this halo region in the remote halo specification provided with setRemoteHaloInfo().
 *
 */
void tausch_recv_double(CTauschDouble *tC, TauschDeviceDirection flags, size_t haloId);

/*!
 *
 * This unpacks the next halo from the received message into provided buffer. This has to be called for all buffers.
 *
 * \param tC
 *  The CTauschDouble object to operate on.
 * \param flags
 *  This is expected to be a bit wise combination of two flags. First choose one of TAUSCH_CPU or TAUSCH_GPU and combine it with either of
 *  TAUSCH_WITHCPU or TAUSCH_WITHGPU.
 * \param haloId
 *  The id of the halo region. This is the index of this halo region in the remote halo specification provided with setRemoteHaloInfo().
 * \param bufferId
 *  The id of the buffer. The order of the buffers will be preserved, i.e., packing buffer with id 1 required unpacking that buffer with id 1.
 *  The numbering of the buffers has to start with 0!
 * \param buf
 *  The buffer to which the extracted data is to be written to according to the remote halo specification. Set to NULL if OpenCL buffer is used!
 * \param bufcl
 *  The OpenCL buffer to which the extracted data is to be written to according to the remote halo specification. Set to NULL if "normal" C buffer is used!
 * \param region
 *  The specification of the region of the halo parts to be packed. This is expected to be a pointer of structs, and each halo region can be
 *  referenced later by its index in this pointer. Each struct must have specified the following 3, 5, or 7 entries:
 *  variable | description
 *  :-------: | -------
 *   startX | The starting x coordinate of the region to be packed
 *   startY | The starting y coordinate of the region to be packed (if present)
 *   startZ | The starting z coordinate of the region to be packed (if present)
 *   width | The width of the region to be packed
 *   height | The height of the region to be packed (if present)
 *   depth | The depth of the region to be packed (if present)
 *
 */
#ifdef TAUSCH_OPENCL
void tausch_unpackNextRecvBuffer_double(CTauschDouble *tC, TauschDeviceDirection flags, size_t haloId, size_t bufferId, double *buf, cl_mem *bufcl, TauschPackRegion region);
#else
void tausch_unpackNextRecvBuffer_double(CTauschDouble *tC, TauschDeviceDirection flags, size_t haloId, size_t bufferId, double *buf, TauschPackRegion region);
#endif

/*!
 *
 * Shortcut function. If only one buffer is used, this will both pack the data out of the provided buffer and send it off, all with one call.
 *
 * \param tC
 *  The CTauschDouble object to operate on.
 * \param flags
 *  This is expected to be a bit wise combination of two flags. First choose one of TAUSCH_CPU or TAUSCH_GPU and combine it with either of
 *  TAUSCH_WITHCPU or TAUSCH_WITHGPU.
 * \param haloId
 *  The id of the halo region. This is the index of this halo region in the local halo specification provided with setLocalHaloInfo().
 * \param buf
 *  The buffer from which the data is to be extracted according to the local halo specification. Set to NULL if OpenCL buffer is used!
 * \param bufcl
 *  The OpenCL buffer from which the data is to be extracted according to the local halo specification. Set to NULL if "normal" C buffer is used!
 * \param msgtag
 *  The message tag to be used for this send.
 * \param region
 *  The specification of the region of the halo parts to be packed. This is expected to be a pointer of structs, and each halo region can be
 *  referenced later by its index in this pointer. Each struct must have specified the following 3, 5, or 7 entries:
 *  variable | description
 *  :-------: | -------
 *   startX | The starting x coordinate of the region to be packed
 *   startY | The starting y coordinate of the region to be packed (if present)
 *   startZ | The starting z coordinate of the region to be packed (if present)
 *   width | The width of the region to be packed
 *   height | The height of the region to be packed (if present)
 *   depth | The depth of the region to be packed (if present)
 *
 */
#ifdef TAUSCH_OPENCL
void tausch_packAndSend_double(CTauschDouble *tC, TauschDeviceDirection flags, size_t haloId, double *buf, cl_mem *bufcl, TauschPackRegion region, int msgtag);
#else
void tausch_packAndSend_double(CTauschDouble *tC, TauschDeviceDirection flags, size_t haloId, double *buf, TauschPackRegion region, int msgtag);
#endif

/*!
 *
 * Shortcut function. If only one buffer is used, this will both receive the message and unpack the received data into the provided buffer,
 * all with one call.
 *
 * \param tC
 *  The CTauschDouble object to operate on.
 * \param flags
 *  This is expected to be a bit wise combination of two flags. First choose one of TAUSCH_CPU or TAUSCH_GPU and combine it with either of
 *  TAUSCH_WITHCPU or TAUSCH_WITHGPU.
 * \param haloId
 *  The id of the halo region. This is the index of this halo region in the remote halo specification provided with setRemoteHaloInfo().
 * \param buf
 *  The buffer to which the extracted data is to be written to according to the remote halo specification. Set to NULL if OpenCL buffer is used!
 * \param bufcl
 *  The OpenCL buffer to which the extracted data is to be written to according to the remote halo specification. Set to NULL if "normal" C buffer is used!
 * \param region
 *  The specification of the region of the halo parts to be packed. This is expected to be a pointer of structs, and each halo region can be
 *  referenced later by its index in this pointer. Each struct must have specified the following 3, 5, or 7 entries:
 *  variable | description
 *  :-------: | -------
 *   startX | The starting x coordinate of the region to be packed
 *   startY | The starting y coordinate of the region to be packed (if present)
 *   startZ | The starting z coordinate of the region to be packed (if present)
 *   width | The width of the region to be packed
 *   height | The height of the region to be packed (if present)
 *   depth | The depth of the region to be packed (if present)
 *
 */
#ifdef TAUSCH_OPENCL
void tausch_recvAndUnpack_double(CTauschDouble *tC, TauschDeviceDirection flags, size_t haloId, double *buf, cl_mem *bufcl, TauschPackRegion region);
#else
void tausch_recvAndUnpack_double(CTauschDouble *tC, TauschDeviceDirection flags, size_t haloId, double *buf, TauschPackRegion region);
#endif

#ifdef __cplusplus
}
#endif


#endif // CTAUSCHDOUBLE_H
