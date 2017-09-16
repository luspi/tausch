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
 * The object that is created by the C API is called CTauschInt. After its creation it needs to be passed as parameter to any call to the API.
 *
 */
typedef void* CTauschInt;

/*!
 *
 * Create and return a new CTauschInt object using the datatype int.
 *
 * \param numBuffers
 *  The number of buffers that will be used. If more than one, they are all combined into one message. All buffers will have to use the same
 *  discretisation! Typical value: 1.
 * \param valuesPerPointPerBuffer
 *  How many values are stored consecutively per point in the same buffer. Each buffer can have different number of values stored per point. This
 *  is expected to be an array of the same size as the number of buffers. If set to NULL, all buffers are assumed to store 1 value per point.
 * \param comm
 *  The MPI Communictor to be used. %CTauschInt will duplicate the communicator, thus it is safe to have multiple instances of %CTauschInt working
 *  with the same communicator. By default, MPI_COMM_WORLD will be used.
 * \param version
 *  Which version of CTauschInt to create. This depends on the dimensionality of the problem and can be any one of the enum TAUSCH_VERSION: TAUSCH_1D,
 *  TAUSCH_2D, or TAUSCH_3D.
 *
 * \return
 *  Return the CTauschInt object created with the specified configuration.
 *
 */
CTauschInt *tausch_new_int(size_t numBuffers, size_t *valuesPerPointPerBuffer, MPI_Comm comm, TauschVersion version);

/*!
 *
 * Deletes the CTauschInt object, cleaning up all the memory.
 *
 * \param tC
 *  The CTauschInt object to be deleted.
 *
 */
void tausch_delete_int(CTauschInt *tC);

/*!
 *
 * Set the info about all local halos that need to be sent to remote MPI ranks.
 *
 * \param tC
 *  The CTauschInt object to operate on.
 * \param numHaloParts
 *  How many different parts there are to the halo
 * \param haloSpecs
 *  The specification of the different halo parts. This is expected to be a pointer of structs, and each halo region can be referenced later by its
 *  index in this pointer. Each struct must have specified the following 3, 5, or 7 entries:
 *   1. The starting x coordinate of the local region
 *   2. The starting y coordinate of the local region (if present)
 *   3. The starting z coordinate of the local region (if present)
 *   4. The width of the region
 *   5. The height of the region (if present)
 *   6. The depth of the region (if present)
 *   7. The receiving processor
 *
 */
void tausch_setLocalHaloInfo_int(CTauschInt *tC, TauschDeviceDirection flags, size_t numHaloParts, TauschHaloSpec *haloSpecs);

/*!
 *
 * Set the info about all remote halos that are needed by this MPI rank.
 *
 * \param tC
 *  The CTauschInt object to operate on.
 * \param numHaloParts
 *  How many different parts there are to the halo
 * \param haloSpecs
 *  The specification of the different halo parts. This is expected to be a pointer of structs, and each halo region can be referenced later by its
 *  index in this pointer. Each struct must have specified the following 3, 5, or 7 entries:
 *   1. The starting x coordinate of the halo region
 *   2. The starting y coordinate of the halo region (if present)
 *   3. The starting z coordinate of the halo region (if present)
 *   4. The width of the halo region
 *   5. The height of the halo region (if present)
 *   6. The depth of the halo region (if present)
 *   7. The sending processor
 *
 */
void tausch_setRemoteHaloInfo_int(CTauschInt *tC, TauschDeviceDirection flags, size_t numHaloParts, TauschHaloSpec *haloSpecs);

/*!
 *
 * Post the receive for the specified remote halo region of the current MPI rank. This doesn't do anything else but call MPI_Irecv().
 *
 * \param tC
 *  The CTauschUnsignedInt object to operate on.
 * \param haloId
 *  The id of the halo region. This is the index of this halo region in the remote halo specification provided with setRemoteHaloInfo().
 * \param mpitag
 *  The mpitag to be used for this MPI_Irecv().
 *
 */
void tausch_postReceive_int(CTauschInt *tC, TauschDeviceDirection flags, size_t haloId, int mpitag);

/*!
 *
 * Post all MPI receives for the current MPI rank. This doesn't do anything else but call MPI_Irecv() for each remote halo region.
 *
 * \param tC
 *  The CTauschUnsignedInt object to operate on.
 * \param mpitag
 *  An array containing the MPI tags, one for each halo region.
 *
 */
void tausch_postAllReceives_int(CTauschInt *tC, TauschDeviceDirection flags, int *mpitag);

/*!
 *
 * This packs the next buffer for a send. This has to be called for all buffers before sending the message.
 *
 * \param tC
 *  The CTauschInt object to operate on.
 * \param haloId
 *  The id of the halo region. This is the index of this halo region in the local halo specification provided with setLocalHaloInfo().
 * \param bufferId
 *  The id of the buffer. The order of the buffers will be preserved, i.e., packing buffer with id 1 required unpacking that buffer with id 1.
 *  The numbering of the buffers has to start with 0!
 * \param buf
 *  The buffer from which the data is to be extracted according to the local halo specification.
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
void tausch_packSendBuffer_int(CTauschInt *tC, TauschDeviceDirection flags, size_t haloId, size_t bufferId, int *buf, cl_mem *bufcl, TauschPackRegion region);

/*!
 *
 * Sends off the send buffer for the specified halo region. This calls MPI_Start() on the respective MPI_Send_init().
 *
 * \param tC
 *  The CTauschInt object to operate on.
 * \param haloId
 *  The id of the halo region. This is the index of this halo region in the local halo specification provided with setLocalHaloInfo().
 * \param mpitag
 *  The mpitag to be used for this MPI_Isend().
 *
 */
void tausch_send_int(CTauschInt *tC, TauschDeviceDirection flags, size_t haloId, int mpitag);

/*!
 *
 * Makes sure the MPI message for the specified halo is received by this buffer. It does not do anything with that message!
 *
 * \param tC
 *  The CTauschInt object to operate on.
 * \param haloId
 *  The id of the halo region. This is the index of this halo region in the remote halo specification provided with setRemoteHaloInfo().
 *
 */
void tausch_recv_int(CTauschInt *tC, TauschDeviceDirection flags, size_t haloId);

/*!
 *
 * This unpacks the next halo from the received message into provided buffer. This has to be called for all buffers.
 *
 * \param tC
 *  The CTauschInt object to operate on.
 * \param haloId
 *  The id of the halo region. This is the index of this halo region in the remote halo specification provided with setRemoteHaloInfo().
 * \param bufferId
 *  The id of the buffer. The order of the buffers will be preserved, i.e., packing buffer with id 1 required unpacking that buffer with id 1.
 *  The numbering of the buffers has to start with 0!
 * \param buf
 *  The buffer to which the extracted data is to be written to according to the remote halo specification
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
void tausch_unpackNextRecvBuffer_int(CTauschInt *tC, TauschDeviceDirection flags, size_t haloId, size_t bufferId, int *buf, cl_mem *bufcl, TauschPackRegion region);

/*!
 *
 * Shortcut function. If only one buffer is used, this will both pack the data out of the provided buffer and send it off, all with one call.
 *
 * \param tC
 *  The CTauschInt object to operate on.
 * \param haloId
 *  The id of the halo region. This is the index of this halo region in the local halo specification provided with setLocalHaloInfo().
 * \param buf
 *  The buffer from which the data is to be extracted according to the local halo specification.
 * \param mpitag
 *  The mpitag to be used for this MPI_Isend().
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
void tausch_packAndSend_int(CTauschInt *tC, TauschDeviceDirection flags, size_t haloId, int *buf, cl_mem *bufcl, TauschPackRegion region, int mpitag);

/*!
 *
 * Shortcut function. If only one buffer is used, this will both receive the MPI message and unpack the received data into the provided buffer,
 * all with one call.
 *
 * \param tC
 *  The CTauschInt object to operate on.
 * \param haloId
 *  The id of the halo region. This is the index of this halo region in the remote halo specification provided with setRemoteHaloInfo().
 * \param buf
 *  The buffer to which the extracted data is to be written to according to the remote halo specification
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
void tausch_recvAndUnpack_int(CTauschInt *tC, TauschDeviceDirection flags, size_t haloId, int *buf, cl_mem *bufcl, TauschPackRegion region);

#ifdef __cplusplus
}
#endif


#endif // CTAUSCHINT_H
