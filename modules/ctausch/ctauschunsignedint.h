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

#ifndef CTAUSCH2DUNSIGNEDINT_H
#define CTAUSCH2DUNSIGNEDINT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
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
CTauschUnsignedInt *tausch_new_unsignedint(size_t numBuffers, size_t *valuesPerPointPerBuffer, MPI_Comm comm, TauschVersion version);

/*!
 *
 * Deletes the CTauschUnsignedInt object, cleaning up all the memory.
 *
 * \param tC
 *  The CTauschUnsignedInt object to be deleted.
 *
 */
void tausch_delete_unsignedint(CTauschUnsignedInt *tC);

/*!
 *
 * Set the info about all local halos that need to be sent to remote MPI ranks.
 *
 * \param tC
 *  The CTauschUnsignedInt object to operate on.
 * \param numHaloParts
 *  How many different parts there are to the halo
 * \param haloSpecs
 *  The specification of the different halo parts. This is expected to be a nested array of int's, the order of which will be preserved,
 *  and each halo region can be referenced later by its index in this array. Each nested array must contain the following 3, 5, or 7 entries:
 *   1. The starting x coordinate of the local region
 *   2. The starting y coordinate of the local region (if present)
 *   3. The starting z coordinate of the local region (if present)
 *   4. The width of the region
 *   5. The height of the region (if present)
 *   6. The depth of the region (if present)
 *   7. The receiving processor
 *
 */
void tausch_setCpuLocalHaloInfo_unsignedint(CTauschUnsignedInt *tC, size_t numHaloParts, TauschHaloSpec *haloSpecs);

/*!
 *
 * Set the info about all remote halos that are needed by this MPI rank.
 *
 * \param tC
 *  The CTauschUnsignedInt object to operate on.
 * \param numHaloParts
 *  How many different parts there are to the halo
 * \param haloSpecs
 *  The specification of the different halo parts. This is expected to be a nested array of int's, the order of which will be preserved,
 *  and each halo region can be referenced later by its index in this array. Each nested array must contain the following 3, 5, or 7 entries:
 *   1. The starting x coordinate of the halo region
 *   2. The starting y coordinate of the halo region (if present)
 *   3. The starting z coordinate of the halo region (if present)
 *   4. The width of the halo region
 *   5. The height of the halo region (if present)
 *   6. The depth of the halo region (if present)
 *   7. The sending processor
 *
 */
void tausch_setCpuRemoteHaloInfo_unsignedint(CTauschUnsignedInt *tC, size_t numHaloParts, TauschHaloSpec *haloSpecs);

/*!
 *
 * Post the receive for the specified remote halo region of the current MPI rank. This doesn't do anything else but call MPI_Irecv().
 *
 * \param tC
 *  The CTauschUnsignedInt object to operate on.
 * \param id
 *  The id of the halo region. This is the index of this halo region in the remote halo specification provided with setRemoteHaloInfo().
 * \param mpitag
 *  The mpitag to be used for this MPI_Irecv().
 *
 */
void tausch_postReceiveCpu_unsignedint(CTauschUnsignedInt *tC, size_t id, int mpitag);

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
void tausch_postAllReceivesCpu_unsignedint(CTauschUnsignedInt *tC, int *mpitag);

/*!
 *
 * This packs the next buffer for a send. This has to be called as many times as there are buffers before sending the message.
 *
 * \param tC
 *  The CTauschUnsignedInt object to operate on.
 * \param id
 *  The id of the halo region. This is the index of this halo region in the local halo specification provided with setLocalHaloInfo().
 * \param buf
 *  The buffer from which the data is to be extracted according to the local halo specification.
 *
 */
void tausch_packNextSendBuffer_unsignedint(CTauschUnsignedInt *tC, size_t haloId, size_t bufferId, unsigned int *buf);

/*!
 *
 * Sends off the send buffer for the specified halo region. This calls MPI_Start() on the respective MPI_Send_init().
 *
 * \param tC
 *  The CTauschUnsignedInt object to operate on.
 * \param id
 *  The id of the halo region. This is the index of this halo region in the local halo specification provided with setLocalHaloInfo().
 * \param mpitag
 *  The mpitag to be used for this MPI_Isend().
 *
 */
void tausch_send_unsignedint(CTauschUnsignedInt *tC, size_t id, int mpitag);

/*!
 *
 * Makes sure the MPI message for the specified halo is received by this buffer. It does not do anything with that message!
 *
 * \param tC
 *  The CTauschUnsignedInt object to operate on.
 * \param id
 *  The id of the halo region. This is the index of this halo region in the remote halo specification provided with setRemoteHaloInfo().
 *
 */
void tausch_recv_unsignedint(CTauschUnsignedInt *tC, size_t id);

/*!
 *
 * This unpacks the next halo from the received message into provided buffer. This has to be called as many times as there are buffers.
 *
 * \param tC
 *  The CTauschUnsignedInt object to operate on.
 * \param id
 *  The id of the halo region. This is the index of this halo region in the remote halo specification provided with setRemoteHaloInfo().
 * \param buf
 *  The buffer to which the extracted data is to be written to according to the remote halo specification
 *
 */
void tausch_unpackNextRecvBuffer_unsignedint(CTauschUnsignedInt *tC, size_t haloId, size_t bufferId, unsigned int *buf);

/*!
 *
 * Shortcut function. If only one buffer is used, this will both pack the data out of the provided buffer and send it off, all with one call.
 *
 * \param tC
 *  The CTauschUnsignedInt object to operate on.
 * \param id
 *  The id of the halo region. This is the index of this halo region in the local halo specification provided with setLocalHaloInfo().
 * \param buf
 *  The buffer from which the data is to be extracted according to the local halo specification.
 * \param mpitag
 *  The mpitag to be used for this MPI_Isend().
 *
 */
void tausch_packAndSend_unsignedint(CTauschUnsignedInt *tC, size_t haloId, size_t bufferId, int mpitag, unsigned int *buf);

/*!
 *
 * Shortcut function. If only one buffer is used, this will both receive the MPI message and unpack the received data into the provided buffer,
 * all with one call.
 *
 * \param tC
 *  The CTauschUnsignedInt object to operate on.
 * \param id
 *  The id of the halo region. This is the index of this halo region in the remote halo specification provided with setRemoteHaloInfo().
 * \param buf
 *  The buffer to which the extracted data is to be written to according to the remote halo specification
 *
 */
void tausch_recvAndUnpack_unsignedint(CTauschUnsignedInt *tC, size_t haloId, size_t bufferId, unsigned int *buf);

#ifdef __cplusplus
}
#endif


#endif // CTAUSCH2DUNSIGNEDINT_H
