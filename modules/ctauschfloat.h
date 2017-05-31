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

#ifndef CTAUSCH2DFLOAT_H
#define CTAUSCH2DFLOAT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#ifdef TAUSCH_OPENCL
#include <CL/cl.h>
#endif

/*!
 *
 * The object that is created by the C API is called CTauschFloat. After its creation it needs to be passed as parameter to any call to the API.
 *
 */
typedef void* CTauschFloat;

/*!
 *
 * Create and return a new CTauschFloat object using the datatype float.
 *
 * \param localDim
 *  Array of size 1 to 3 holding the dimension(s) of the local partition (not the global dimensions), with the x dimension being the first value, the
 *  y dimension (if present) being the second value, and the z dimension (if present) the final value.
 * \param haloWidth
 *  Array of size 2, 4, or 6 (depending on the dimensionality) containing the widths of the CPU-to-CPU halos, i.e., the inter-MPI halo. The order in
 *  which the halo widths are expected to be stored is: LEFT -> RIGHT -> TOP -> BOTTOM -> FRONT -> BACK. Depending on the dimensionality of the
 *  problem only a subset might be possible to be specified.
 * \param numBuffers
 *  The number of buffers that will be used. If more than one, they are all combined into one message. All buffers will have to use the same
 *  discretisation! Typical value: 1.
 * \param valuesPerPoint
 *  How many values are stored consecutively per point in the same buffer. All points will have to have the same number of values stored for them.
 *  Typical value: 1
 * \param comm
 *  The MPI Communictor to be used. %CTauschFloat will duplicate the communicator, thus it is safe to have multiple instances of %CTauschFloat working
 *  with the same communicator. By default, MPI_COMM_WORLD will be used.
 * \param version
 *  Which version of CTauschFloat to create. This depends on the dimensionality of the problem and can be any one of the enum TAUSCH_VERSION: TAUSCH_1D,
 *  TAUSCH_2D, or TAUSCH_3D.
 *
 * \return
 *  Return the CTauschFloat object created with the specified configuration.
 *
 */
CTauschFloat *tausch_new_float(int *localDim, int *haloWidth, int numBuffers, int valuesPerPoint, MPI_Comm comm, TauschVersion version);

/*!
 *
 * Deletes the CTauschFloat object, cleaning up all the memory.
 *
 * \param tC
 *  The CTauschFloat object to be deleted.
 *
 */
void tausch_delete_float(CTauschFloat *tC);

/*!
 *
 * Set the info about all local halos that need to be sent to remote MPI ranks.
 *
 * \param tC
 *  The CTauschFloat object to operate on.
 * \param numHaloParts
 *  How many different parts there are to the halo
 * \param haloSpecs
 *  The specification of the different halo parts. This is expected to be a an array of arrays of int's. Each array of int's contains 4, 6, or 8
 *  entries (depending on the dimensionality of the problem), the order of which will be preserved, and each halo region can be referenced later by
 *  its index in this array. The entries are:
 *   1. The starting x coordinate of the local region
 *   2. The starting y coordinate of the local region (if present)
 *   3. The starting z coordinate of the local region (if present)
 *   4. The width of the region
 *   5. The height of the region (if present)
 *   6. The depth of the region (if present)
 *   7. The receiving processor
 *   8. A unique id that matches the id for the corresponding local halo region of the sending MPI rank.
 *
 */
void tausch_setCpuLocalHaloInfo_float(CTauschFloat *tC, int numHaloParts, int **haloSpecs);

/*!
 *
 * Set the info about all remote halos that are needed by this MPI rank.
 *
 * \param tC
 *  The CTauschFloat object to operate on.
 * \param numHaloParts
 *  How many different parts there are to the halo
 * \param haloSpecs
 *  The specification of the different halo parts. This is expected to be a an array of arrays of int's. Each array of int's contains 4, 6, or 8
 *  entries (depending on the dimensionality of the problem), the order of which will be preserved, and each halo region can be referenced later by
 *  its index in this array. The entries ares:
 *   1. The starting x coordinate of the halo region
 *   2. The starting y coordinate of the halo region (if present)
 *   3. The starting z coordinate of the halo region (if present)
 *   4. The width of the halo region
 *   5. The height of the halo region (if present)
 *   6. The depth of the halo region (if present)
 *   7. The sending processor
 *   8. A unique id that matches the id for the corresponding remote halo region of the receiving MPI rank.
 *
 */
void tausch_setCpuRemoteHaloInfo_float(CTauschFloat *tC, int numHaloParts, int **haloSpecs);

/*!
 *
 * Post all MPI receives for the current MPI rank. This doesn't do anything else but call MPI_Start() and all MPI_Recv_init().
 *
 * \param tC
 *  The CTauschFloat object to operate on.
 *
 */
void tausch_postMpiReceives_float(CTauschFloat *tC);

/*!
 *
 * This packs the next buffer for a send. This has to be called as many times as there are buffers before sending the message.
 *
 * \param tC
 *  The CTauschFloat object to operate on.
 * \param id
 *  The id of the halo region. This is the index of this halo region in the local halo specification provided with setLocalHaloInfo().
 * \param buf
 *  The buffer from which the data is to be extracted according to the local halo specification.
 *
 */
void tausch_packNextSendBuffer_float(CTauschFloat *tC, int id, float *buf);

/*!
 *
 * Sends off the send buffer for the specified halo region. This calls MPI_Start() on the respective MPI_Send_init().
 *
 * \param tC
 *  The CTauschFloat object to operate on.
 * \param id
 *  The id of the halo region. This is the index of this halo region in the local halo specification provided with setLocalHaloInfo().
 *
 */
void tausch_send_float(CTauschFloat *tC, int id);

/*!
 *
 * Makes sure the MPI message for the specified halo is received by this buffer. It does not do anything with that message!
 *
 * \param tC
 *  The CTauschFloat object to operate on.
 * \param id
 *  The id of the halo region. This is the index of this halo region in the remote halo specification provided with setRemoteHaloInfo().
 *
 */
void tausch_recv_float(CTauschFloat *tC, int id);

/*!
 *
 * This unpacks the next halo from the received message into provided buffer. This has to be called as many times as there are buffers.
 *
 * \param tC
 *  The CTauschFloat object to operate on.
 * \param id
 *  The id of the halo region. This is the index of this halo region in the remote halo specification provided with setRemoteHaloInfo().
 * \param buf
 *  The buffer to which the extracted data is to be written to according to the remote halo specification
 *
 */
void tausch_unpackNextRecvBuffer_float(CTauschFloat *tC, int id, float *buf);

/*!
 *
 * Shortcut function. If only one buffer is used, this will both pack the data out of the provided buffer and send it off, all with one call.
 *
 * \param tC
 *  The CTauschFloat object to operate on.
 * \param id
 *  The id of the halo region. This is the index of this halo region in the local halo specification provided with setLocalHaloInfo().
 * \param buf
 *  The buffer from which the data is to be extracted according to the local halo specification.
 *
 */
void tausch_packAndSend_float(CTauschFloat *tC, int id, float *buf);

/*!
 *
 * Shortcut function. If only one buffer is used, this will both receive the MPI message and unpack the received data into the provided buffer,
 * all with one call.
 *
 * \param tC
 *  The CTauschFloat object to operate on.
 * \param id
 *  The id of the halo region. This is the index of this halo region in the remote halo specification provided with setRemoteHaloInfo().
 * \param buf
 *  The buffer to which the extracted data is to be written to according to the remote halo specification
 *
 */
void tausch_recvAndUnpack_float(CTauschFloat *tC, int id, float *buf);

#ifdef __cplusplus
}
#endif


#endif // CTAUSCH2DFLOAT_H
