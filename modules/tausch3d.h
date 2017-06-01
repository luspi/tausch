/*!
 * \file
 * \author  Lukas Spies <LSpies@illinois.edu>
 * \version 1.0
 *
 * \brief
 *  Three-dimensional halo exchange library.
 *
 *  A library providing a clean and efficient interface for halo exchange in Three dimensions.
 *
 */

#ifndef TAUSCH3D_H
#define TAUSCH3D_H

#include "tausch.h"
#include <mpi.h>
#include <iostream>

/*!
 *
 * \brief
 *  A library providing a clean and efficient interface for halo exchange in two dimensions.
 *
 * %Tausch3D is a library that provides a clean and efficient C and C++ API for halo exchange for three dimensional domains. It doesn't assume
 * anything about the grid, except that the data is stored in one contiguous buffer. After specifying the local and remote halo regions, it takes care
 * of extracting the data for each halo from as many buffers as there are, sends the data for each halo combined into a single message each, and
 * unpacks the received data into the same number of buffers again. This is a template class and can be used with any of the following data types:
 * double, float, int, unsigned int, long, long long, long double.
 */
template <class real_t>
class Tausch3D : public Tausch<real_t> {

public:

    /*!
     * The constructor, initiating the 3D Tausch object.
     *
     * \param localDim
     *  Array of size 3 holding the dimensions of the local partition (not the global dimensions), with the x dimension being the first value, the
     *  y dimension being the second one, and the z dimension being the last one.
     * \param haloWidth
     *  Array of size 6 containing the widths of the CPU-to-CPU halos, i.e., the inter-MPI halo. The order in which the halo widths are expected to be
     *  stored is: LEFT -> RIGHT -> TOP -> BOTTOM -> FRONT -> BACK
     * \param numBuffers
     *  The number of buffers that will be used. If more than one, they are all combined into one message. All buffers will have to use the same
     *  discretisation! Default value: 1
     * \param valuesPerPoint
     *  How many values are stored consecutively per point in the same buffer. All points will have to have the same number of values stored for them.
     *  Default value: 1
     * \param comm
     *  The MPI Communictor to be used. %Tausch3D will duplicate the communicator, thus it is safe to have multiple instances of %Tausch3D working
     *  with the same communicator. By default, MPI_COMM_WORLD will be used.
     */
    Tausch3D(int *localDim, int *haloWidth, MPI_Datatype mpiDataType, int numBuffers = 1, int valuesPerPoint = 1, MPI_Comm comm = MPI_COMM_WORLD);

    /*!
     * The destructor cleaning up all memory.
     */
    ~Tausch3D();

    /*!
     * Set the info about all local halos that need to be sent to remote MPI ranks.
     * \param numHaloParts
     *  How many different parts there are to the halo
     * \param haloSpecs
     *  The specification of the different halo parts. This is expected to be a an array of arrays of int's. Each array of int's contains 8 entries,
     *  the order of which will be preserved, and each halo region can be referenced later by its index in this array. The 8 entries are:
     *   1. The starting x coordinate of the local region
     *   2. The starting y coordinate of the local region
     *   3. The starting z coordinate of the local region
     *   4. The width of the region
     *   5. The height of the region
     *   6. The depth of the region
     *   7. The receiving processor
     *   8. A unique id that matches the id for the corresponding local halo region of the sending MPI rank.
     */
    void setLocalHaloInfoCpu(int numHaloParts, int **haloSpecs);

    /*!
     * Set the info about all remote halos that are needed by this MPI rank.
     * \param numHaloParts
     *  How many different parts there are to the halo
     * \param haloSpecs
     *  The specification of the different halo parts. This is expected to be a an array of arrays of int's. Each array of int's contains 8 entries,
     *  the order of which will be preserved, and each halo region can be referenced later by its index in this array. The 8 entries ares:
     *   1. The starting x coordinate of the halo region
     *   2. The starting y coordinate of the halo region
     *   3. The starting z coordinate of the halo region
     *   4. The width of the halo region
     *   5. The height of the halo region
     *   6. The depth of the halo region
     *   7. The sending processor
     *   8. A unique id that matches the id for the corresponding remote halo region of the receiving MPI rank.
     */
    void setRemoteHaloInfoCpu(int numHaloParts, int **haloSpecs);

    /*!
     * Post all receives for the current MPI rank. This doesn't do anything else but call MPI_Start() and all MPI_Recv_init().
     */
    void postReceivesCpu();

    /*!
     * This packs the next buffer for a send. This has to be called as many times as there are buffers before sending the message.
     * \param id
     *  The id of the halo region. This is the index of this halo region in the local halo specification provided with setLocalHaloInfo().
     * \param buf
     *  The buffer from which the data is to be extracted according to the local halo specification.
     */
    void packNextSendBufferCpu(int id, real_t *buf);

    /*!
     * Sends off the send buffer for the specified halo region. This calls MPI_Start() on the respective MPI_Send_init().
     * \param id
     *  The id of the halo region. This is the index of this halo region in the local halo specification provided with setLocalHaloInfo().
     */
    void sendCpu(int id);

    /*!
     * Makes sure the MPI message for the specified halo is received by this buffer. It does not do anything with that message!
     * \param id
     *  The id of the halo region. This is the index of this halo region in the remote halo specification provided with setRemoteHaloInfo().
     */
    void recvCpu(int id);

    /*!
     * This unpacks the next halo from the received message into the provided buffer. This has to be called as many times as there are buffers.
     * \param id
     *  The id of the halo region. This is the index of this halo region in the remote halo specification provided with setRemoteHaloInfo().
     * \param[out] buf
     *  The buffer to which the extracted data is to be written to according to the remote halo specification
     */
    void unpackNextRecvBufferCpu(int id, real_t *buf);

    /*!
     * Shortcut function. If only one buffer is used, this will both pack the data out of the provided buffer and send it off, all with one call.
     * \param id
     *  The id of the halo region. This is the index of this halo region in the local halo specification provided with setLocalHaloInfo().
     * \param buf
     *  The buffer from which the data is to be extracted according to the local halo specification.
     */
    void packAndSendCpu(int id, real_t *buf);
    /*!
     * Shortcut function. If only one buffer is used, this will both receive the MPI message and unpack the received data into the provided buffer,
     * all with one call.
     * \param id
     *  The id of the halo region. This is the index of this halo region in the remote halo specification provided with setRemoteHaloInfo().
     * \param[out] buf
     *  The buffer to which the extracted data is to be written to according to the remote halo specification
     */
    void recvAndUnpackCpu(int id, real_t *buf);

private:

    int localDim[3], haloWidth[6];

    MPI_Comm TAUSCH_COMM;
    int mpiRank, mpiSize;

    int localHaloNumParts;
    int **localHaloSpecs;
    int remoteHaloNumParts;
    int **remoteHaloSpecs;

    int numBuffers;
    int valuesPerPoint;

    real_t **mpiRecvBuffer;
    real_t **mpiSendBuffer;
    MPI_Request *mpiRecvRequests;
    MPI_Request *mpiSendRequests;
    MPI_Datatype mpiDataType;

    int *numBuffersPacked;
    int *numBuffersUnpacked;

};


#endif // TAUSCH3D_H
