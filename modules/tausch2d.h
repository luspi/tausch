/*!
 * \file
 * \author  Lukas Spies <LSpies@illinois.edu>
 * \version 1.0
 */

#ifndef TAUSCH2D_H
#define TAUSCH2D_H

#include "tausch.h"
#include <mpi.h>
#include <iostream>

#ifndef TAUSCH_ENUM
#define TAUSCH_ENUM
namespace Tausch {
    /*!
     * These are the edges available for inter-MPI halo exchanges: LEFT, RIGHT, TOP, BOTTOM.
     */
    enum Edges { LEFT, RIGHT, TOP, BOTTOM };

    /*!
     * These are the two dimensions used, used for clarity as to which array entry is which dimension: X, Y.
     */
    enum Dimensions { X, Y, Z };

}
#endif // TAUSCH_ENUM

/*!
 *
 * \brief
 *  A library providing a clean and efficient interface for halo exchange in two dimensions.
 *
 * %Tausch2D is a library that provides a clean and efficient C and C++ API for halo exchange for two dimensional structured grids. It doesn't assume anything about the grid, except that the data is stored in one continuous buffer. After specifying the local and remote halo regions, it takes care of extracting the data for each halo from as many buffers as there are, sends the data for each halo combined into one message each, and unpacks the received data into the same number of halos again.
 */
template <class real_t>
class Tausch2D : public TauschBase {

public:

    /*!
     * The constructor, initiating the 2D Tausch object.
     *
     * \param localDim
     *  Array of size 2 holding the dimensions of the local partition (not the global dimensions), with the x dimension being the first value and the
     *  y dimension being the second one.
     * \param haloWidth
     *  Array of size 4 containing the widths of the CPU-to-CPU halos, i.e., the inter-MPI halo. The order in which the halo widths are expected to be
     *  stored is: LEFT -> RIGHT -> TOP -> BOTTOM
     * \param comm
     *  The MPI Communictor to be used. %Tausch2D will duplicate the communicator, thus it is safe to have multiple instances of %Tausch2D working
     *  with the same communicator. By default, MPI_COMM_WORLD will be used.
     */
    Tausch2D(int *localDim, int *haloWidth, int numBuffers = 1, int valuesPerPoint = 1, MPI_Comm comm = MPI_COMM_WORLD);

    /*!
     * The destructor cleaning up all memory.
     */
    ~Tausch2D();

    /*!
     * Set the info about all local halos that need to be sent to remote MPI ranks.
     * \param numBuffers
     *  The number of buffers there are. If more than one, they are all combined into one message.
     * \param numHaloParts
     *  How many different parts there are to the halo
     * \param haloSpecs
     *  The specification of the different halo parts. This is expected to be a an array of arrays of int's. Each array of int's contains 5 entries,
     *  the order of which will be preserved, and each halo region can be referenced later by its index in this array. The 5 entries are:
     *   1. The starting x coordinate of the local region
     *   2. The starting y coordinate of the local region
     *   3. The width of the region
     *   4. The height of the region
     *   5. The receiving processor
     */
    void setLocalHaloInfo(int numHaloParts, int **haloSpecs)
    ;
    /*!
     * Set the info about all remote halos that are needed by this MPI rank.
     * \param numBuffers
     *  The number of buffers there are. If more than one, they are all combined into one message.
     * \param numHaloParts
     *  How many different parts there are to the halo
     * \param haloSpecs
     *  The specification of the different halo parts. This is expected to be a an array of arrays of int's. Each array of int's contains 5 entrie,
     *  the order of which will be preserved, and each halo region can be referenced later by its index in this array. The 5 entries ares:
     *   1. The starting x coordinate of the halo region
     *   2. The starting y coordinate of the halo region
     *   3. The width of the halo region
     *   4. The height of the halo region
     *   5. The sending processor
     */
    void setRemoteHaloInfo(int numHaloParts, int **haloSpecs);

    /*!
     * Post all receives for the current MPI rank. This doesn't do anything else but call MPI_Start() and all MPI_Recv_init().
     */
    void postReceives();

    /*!
     * This packs the next buffer for a send. This has to be called as many times as there are buffers before sending the message.
     * \param id
     *  The id of the halo region. This is the index of this halo region in the local halo specification provided with setLocalHaloInfo().
     * \param buf
     *  The buffer from which the data is to be extracted according to the local halo specification.
     */
    void packNextSendBuffer(int id, real_t *buf);

    /*!
     * Sends off the send buffer for the specified halo region. This calls MPI_Start() on the respective MPI_Send_init().
     * \param id
     *  The id of the halo region. This is the index of this halo region in the local halo specification provided with setLocalHaloInfo().
     */
    void send(int id);

    /*!
     * Makes sure the MPI message for the specified halo is received by this buffer. It does not do anything with that message!
     * \param id
     *  The id of the halo region. This is the index of this halo region in the remote halo specification provided with setRemoteHaloInfo().
     */
    void recv(int id);

    /*!
     * This unpacks the next buffer from the received message. This has to be called as many times as there are buffers.
     * \param id
     *  The id of the halo region. This is the index of this halo region in the remote halo specification provided with setRemoteHaloInfo().
     * \param buf
     *  The buffer to which the extracted data is to be written to according to the remote halo specification
     */
    void unpackNextRecvBuffer(int id, real_t *buf);

    void packAndSend(int id, real_t *buf);
    void recvAndUnpack(int id, real_t *buf);

private:

    int localDim[2], haloWidth[4];

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
    MPI_Datatype mpiDatatype;

    int *numBuffersPacked;
    int *numBuffersUnpacked;

};


#endif // TAUSCH2D_H
