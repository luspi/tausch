/*!
 * \file
 * \author  Lukas Spies <LSpies@illinois.edu>
 * \version 1.0
 *
 * \brief
 *  Two-dimensional halo exchange library.
 *
 *  A library providing a clean and efficient interface for halo exchange in two dimensions.
 *
 */

#ifndef TAUSCH2D_H
#define TAUSCH2D_H

#include "tausch.h"
#include <mpi.h>
#include <iostream>

#ifndef TAUSCH_DIMENSIONS
#define TAUSCH_DIMENSIONS
/*!
 * These are the three dimensions that can be used with Tausch, providing better clarity as to which array entry is which dimension: X, Y, Z.
 * Depending on the dimensionality of the use case, not all might be available for use.
 */
enum TauschDimensions { TAUSCH_X, TAUSCH_Y, TAUSCH_Z };
#endif // TAUSCH_DIMENSIONS

/*!
 *
 * \brief
 *  A library providing a clean and efficient interface for halo exchange in two dimensions.
 *
 * %Tausch2D is a library that provides a clean and efficient C and C++ API for halo exchange for two dimensional domains. It doesn't assume
 * anything about the grid, except that the data is stored in one contiguous buffer. After specifying the local and remote halo regions, it takes care
 * of extracting the data for each halo from as many buffers as there are, sends the data for each halo combined into a single message each, and
 * unpacks the received data into the same number of buffers again. This is a template class and can be used with any of the following data types:
 * char, char16_t, char32_t, wchar_t, signed char, short int, int, long, long long, unsigned char, unsigned short int, unsigned int, unsigned long,
 * unsigned long long, float, double, long double, bool.
 */
template <class buf_t>
class Tausch2D : public Tausch<buf_t> {

public:

    /*!
     * The constructor, initiating the 2D Tausch object.
     *
     * \param localDim
     *  Array of size 2 holding the dimensions of the local partition (not the global dimensions), with the x dimension being the first value and the
     *  y dimension being the second one. Note: These dimensions <b>DO INCLUDE</b> the halo widths!
     * \param mpiDataType
     *  The MPI_Datatype corresponding to the datatype used for the template.
     * \param numBuffers
     *  The number of buffers that will be used. If more than one, they are all combined into one message. All buffers will have to use the same
     *  discretisation! Default value: 1
     * \param valuesPerPoint
     *  How many values are stored consecutively per point in the same buffer. All points will have to have the same number of values stored for them.
     *  Default value: 1
     * \param comm
     *  The MPI Communictor to be used. %Tausch2D will duplicate the communicator, thus it is safe to have multiple instances of %Tausch2D working
     *  with the same communicator. By default, MPI_COMM_WORLD will be used.
     */
    Tausch2D(int *localDim, MPI_Datatype mpiDataType, int numBuffers = 1, int valuesPerPoint = 1, MPI_Comm comm = MPI_COMM_WORLD);

    /*!
     * The destructor cleaning up all memory.
     */
    ~Tausch2D();

    /*!
     * Set the info about all local halos that need to be sent to remote MPI ranks.
     * \param numHaloParts
     *  How many different parts there are to the halo
     * \param haloSpecs
     *  The specification of the different halo parts. This is expected to be a nested array of int's, the order of which will be preserved,
     *  and each halo region can be referenced later by its index in this array. Each nested array must contain the following 5 entries:
     *   1. The starting x coordinate of the local region
     *   2. The starting y coordinate of the local region
     *   3. The width of the region
     *   4. The height of the region
     *   5. The receiving processor
     */
    void setLocalHaloInfoCpu(int numHaloParts, int **haloSpecs);

    /*!
     * Set the info about all remote halos that are needed by this MPI rank.
     * \param numHaloParts
     *  How many different parts there are to the halo
     * \param haloSpecs
     *  The specification of the different halo parts. This is expected to be a nested array of int's, the order of which will be preserved,
     *  and each halo region can be referenced later by its index in this array. Each nested array must contain the following 5 entries:
     *   1. The starting x coordinate of the halo region
     *   2. The starting y coordinate of the halo region
     *   3. The width of the halo region
     *   4. The height of the halo region
     *   5. The sending processor
     */
    void setRemoteHaloInfoCpu(int numHaloParts, int **haloSpecs);

    /*!
     * Post the receive for the specified remote halo region of the current MPI rank. This doesn't do anything else but call MPI_Irecv().
     * \param id
     *  The id of the halo region. This is the index of this halo region in the local halo specification provided with setRemoteHaloInfo().
     * \param mpitag
     *  The mpitag to be used for this MPI receive.
     */
    void postReceiveCpu(int id, int mpitag);

    /*!
     * Post all receives for the current MPI rank. This doesn't do anything else but call MPI_Irecv() for each remote halo region.
     * \param mpitag
     *  An array containing the MPI tags, one for each halo region.
     */
    void postAllReceivesCpu(int *mpitag);

    /*!
     * This packs the next buffer for a send. This has to be called as many times as there are buffers before sending the message.
     * \param id
     *  The id of the halo region. This is the index of this halo region in the local halo specification provided with setLocalHaloInfo().
     * \param buf
     *  The buffer from which the data is to be extracted according to the local halo specification.
     */
    void packNextSendBufferCpu(int id, buf_t *buf);

    /*!
     * Sends off the send buffer for the specified halo region. This calls MPI_Start() on the respective MPI_Send_init().
     * \param id
     *  The id of the halo region. This is the index of this halo region in the local halo specification provided with setLocalHaloInfo().
     * \param mpitag
     *  The mpitag to be used for this MPI_Isend().
     */
    void sendCpu(int id, int mpitag);

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
    void unpackNextRecvBufferCpu(int id, buf_t *buf);

    /*!
     * Shortcut function. If only one buffer is used, this will both pack the data out of the provided buffer and send it off, all with one call.
     * \param id
     *  The id of the halo region. This is the index of this halo region in the local halo specification provided with setLocalHaloInfo().
     * \param buf
     *  The buffer from which the data is to be extracted according to the local halo specification.
     * \param mpitag
     *  The mpitag to be used for this MPI_Isend().
     */
    void packAndSendCpu(int id, int mpitag, buf_t *buf);
    /*!
     * Shortcut function. If only one buffer is used, this will both receive the MPI message and unpack the received data into the provided buffer,
     * all with one call.
     * \param id
     *  The id of the halo region. This is the index of this halo region in the remote halo specification provided with setRemoteHaloInfo().
     * \param[out] buf
     *  The buffer to which the extracted data is to be written to according to the remote halo specification
     */
    void recvAndUnpackCpu(int id, buf_t *buf);

private:

    int localDim[2];

    MPI_Comm TAUSCH_COMM;
    int mpiRank, mpiSize;

    int localHaloNumParts;
    int **localHaloSpecs;
    int remoteHaloNumParts;
    int **remoteHaloSpecs;

    int numBuffers;
    int valuesPerPoint;

    buf_t **mpiRecvBuffer;
    buf_t **mpiSendBuffer;
    MPI_Request *mpiRecvRequests;
    MPI_Request *mpiSendRequests;
    MPI_Datatype mpiDataType;

    int *numBuffersPacked;
    int *numBuffersUnpacked;

};


#endif // TAUSCH2D_H
