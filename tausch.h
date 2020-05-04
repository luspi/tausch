#ifndef TAUSCH_H
#define TAUSCH_H

#include <mpi.h>
#include <vector>
#include <array>
#include <memory>
#include <iostream>
#include <map>
#include <cstring>

#ifdef TAUSCH_CUDA
#include <cuda_runtime.h>
#endif

class Tausch {

public:

    /**
     * @brief
     * Constructor of a new Tausch object.
     *
     * This constructs a new Tausch object. The same object can be used for any trivial type.
     * Only one Tausch object should be used per MPI rank.
     *
     * @param comm
     * The default communicator to use. Defaults to MPI_COMM_WORLD.
     * When sending/receiving data the communicator can be temporarily overwritten.
     * @param useDuplicateOfCommunicator
     * By default Tausch will duplicate the communicator. This isolates Tausch from
     * the rest of the (MPI) world and avoids any potential interference.
     */
    Tausch(const MPI_Comm comm = MPI_COMM_WORLD, const bool useDuplicateOfCommunicator = true) {

        if(useDuplicateOfCommunicator)
            MPI_Comm_dup(comm, &TAUSCH_COMM);
        else
            TAUSCH_COMM = comm;

    }

    /**
     * @brief
     * This enum can be used when enabling/disabling certain optimization strategies.
     */
    enum Communication {
        Auto = 1,
        DerivedMpiDatatype = 2
    };

    /***********************************************************************/
    /*                           ADD LOCAL HALO                            */
    /***********************************************************************/

    /**
     * @brief
     * Add a new halo region (overloaded function).
     *
     * This overloaded function adds a new halo region to the Tausch object.
     *
     * @param haloIndices
     * The halo indices can be specified as a vector of integers specifying the location
     * where the halo data is stored in some buffer. Tausch will internally convert this
     * list of integers into a set of rectangular subregions with each subregion stored
     * using four integers.
     * @param typeSize
     * The size of the data type for this halo (same type for all buffers).
     * @param numBuffers
     * How many buffers are to be used for the given halo specification.
     * @param remoteMpiRank
     * What remote MPI rank this/these halo region/s will be sent to.
     *
     * @return
     * This function returns the halo id (needed for referencing this halo later-on).
     */
    inline size_t addSendHaloInfo(std::vector<int> haloIndices,
                                  const size_t typeSize,
                                  const int numBuffers = 1,
                                  const int remoteMpiRank = -1) {
        std::vector<std::vector<std::array<int, 4> > > indices;
        std::vector<size_t> typeSizePerBuffer;
        for(size_t i = 0; i < numBuffers; ++i) {
            indices.push_back(extractHaloIndicesWithStride(haloIndices));
            typeSizePerBuffer.push_back(typeSize);
        }
        return addSendHaloInfo(indices, typeSizePerBuffer, remoteMpiRank);
    }

    /**
     * @brief
     * Add a new halo region (overloaded function).
     *
     * This overloaded function adds a new halo region to the Tausch object.
     *
     * @param haloIndices
     * The halo indices can be specified as a vector of integers specifying the location
     * where the halo data is stored in some buffer. Tausch will internally convert this
     * list of integers into a set of rectangular subregions with each subregion stored
     * using four integers. Each buffer can have their own halo indices.
     * @param typeSize
     * The size of the data type for this halo (same type for all buffers).
     * @param remoteMpiRank
     * What remote MPI rank this/these halo region/s will be sent to.
     *
     * @return
     * This function returns the halo id (needed for referencing this halo later-on).
     */
    inline size_t addSendHaloInfo(std::vector<std::vector<int> > haloIndices,
                                  const size_t typeSize,
                                  const int remoteMpiRank = -1) {
        std::vector<std::vector<std::array<int, 4> > > indices;
        std::vector<size_t> typeSizePerBuffer;
        for(auto bufIndices : haloIndices) {
            indices.push_back(extractHaloIndicesWithStride(bufIndices));
            typeSizePerBuffer.push_back(typeSize);
        }
        return addSendHaloInfo(indices, typeSizePerBuffer, remoteMpiRank);
    }

    /**
     * @brief
     * Add a new halo region (overloaded function).
     *
     * This overloaded function adds a new halo region to the Tausch object.
     *
     * @param haloIndices
     * The halo indices can be specified as vector of arrays. Each array consists of four
     * integers describing a rectangular subregion of the halo. All arrays together
     * describe the full halo region.
     * @param typeSize
     * The size of the data type for this halo (same type for all buffers).
     * @param numBuffers
     * How many buffers are to be used for the given halo specification.
     * @param remoteMpiRank
     * What remote MPI rank this/these halo region/s will be sent to.
     *
     * @return
     * This function returns the halo id (needed for referencing this halo later-on).
     */
    inline size_t addSendHaloInfo(std::vector<std::array<int, 4> > haloIndices,
                                  const size_t typeSize,
                                  const int numBuffers = 1,
                                  const int remoteMpiRank = -1) {
        std::vector<size_t> typeSizePerBuffer;
        for(size_t i = 0; i < numBuffers; ++i)
            typeSizePerBuffer.push_back(typeSize);
        return addSendHaloInfo(haloIndices, typeSizePerBuffer, remoteMpiRank);
    }

    /**
     * @brief
     *  Add a new halo region (overloaded function).
     *
     * This overloaded function adds a new halo region to the Tausch object.
     *
     * @param haloIndices
     *  The halo indices are specified as vectors with two levels:<br>
     *  1) The first level corresponds to the buffer id. Multiple buffers can be combined into a single message and each buffer can have their own halo specification.<br>
     *  2) The second level corresponds to a rectangular subregion of this halo specification, with each rectangular subregion specified using four integers.
     * @param typeSize
     * The size of the data type for this halo (same type for all buffers).
     * @param remoteMpiRank
     *  What remote MPI rank this/these halo region/s will be sent to.
     *
     * @return
     *  This function returns the halo id (needed for referencing this halo later-on).
     */
    inline size_t addSendHaloInfo(std::vector<std::vector<std::array<int, 4> > > haloIndices,
                                  const size_t typeSize,
                                  const int remoteMpiRank = -1) {
        std::vector<size_t> typeSizePerBuffer;
        for(size_t bufferId = 0; bufferId < haloIndices.size(); ++bufferId)
            typeSizePerBuffer.push_back(typeSize);
        return addSendHaloInfo(haloIndices, typeSizePerBuffer, remoteMpiRank);
    }

    /**
     * @brief
     * Add a new halo region (overloaded function).
     *
     * This overloaded function adds a new halo region to the Tausch object.
     *
     * @param haloIndices
     * The halo indices can be specified as a vector of integers specifying the location
     * where the halo data is stored in some buffer. Tausch will internally convert this
     * list of integers into a set of rectangular subregions with each subregion stored
     * using four integers.
     * @param typeSizePerBuffer
     * The size of the data type for this halo, one for each buffer.
     * @param remoteMpiRank
     * What remote MPI rank this/these halo region/s will be sent to.
     *
     * @return
     * This function returns the halo id (needed for referencing this halo later-on).
     */
    inline size_t addSendHaloInfo(std::vector<int> haloIndices,
                                  const std::vector<size_t> typeSizePerBuffer,
                                  const int remoteMpiRank = -1) {
        std::vector<std::vector<std::array<int, 4> > > indices;
        for(size_t i = 0; i < typeSizePerBuffer.size(); ++i) {
            auto ind = extractHaloIndicesWithStride(haloIndices);
            indices.push_back(ind);
        }
        return addSendHaloInfo(indices, typeSizePerBuffer, remoteMpiRank);
    }

    /**
     * @brief
     * Add a new halo region (overloaded function).
     *
     * This overloaded function adds a new halo region to the Tausch object.
     *
     * @param haloIndices
     * The halo indices can be specified as a vector of integers specifying the location
     * where the halo data is stored in some buffer. Tausch will internally convert this
     * list of integers into a set of rectangular subregions with each subregion stored
     * using four integers. Each buffer can have their own halo indices.
     * @param typeSizePerBuffer
     * The size of the data type for this halo, one for each buffer.
     * @param remoteMpiRank
     * What remote MPI rank this/these halo region/s will be sent to.
     *
     * @return
     * This function returns the halo id (needed for referencing this halo later-on).
     */
    inline size_t addSendHaloInfo(std::vector<std::vector<int> > haloIndices,
                                  const std::vector<size_t> typeSizePerBuffer,
                                  const int remoteMpiRank = -1) {
        std::vector<std::vector<std::array<int, 4> > > indices;
        for(size_t i = 0; i < haloIndices.size(); ++i)
            indices.push_back(extractHaloIndicesWithStride(haloIndices[i]));
        return addSendHaloInfo(indices, typeSizePerBuffer, remoteMpiRank);
    }

    /**
     * @brief
     * Add a new halo region (overloaded function).
     *
     * This overloaded function adds a new halo region to the Tausch object.
     *
     * @param haloIndices
     * The halo indices can be specified as vector of arrays. Each array consists of four
     * integers describing a rectangular subregion of the halo. All arrays together
     * describe the full halo region.
     * @param typeSizePerBuffer
     * The size of the data type for this halo, one for each buffer.
     * @param remoteMpiRank
     * What remote MPI rank this/these halo region/s will be sent to.
     *
     * @return
     * This function returns the halo id (needed for referencing this halo later-on).
     */
    inline size_t addSendHaloInfo(std::vector<std::array<int, 4> > haloIndices,
                                  const std::vector<size_t> typeSizePerBuffer,
                                  const int remoteMpiRank = -1) {
        std::vector<std::vector<std::array<int, 4> > > indices;
        for(size_t i = 0; i < typeSizePerBuffer.size(); ++i)
            indices.push_back(haloIndices);
        return addSendHaloInfo(indices, typeSizePerBuffer, remoteMpiRank);
    }

    /**
     * @brief
     * Add a new halo region.
     *
     * This adds a new halo region to the Tausch object.
     *
     * @param haloIndices
     * The halo indices are specified as vectors with two levels:<br>
     * 1) The first level corresponds to the buffer id. Multiple buffers can be combined into a single message and each buffer can have their own halo specification.<br>
     * 2) The second level corresponds to a rectangular subregion of this halo specification, with each rectangular subregion specified using four integers.
     * @param typeSizePerBuffer
     * A vector with the size of the data type for each buffer
     * @param remoteMpiRank
     * What remote MPI rank this/these halo region/s will be sent to.
     *
     * @return
     * This function returns the halo id (needed for referencing this halo later-on).
     */
    inline size_t addSendHaloInfo(std::vector<std::vector<std::array<int, 4> > > haloIndices,
                                  const std::vector<size_t> typeSizePerBuffer,
                                  const int remoteMpiRank = -1) {

        std::vector<std::vector<std::array<int, 4> > > indices;
        for(size_t i = 0; i < haloIndices.size(); ++i)
            indices.push_back(convertToUnsignedCharIndices(haloIndices[i], typeSizePerBuffer[i]));

        int totalHaloSize = 0;
        std::vector<int> haloSizePerBuffer;
        for(size_t bufferId = 0; bufferId < indices.size(); ++bufferId) {
            size_t bufHaloSize = 0;
            for(size_t iSize = 0; iSize < indices[bufferId].size(); ++iSize) {
                auto tuple = indices[bufferId][iSize];
                auto s = tuple[1]*tuple[2];
                totalHaloSize += s;
                bufHaloSize += s;
            }
            haloSizePerBuffer.push_back(bufHaloSize);
        }

        sendHaloIndices.push_back(indices);
        sendHaloIndicesSizePerBuffer.push_back(haloSizePerBuffer);
        sendHaloIndicesSizeTotal.push_back(totalHaloSize);
        sendHaloNumBuffers.push_back(indices.size());
        sendHaloCommunicationStrategy.push_back(Communication::Auto);
        sendHaloRemoteRank.push_back(remoteMpiRank);

        void *newbuf = NULL;
        posix_memalign(&newbuf, 64, totalHaloSize*sizeof(unsigned char));
        unsigned char *newbuf_buft = reinterpret_cast<unsigned char*>(newbuf);
        unsigned char zero = 0;
        std::fill_n(newbuf_buft, totalHaloSize, zero);
        sendBuffer.push_back(std::unique_ptr<unsigned char[]>(std::move(newbuf_buft)));

        std::vector<MPI_Request> perBufRequests;
        std::vector<bool> perBufSetup;
        for(int iBuf = 0; iBuf < haloSizePerBuffer.size(); ++iBuf) {
            perBufRequests.push_back(MPI_Request());
            perBufSetup.push_back(false);
        }
        sendHaloMpiRequests.push_back(perBufRequests);
        sendHaloMpiSetup.push_back(perBufSetup);

        return sendBuffer.size()-1;

    }

    inline void delSendHaloInfo(size_t haloId) {
        sendBuffer[haloId].reset(nullptr);
    }

    /***********************************************************************/
    /*                          ADD REMOTE HALO                            */
    /***********************************************************************/

    inline size_t addRecvHaloInfo(std::vector<int> haloIndices,
                                  const size_t typeSize,
                                  const int numBuffers = 1,
                                  const int remoteMpiRank = -1) {
        std::vector<std::vector<std::array<int, 4> > > indices;
        std::vector<size_t> typeSizePerBuffer;
        for(size_t i = 0; i < numBuffers; ++i) {
            indices.push_back(extractHaloIndicesWithStride(haloIndices));
            typeSizePerBuffer.push_back(typeSize);
        }
        return addRecvHaloInfo(indices, typeSizePerBuffer, remoteMpiRank);
    }

    inline size_t addRecvHaloInfo(std::vector<std::vector<int> > haloIndices,
                                  const size_t typeSize,
                                  const int remoteMpiRank = -1) {
        std::vector<std::vector<std::array<int, 4> > > indices;
        std::vector<size_t> typeSizePerBuffer;
        for(auto bufIndices : haloIndices) {
            indices.push_back(extractHaloIndicesWithStride(bufIndices));
            typeSizePerBuffer.push_back(typeSize);
        }
        return addRecvHaloInfo(indices, typeSizePerBuffer, remoteMpiRank);
    }

    inline size_t addRecvHaloInfo(std::vector<std::array<int, 4> > haloIndices,
                                  const size_t typeSize,
                                  const int numBuffers = 1,
                                  const int remoteMpiRank = -1) {
        std::vector<std::vector<std::array<int, 4> > > indices;
        std::vector<size_t> typeSizePerBuffer;
        for(size_t i = 0; i < numBuffers; ++i) {
            indices.push_back(haloIndices);
            typeSizePerBuffer.push_back(typeSize);
        }
        return addRecvHaloInfo(indices, typeSizePerBuffer, remoteMpiRank);
    }

    inline size_t addRecvHaloInfo(std::vector<std::vector<std::array<int, 4> > > haloIndices,
                                  const size_t typeSize,
                                  const int remoteMpiRank = -1) {
        std::vector<size_t> typeSizePerBuffer;
        for(size_t bufferId = 0; bufferId < haloIndices.size(); ++bufferId)
            typeSizePerBuffer.push_back(typeSize);
        return addRecvHaloInfo(haloIndices, typeSizePerBuffer, remoteMpiRank);
    }

    inline size_t addRecvHaloInfo(std::vector<int> haloIndices,
                                  const std::vector<size_t> typeSizePerBuffer,
                                  const int remoteMpiRank = -1) {
        std::vector<std::vector<std::array<int, 4> > > indices;
        for(size_t i = 0; i < typeSizePerBuffer.size(); ++i)
            indices.push_back(extractHaloIndicesWithStride(haloIndices));
        return addRecvHaloInfo(indices, typeSizePerBuffer, remoteMpiRank);
    }

    inline size_t addRecvHaloInfo(std::vector<std::vector<int> > haloIndices,
                                  const std::vector<size_t> typeSizePerBuffer,
                                  const int remoteMpiRank = -1) {
        std::vector<std::vector<std::array<int, 4> > > indices;
        for(size_t i = 0; i < typeSizePerBuffer.size(); ++i)
            indices.push_back(extractHaloIndicesWithStride(haloIndices[i]));
        return addRecvHaloInfo(indices, typeSizePerBuffer, remoteMpiRank);
    }

    inline size_t addRecvHaloInfo(std::vector<std::array<int, 4> > haloIndices,
                                  const std::vector<size_t> typeSizePerBuffer,
                                  const int remoteMpiRank = -1) {
        std::vector<std::vector<std::array<int, 4> > > indices;
        for(size_t i = 0; i < typeSizePerBuffer.size(); ++i)
            indices.push_back(haloIndices);
        return addRecvHaloInfo(indices, typeSizePerBuffer, remoteMpiRank);
    }

    inline size_t addRecvHaloInfo(std::vector<std::vector<std::array<int, 4> > > haloIndices,
                                  const std::vector<size_t> typeSizePerBuffer,
                                  const int remoteMpiRank = -1) {

        std::vector<std::vector<std::array<int, 4> > > indices;
        for(size_t i = 0; i < haloIndices.size(); ++i)
            indices.push_back(convertToUnsignedCharIndices(haloIndices[i], typeSizePerBuffer[i]));

        size_t totalHaloSize = 0;
        std::vector<int> haloSizePerBuffer;
        for(size_t bufferId = 0; bufferId < indices.size(); ++bufferId) {
            int bufHaloSize = 0;
            for(size_t iSize = 0; iSize < indices[bufferId].size(); ++iSize) {
                auto tuple = indices[bufferId][iSize];
                auto s = tuple[1]*tuple[2];
                totalHaloSize += s;
                bufHaloSize += s;
            }
            haloSizePerBuffer.push_back(bufHaloSize);
        }

        recvHaloIndices.push_back(indices);
        recvHaloIndicesSizePerBuffer.push_back(haloSizePerBuffer);
        recvHaloIndicesSizeTotal.push_back(totalHaloSize);
        recvHaloNumBuffers.push_back(indices.size());
        recvHaloCommunicationStrategy.push_back(Communication::Auto);
        recvHaloRemoteRank.push_back(remoteMpiRank);

        void *newbuf = NULL;
        posix_memalign(&newbuf, 64, totalHaloSize*sizeof(unsigned char));
        unsigned char *newbuf_buft = reinterpret_cast<unsigned char*>(newbuf);
        unsigned char zero = 0;
        std::fill_n(newbuf_buft, totalHaloSize, zero);
        recvBuffer.push_back(std::unique_ptr<unsigned char[]>(std::move(newbuf_buft)));

        std::vector<MPI_Request> perBufRequests;
        std::vector<bool> perBufSetup;
        for(int iBuf = 0; iBuf < haloSizePerBuffer.size(); ++iBuf) {
            perBufRequests.push_back(MPI_Request());
            perBufSetup.push_back(false);
        }
        recvHaloMpiRequests.push_back(perBufRequests);
        recvHaloMpiSetup.push_back(perBufSetup);

        return recvBuffer.size()-1;

    }

    inline void delRecvHaloInfo(size_t haloId) {
        recvBuffer[haloId].reset(nullptr);
    }

    /***********************************************************************/
    /*                      SET COMMUNICATION STRATEGY                     */
    /***********************************************************************/

    void setSendCommunicationStrategy(size_t haloId, Communication strategy) {

        sendHaloCommunicationStrategy[haloId] = strategy;

        if((strategy&Communication::DerivedMpiDatatype) == Communication::DerivedMpiDatatype) {

            std::vector<MPI_Datatype> sendHaloTypePerBuffer;

            for(auto const & perbuf : sendHaloIndices[haloId]) {

                std::vector<MPI_Datatype> vectorDataTypes;
                std::vector<MPI_Aint> displacement;
                std::vector<int> blocklength;

                vectorDataTypes.reserve(perbuf.size());
                displacement.reserve(perbuf.size());
                blocklength.reserve(perbuf.size());

                for(auto const & item : perbuf) {

                    MPI_Datatype vec;
                    MPI_Type_vector(item[2], item[1], item[3], MPI_CHAR, &vec);
                    MPI_Type_commit(&vec);

                    vectorDataTypes.push_back(vec);
                    displacement.push_back(item[0]);
                    blocklength.push_back(1);

                }

                MPI_Datatype newtype;
                MPI_Type_create_struct(perbuf.size(), blocklength.data(), displacement.data(), vectorDataTypes.data(), &newtype);
                MPI_Type_commit(&newtype);
                sendHaloDerivedDatatype[haloId].push_back(newtype);

            }

            sendBuffer[haloId] = std::unique_ptr<unsigned char[]>(new unsigned char[1]);

        }

    }

    void setRecvCommunicationStrategy(size_t haloId, Communication strategy) {

        recvHaloCommunicationStrategy[haloId] = strategy;

        if((strategy&Communication::DerivedMpiDatatype) == Communication::DerivedMpiDatatype) {

            std::vector<MPI_Datatype> recvHaloTypePerBuffer;

            for(auto const & perbuf : recvHaloIndices[haloId]) {

                std::vector<MPI_Datatype> vectorDataTypes;
                std::vector<MPI_Aint> displacement;
                std::vector<int> blocklength;

                vectorDataTypes.reserve(perbuf.size());
                displacement.reserve(perbuf.size());
                blocklength.reserve(perbuf.size());

                for(auto const & item : perbuf) {

                    MPI_Datatype vec;
                    MPI_Type_vector(item[2], item[1], item[3], MPI_CHAR, &vec);
                    MPI_Type_commit(&vec);

                    vectorDataTypes.push_back(vec);
                    displacement.push_back(item[0]);
                    blocklength.push_back(1);

                }

                MPI_Datatype newtype;
                MPI_Type_create_struct(perbuf.size(), blocklength.data(), displacement.data(), vectorDataTypes.data(), &newtype);
                MPI_Type_commit(&newtype);
                recvHaloDerivedDatatype[haloId].push_back(newtype);

            }

            recvBuffer[haloId] = std::unique_ptr<unsigned char[]>(new unsigned char[1]);

        }

    }



    /***********************************************************************/
    /*                          SEND HALO BUFFER                           */
    /***********************************************************************/

    void setSendHaloBuffer(int haloId, int bufferId, int* buf) {
        setSendHaloBuffer(haloId, bufferId, reinterpret_cast<unsigned char*>(buf));
    }

    void setSendHaloBuffer(int haloId, int bufferId, double* buf) {
        setSendHaloBuffer(haloId, bufferId, reinterpret_cast<unsigned char*>(buf));
    }

    void setSendHaloBuffer(int haloId, int bufferId, unsigned char* buf) {
        sendHaloBuffer[haloId][bufferId] = buf;
    }



    /***********************************************************************/
    /*                          RECV HALO BUFFER                           */
    /***********************************************************************/

    void setRecvHaloBuffer(int haloId, int bufferId, int* buf) {
        setRecvHaloBuffer(haloId, bufferId, reinterpret_cast<unsigned char*>(buf));
    }

    void setRecvHaloBuffer(int haloId, int bufferId, double* buf) {
        setRecvHaloBuffer(haloId, bufferId, reinterpret_cast<unsigned char*>(buf));
    }

    void setRecvHaloBuffer(int haloId, int bufferId, unsigned char* buf) {
        recvHaloBuffer[haloId][bufferId] = buf;
    }



    /***********************************************************************/
    /*                             PACK BUFFER                             */
    /***********************************************************************/

    void packSendBuffer(const size_t haloId, const size_t bufferId, const double *buf) {
        packSendBuffer(haloId, bufferId, reinterpret_cast<const unsigned char*>(buf));
    }

    void packSendBuffer(const size_t haloId, const size_t bufferId, const int *buf) {
        packSendBuffer(haloId, bufferId, reinterpret_cast<const unsigned char*>(buf));
    }

    void packSendBuffer(const size_t haloId, const size_t bufferId, const unsigned char *buf) {

        size_t bufferOffset = 0;
        for(size_t i = 0; i < bufferId; ++i)
            bufferOffset += sendHaloIndicesSizePerBuffer[haloId][i];

        size_t mpiSendBufferIndex = 0;
        for(auto const & region : sendHaloIndices[haloId][bufferId]) {

            const size_t &region_start = region[0];
            const size_t &region_howmanycols = region[1];
            const size_t &region_howmanyrows = region[2];
            const size_t &region_stridecol = region[3];

            for(size_t rows = 0; rows < region_howmanyrows; ++rows) {

                std::memcpy(&sendBuffer[haloId][bufferOffset + mpiSendBufferIndex], &buf[region_start + rows*region_stridecol], region_howmanycols);
                mpiSendBufferIndex += region_howmanycols;

            }

        }

    }

    /***********************************************************************/
    /*                            SEND MESSAGE                             */
    /***********************************************************************/

    inline MPI_Request send(size_t haloId, const int msgtag, const int remoteMpiRank = -1, const int bufferId = -1, const bool blocking = false, MPI_Comm communicator = MPI_COMM_NULL) {

        if(sendHaloIndicesSizeTotal[haloId] == 0)
            return MPI_REQUEST_NULL;

        if(communicator == MPI_COMM_NULL)
            communicator = TAUSCH_COMM;

        int useRemoteMpiRank = sendHaloRemoteRank.at(haloId);
        if(remoteMpiRank != -1)
            useRemoteMpiRank = remoteMpiRank;

        // if we stay on the same rank, we don't need to use MPI
        int myRank;
        MPI_Comm_rank(communicator, &myRank);
        if(useRemoteMpiRank == myRank && (sendHaloCommunicationStrategy[haloId]&Communication::Auto) == Communication::Auto) {
            msgtagToHaloId[myRank*1000000 + msgtag] = haloId;
            return MPI_REQUEST_NULL;
        }

        int useBufferId = 0;
        if((sendHaloCommunicationStrategy[haloId]&Communication::DerivedMpiDatatype) == Communication::DerivedMpiDatatype)
            useBufferId = bufferId;

        if(!sendHaloMpiSetup[haloId][useBufferId]) {

            sendHaloMpiSetup[haloId][useBufferId] = true;

            if((sendHaloCommunicationStrategy[haloId]&Communication::DerivedMpiDatatype) == Communication::DerivedMpiDatatype)
                MPI_Send_init(sendHaloBuffer[haloId][useBufferId], 1, sendHaloDerivedDatatype[haloId][useBufferId],
                              useRemoteMpiRank, msgtag, communicator,
                              &sendHaloMpiRequests[haloId][useBufferId]);
            else
                MPI_Send_init(sendBuffer[haloId].get(), sendHaloIndicesSizeTotal[haloId], MPI_CHAR,
                              useRemoteMpiRank, msgtag, communicator,
                              &sendHaloMpiRequests[haloId][0]);

        } else

            MPI_Wait(&sendHaloMpiRequests[haloId][useBufferId], MPI_STATUS_IGNORE);

        MPI_Start(&sendHaloMpiRequests[haloId][useBufferId]);
        if(blocking)
            MPI_Wait(&sendHaloMpiRequests[haloId][useBufferId], MPI_STATUS_IGNORE);

        return sendHaloMpiRequests[haloId][useBufferId];

    }

    /***********************************************************************/
    /*                         RECEIVE MESSAGE                             */
    /***********************************************************************/

    inline MPI_Request recv(size_t haloId, const int msgtag, const int remoteMpiRank = -1, const int bufferId = -1, const bool blocking = true, MPI_Comm communicator = MPI_COMM_NULL) {

        if(recvHaloIndicesSizeTotal[haloId] == 0)
            return MPI_REQUEST_NULL;

        if(communicator == MPI_COMM_NULL)
            communicator = TAUSCH_COMM;

        int useRemoteMpiRank = recvHaloRemoteRank.at(haloId);
        if(remoteMpiRank != -1)
            useRemoteMpiRank = remoteMpiRank;

        // if we stay on the same rank, we don't need to use MPI
        int myRank;
        MPI_Comm_rank(communicator, &myRank);
        if(useRemoteMpiRank == myRank && (recvHaloCommunicationStrategy[haloId]&Communication::Auto) == Communication::Auto) {
            const int remoteHaloId = msgtagToHaloId[myRank*1000000 + msgtag];
            std::memcpy(recvBuffer[haloId].get(), sendBuffer[remoteHaloId].get(), recvHaloIndicesSizeTotal[haloId]);
            return MPI_REQUEST_NULL;
        }

        int useBufferId = 0;
        if((recvHaloCommunicationStrategy[haloId]&Communication::DerivedMpiDatatype) == Communication::DerivedMpiDatatype)
            useBufferId = bufferId;

        if(!recvHaloMpiSetup[haloId][useBufferId]) {

            recvHaloMpiSetup[haloId][useBufferId] = true;

            if((recvHaloCommunicationStrategy[haloId]&Communication::DerivedMpiDatatype) == Communication::DerivedMpiDatatype)
                MPI_Recv_init(recvHaloBuffer[haloId][useBufferId], 1, recvHaloDerivedDatatype[haloId][useBufferId],
                              useRemoteMpiRank, msgtag, communicator,
                              &recvHaloMpiRequests[haloId][useBufferId]);
            else
                MPI_Recv_init(recvBuffer[haloId].get(), recvHaloIndicesSizeTotal[haloId], MPI_CHAR,
                              useRemoteMpiRank, msgtag, communicator,
                              &recvHaloMpiRequests[haloId][0]);

        } else
            MPI_Wait(&recvHaloMpiRequests[haloId][useBufferId], MPI_STATUS_IGNORE);

        MPI_Start(&recvHaloMpiRequests[haloId][useBufferId]);
        if(blocking)
            MPI_Wait(&recvHaloMpiRequests[haloId][useBufferId], MPI_STATUS_IGNORE);

        return recvHaloMpiRequests[haloId][useBufferId];

    }

    /***********************************************************************/
    /*                           UNPACK BUFFER                             */
    /***********************************************************************/

    void unpackRecvBuffer(const size_t haloId, const size_t bufferId, double *buf) {
        unpackRecvBuffer(haloId, bufferId, reinterpret_cast<unsigned char*>(buf));
    }

    void unpackRecvBuffer(const size_t haloId, const size_t bufferId, int *buf) {
        unpackRecvBuffer(haloId, bufferId, reinterpret_cast<unsigned char*>(buf));
    }

    void unpackRecvBuffer(const size_t haloId, const size_t bufferId, unsigned char *buf) {

        size_t bufferOffset = 0;
        for(size_t i = 0; i < bufferId; ++i)
            bufferOffset += recvHaloIndicesSizePerBuffer[haloId][i];

        size_t mpiRecvBufferIndex = 0;

        for(auto const & region : recvHaloIndices[haloId][bufferId]) {

            const size_t &region_start = region[0];
            const size_t &region_howmanycols = region[1];
            const size_t &region_howmanyrows = region[2];
            const size_t &region_stridecol = region[3];

            for(size_t rows = 0; rows < region_howmanyrows; ++rows) {

                std::memcpy(&buf[region_start + rows*region_stridecol], &recvBuffer[haloId][bufferOffset + mpiRecvBufferIndex], region_howmanycols);
                mpiRecvBufferIndex += region_howmanycols;

            }

        }

    }

    /***********************************************************************/
    /***********************************************************************/

#ifdef TAUSCH_CUDA

    /***********************************************************************/
    /*                         PACK BUFFER (CUDA)                          */
    /***********************************************************************/

    void packSendBufferCUDA(const size_t haloId, const size_t bufferId, const double *buf) {
        packSendBufferCUDA(haloId, bufferId, reinterpret_cast<const unsigned char*>(buf));
    }

    void packSendBufferCUDA(const size_t haloId, const size_t bufferId, const int *buf) {
        packSendBufferCUDA(haloId, bufferId, reinterpret_cast<const unsigned char*>(buf));
    }

    void packSendBufferCUDA(const size_t haloId, const size_t bufferId, const unsigned char *buf) {

        size_t bufferOffset = 0;
        for(size_t i = 0; i < bufferId; ++i)
            bufferOffset += sendHaloIndicesSizePerBuffer[haloId][i];

        size_t mpiSendBufferIndex = 0;
        for(auto const & region : sendHaloIndices[haloId][bufferId]) {

            const size_t &region_start = region[0];
            const size_t &region_howmanycols = region[1];
            const size_t &region_howmanyrows = region[2];
            const size_t &region_striderow = region[3];

            for(size_t rows = 0; rows < region_howmanyrows; ++rows) {

                cudaError_t err = cudaMemcpy(&sendBuffer[haloId][bufferOffset + mpiSendBufferIndex],
                                             &buf[region_start+rows*region_striderow],
                                             region_howmanycols*sizeof(unsigned char),
                                             cudaMemcpyDeviceToHost);

                if(err != cudaSuccess)
                    std::cout << "Tausch::packSendBufferCUDA(): CUDA error detected: " << cudaGetErrorString(err) << " (" << err << ")" << std::endl;

                mpiSendBufferIndex += region_howmanycols;

            }

        }

    }

    /***********************************************************************/
    /*                        UNPACK BUFFER (CUDA)                         */
    /***********************************************************************/

    void unpackRecvBufferCUDA(const size_t haloId, const size_t bufferId, double *buf) {
        unpackRecvBufferCUDA(haloId, bufferId, reinterpret_cast<unsigned char*>(buf));
    }

    void unpackRecvBufferCUDA(const size_t haloId, const size_t bufferId, int *buf) {
        unpackRecvBufferCUDA(haloId, bufferId, reinterpret_cast<unsigned char*>(buf));
    }

    void unpackRecvBufferCUDA(const size_t haloId, const size_t bufferId, unsigned char *buf) {

        size_t bufferOffset = 0;
        for(size_t i = 0; i < bufferId; ++i)
            bufferOffset += recvHaloIndicesSizePerBuffer[haloId][i];

        size_t mpiRecvBufferIndex = 0;

        for(auto const & region : recvHaloIndices[haloId][bufferId]) {

            const size_t &region_start = region[0];
            const size_t &region_howmanycols = region[1];
            const size_t &region_howmanyrows = region[2];
            const size_t &region_striderow = region[3];

            for(int rows = 0; rows < region_howmanyrows; ++rows) {

                cudaError_t err = cudaMemcpy(&buf[region_start+rows*region_striderow],
                                             &recvBuffer[haloId][bufferOffset + mpiRecvBufferIndex],
                                             region_howmanycols*sizeof(unsigned char),
                                             cudaMemcpyHostToDevice);
                if(err != cudaSuccess)
                    std::cout << "Tausch::unpackRecvBufferCUDA(): CUDA error detected: " << cudaGetErrorString(err) << " (" << err << ")" << std::endl;

                mpiRecvBufferIndex += region_howmanycols;

            }

        }

    }

#endif



    /***********************************************************************/
    /***********************************************************************/

    inline std::vector<std::array<int, 4> > extractHaloIndicesWithStride(std::vector<int> indices) {

        // nothing to do
        if(indices.size() == 0)
            return std::vector<std::array<int, 4> >();

        // first we build a collection of all consecutive rows
        std::vector<std::array<int, 2> > rows;

        int curIndex = 1;
        int start = indices[0];
        int howmany = 1;
        while(curIndex < indices.size()) {

            if(indices[curIndex]-indices[curIndex-1] == 1)
                ++howmany;
            else {

                rows.push_back({start, howmany});

                start = indices[curIndex];
                howmany = 1;

            }

            ++curIndex;

        }

        rows.push_back({start, howmany});

        // second we look for simple patterns within these rows
        std::vector<std::array<int, 4> > ret;

        ret.push_back({rows[0][0], rows[0][1], 1, 0});

        for(size_t currow = 1; currow < rows.size(); ++currow) {

            if(rows[currow][1] == ret.back()[1] && (ret.back()[3] == 0 || rows[currow][0]-(ret.back()[0]+(ret.back()[2]-1)*ret.back()[3]) == ret.back()[3])) {

                if(ret.back()[3] == 0) {
                    ++ret.back()[2];
                    ret.back()[3] = (rows[currow][0]-ret.back()[0]);
                } else
                    ++ret.back()[2];

            } else {

                ret.push_back({rows[currow][0], rows[currow][1], 1, 0});
            }

        }

        return ret;

    }

    inline std::vector<std::array<int, 4> > convertToUnsignedCharIndices(std::vector<std::array<int, 4> > indices, const size_t typeSize) {

        if(typeSize == 1)
            return indices;

        std::vector<std::array<int, 4> > ret;

        for(auto region : indices)
            ret.push_back({region[0]*static_cast<int>(typeSize),
                           region[1]*static_cast<int>(typeSize),
                           region[2],
                           region[3]*static_cast<int>(typeSize)});

        return ret;

    }

private:

    MPI_Comm TAUSCH_COMM;


    std::vector<std::vector<std::vector<std::array<int, 4> > > > sendHaloIndices;
    std::vector<std::vector<int> > sendHaloIndicesSizePerBuffer;
    std::vector<int> sendHaloIndicesSizeTotal;
    std::vector<int> sendHaloNumBuffers;
    std::vector<int> sendHaloRemoteRank;
    std::vector<std::unique_ptr<unsigned char[]> > sendBuffer;
    std::vector<std::vector<MPI_Request> > sendHaloMpiRequests;
    std::vector<std::vector<bool> > sendHaloMpiSetup;
    std::vector<Communication> sendHaloCommunicationStrategy;
    std::map<int, std::map<int, unsigned char*> > sendHaloBuffer;
    std::map<int, std::vector<MPI_Datatype> > sendHaloDerivedDatatype;

    std::vector<std::vector<std::vector<std::array<int, 4> > > > recvHaloIndices;
    std::vector<std::vector<int> > recvHaloIndicesSizePerBuffer;
    std::vector<int> recvHaloIndicesSizeTotal;
    std::vector<int> recvHaloNumBuffers;
    std::vector<int> recvHaloRemoteRank;
    std::vector<std::unique_ptr<unsigned char[]> > recvBuffer;
    std::vector<std::vector<MPI_Request> > recvHaloMpiRequests;
    std::vector<std::vector<bool> > recvHaloMpiSetup;
    std::vector<Communication> recvHaloCommunicationStrategy;
    std::map<int, std::map<int, unsigned char*> > recvHaloBuffer;
    std::map<int, std::vector<MPI_Datatype> > recvHaloDerivedDatatype;

    // this is used for exchanges on same mpi rank
    std::map<int, int> msgtagToHaloId;

};


#endif
