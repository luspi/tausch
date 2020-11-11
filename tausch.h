/**
 *
 * \copyright 2020 Lukas Spies. Licensed under the MIT license.
 *
 * \mainpage Tausch - A generic halo exchange library.
 *
 * \section intro_sec Introduction
 *
 * Tausch is a halo exchange library that is designed to be maximally flexible (can be used almost
 * anywhere), minimally instrusive (easy to adapt into existing code), maximally efficient
 * (performs just as good or better than any custom solution), and requires minimal memory
 * resources (compressed halo metadata).
 *
 * \section install_sec Install
 *
 * Tausch is a header-only library and thus is very easy to use, no linking or precompiling necessary.
 *
 * In order to use Tausch one only needs to include the single header file tausch.h. Using CMake the
 * tausch header file can be installed into your system location and thence can be included from
 * anywhere by `#include <tausch/tausch.h>`
 *
 * \section Usage
 *
 * Tausch is very straight-forward to use. A very simple example is shown below:
 *
 * @include examplecode.cpp
 *
 * The above code can be compiled with `mpic++ examplecode.cpp -o example`.
 *
 * Running the above example will lead to the following output:<br>
 * <code>$ ./example<br>
 *  Input buffer: 1 2 3 4 5<br>
 *  Output buffer: 1 0 2 0 3 0 4 0 5 0</code>
 *
 * \section support_sec Support
 *
 * Tausch is hosted on GitHub. To ask questions, report bugs, and for any other type of support
 * please open an issue there: https://github.com/luspi/tausch
 *
 */
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
#   include <cuda_runtime.h>
#endif

#ifdef TAUSCH_OPENCL
#   ifdef __has_include
#       if __has_include("CL/cl2.hpp")
#           define CL_HPP_ENABLE_EXCEPTIONS
#           define CL_HPP_TARGET_OPENCL_VERSION 120
#           define CL_HPP_MINIMUM_OPENCL_VERSION 120
#           define CL_HPP_ENABLE_SIZE_T_COMPATIBILITY
#           include <CL/cl2.hpp>
#       elif __has_include("CL/cl.hpp")
#           define __CL_ENABLE_EXCEPTIONS
#           include <CL/cl.hpp>
#       else
#           #define CL_TARGET_OPENCL_VERSION 120
#           include <CL/cl.h>
#       endif
#   else
#      define __CL_ENABLE_EXCEPTIONS
#      include <CL/cl.hpp>
#   endif
#endif

/**
 * @brief
 * The Tausch class object.
 *
 * All features of Tausch are encapsulated in this single class object.
 */
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
        Default = 1,
        TryDirectCopy = 2,
        DerivedMpiDatatype = 4,
        CUDAAwareMPI = 8,
        MPIPersistent = 16
    };

    /***********************************************************************/
    /*                           ADD LOCAL HALO                            */
    /***********************************************************************/

    /**
     * \overload
     *
     * Here the halo indices can be specified as integers of halo data locations in the buffer with the
     * halo indices used for all of the numBuffers buffers. All buffer have the same data type.
     */
    inline size_t addSendHaloInfo(std::vector<int> haloIndices,
                                  const size_t typeSize,
                                  const int numBuffers = 1,
                                  const int remoteMpiRank = -1) {
        std::vector<std::vector<std::array<int, 4> > > indices;
        std::vector<size_t> typeSizePerBuffer;
        for(int i = 0; i < numBuffers; ++i) {
            indices.push_back(extractHaloIndicesWithStride(haloIndices));
            typeSizePerBuffer.push_back(typeSize);
        }
        return addSendHaloInfo(indices, typeSizePerBuffer, remoteMpiRank);
    }

    /**
     * \overload
     *
     * Here the halo indices can be specified as integers of halo data locations in the buffer, with a
     * different list of halo indices per buffer. All buffer have the same data type.
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
     * \overload
     *
     * Here the same set of halo indices is used for all of the numBuffers buffers. All buffer have
     * the same data type.
     */
    inline size_t addSendHaloInfo(std::vector<std::array<int, 4> > haloIndices,
                                  const size_t typeSize,
                                  const int numBuffers = 1,
                                  const int remoteMpiRank = -1) {
        std::vector<size_t> typeSizePerBuffer;
        for(int i = 0; i < numBuffers; ++i)
            typeSizePerBuffer.push_back(typeSize);
        return addSendHaloInfo(haloIndices, typeSizePerBuffer, remoteMpiRank);
    }

    /**
     * \overload
     *
     * Here each buffer has the same data type.
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
     * \overload
     *
     * Here the halo indices can be specified as integers of halo data locations in the buffer with
     * the halo indices used for all of the buffers. All buffers can have their own data type.
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
     * \overload
     *
     * Here the halo indices can be specified as integers of halo data locations in the buffer,
     * with a different list of halo indices per buffer. All buffer can have their own data type.
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
     * \overload
     *
     * Here the same set of halo indices is used for all of the numBuffers buffers. All buffer can
     * have their own data type.
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
     * Add a new send-halo region.
     *
     * This adds a new send-halo region to the Tausch object.
     *
     * @param haloIndices
     * The halo indices are specified as vectors with two levels:<br>
     * 1) The first level corresponds to the buffer id. Multiple buffers can be combined into a
     * single message and each buffer can have their own halo specification.<br>
     * 2) The second level corresponds to a rectangular subregion of this halo specification, with
     * each rectangular subregion specified using four integers.
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
        sendHaloCommunicationStrategy.push_back(Communication::Default);
        sendHaloRemoteRank.push_back(remoteMpiRank);

        void *newbuf = NULL;
        posix_memalign(&newbuf, 64, totalHaloSize*sizeof(unsigned char));
        unsigned char *newbuf_buft = reinterpret_cast<unsigned char*>(newbuf);
        unsigned char zero = 0;
        std::fill_n(newbuf_buft, totalHaloSize, zero);
        sendBuffer.push_back(std::unique_ptr<unsigned char[]>(std::move(newbuf_buft)));

#ifdef TAUSCH_CUDA
        cudaSendBuffer.push_back(nullptr);
#endif

        std::vector<MPI_Request> perBufRequests;
        std::vector<bool> perBufSetup;
        for(size_t iBuf = 0; iBuf < haloSizePerBuffer.size(); ++iBuf) {
            perBufRequests.push_back(MPI_Request());
            perBufSetup.push_back(false);
        }
        sendHaloMpiRequests.push_back(perBufRequests);
        sendHaloMpiSetup.push_back(perBufSetup);

        return sendBuffer.size()-1;

    }

    /**
     * @brief
     * Delete a send-halo of a given halo id.
     *
     * This deletes a send-halo with the given halo id
     *
     * @param haloId
     * The halo id returned by the addSendHaloInfo() member function.
     */
    inline void delSendHaloInfo(size_t haloId) {
        sendBuffer[haloId].reset(nullptr);
    }

    /***********************************************************************/
    /*                          ADD REMOTE HALO                            */
    /***********************************************************************/

    /**
     * \overload
     *
     * Here the halo indices can be specified as integers of halo data locations in the buffer with
     * the halo indices used for all of the numBuffers buffers. All buffer have the same data type.
     */
    inline size_t addRecvHaloInfo(std::vector<int> haloIndices,
                                  const size_t typeSize,
                                  const int numBuffers = 1,
                                  const int remoteMpiRank = -1) {
        std::vector<std::vector<std::array<int, 4> > > indices;
        std::vector<size_t> typeSizePerBuffer;
        for(int i = 0; i < numBuffers; ++i) {
            indices.push_back(extractHaloIndicesWithStride(haloIndices));
            typeSizePerBuffer.push_back(typeSize);
        }
        return addRecvHaloInfo(indices, typeSizePerBuffer, remoteMpiRank);
    }

    /**
     * \overload
     *
     * Here the halo indices can be specified as integers of halo data locations in the buffer,
     * with a different list of halo indices per buffer. All buffer have the same data type.
     */
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

    /**
     * \overload
     *
     * Here the same set of halo indices is used for all of the numBuffers buffers. All buffer
     * have the same data type.
     */
    inline size_t addRecvHaloInfo(std::vector<std::array<int, 4> > haloIndices,
                                  const size_t typeSize,
                                  const int numBuffers = 1,
                                  const int remoteMpiRank = -1) {
        std::vector<std::vector<std::array<int, 4> > > indices;
        std::vector<size_t> typeSizePerBuffer;
        for(int i = 0; i < numBuffers; ++i) {
            indices.push_back(haloIndices);
            typeSizePerBuffer.push_back(typeSize);
        }
        return addRecvHaloInfo(indices, typeSizePerBuffer, remoteMpiRank);
    }

    /**
     * \overload
     *
     * Here each buffer has the same data type.
     */
    inline size_t addRecvHaloInfo(std::vector<std::vector<std::array<int, 4> > > haloIndices,
                                  const size_t typeSize,
                                  const int remoteMpiRank = -1) {
        std::vector<size_t> typeSizePerBuffer;
        for(size_t bufferId = 0; bufferId < haloIndices.size(); ++bufferId)
            typeSizePerBuffer.push_back(typeSize);
        return addRecvHaloInfo(haloIndices, typeSizePerBuffer, remoteMpiRank);
    }

    /**
     * \overload
     *
     * Here the halo indices can be specified as integers of halo data locations in the buffer with
     * the halo indices used for all of the buffers. All buffers can have their own data type.
     */
    inline size_t addRecvHaloInfo(std::vector<int> haloIndices,
                                  const std::vector<size_t> typeSizePerBuffer,
                                  const int remoteMpiRank = -1) {
        std::vector<std::vector<std::array<int, 4> > > indices;
        for(size_t i = 0; i < typeSizePerBuffer.size(); ++i)
            indices.push_back(extractHaloIndicesWithStride(haloIndices));
        return addRecvHaloInfo(indices, typeSizePerBuffer, remoteMpiRank);
    }

    /**
     * \overload
     *
     * Here the halo indices can be specified as integers of halo data locations in the buffer,
     * with a different list of halo indices per buffer. All buffer can have their own data type.
     */
    inline size_t addRecvHaloInfo(std::vector<std::vector<int> > haloIndices,
                                  const std::vector<size_t> typeSizePerBuffer,
                                  const int remoteMpiRank = -1) {
        std::vector<std::vector<std::array<int, 4> > > indices;
        for(size_t i = 0; i < typeSizePerBuffer.size(); ++i)
            indices.push_back(extractHaloIndicesWithStride(haloIndices[i]));
        return addRecvHaloInfo(indices, typeSizePerBuffer, remoteMpiRank);
    }

    /**
     * \overload
     *
     * Here the same set of halo indices is used for all of the numBuffers buffers. All buffer can
     * have their own data type.
     */
    inline size_t addRecvHaloInfo(std::vector<std::array<int, 4> > haloIndices,
                                  const std::vector<size_t> typeSizePerBuffer,
                                  const int remoteMpiRank = -1) {
        std::vector<std::vector<std::array<int, 4> > > indices;
        for(size_t i = 0; i < typeSizePerBuffer.size(); ++i)
            indices.push_back(haloIndices);
        return addRecvHaloInfo(indices, typeSizePerBuffer, remoteMpiRank);
    }

    /**
     * @brief
     * Add a new recv-halo region.
     *
     * This adds a new recv-halo region to the Tausch object.
     *
     * @param haloIndices
     * The halo indices are specified as vectors with two levels:<br>
     * 1) The first level corresponds to the buffer id. Multiple buffers can be combined into a
     * single message and each buffer can have their own halo specification.<br>
     * 2) The second level corresponds to a rectangular subregion of this halo specification, with
     * each rectangular subregion specified using four integers.
     * @param typeSizePerBuffer
     * A vector with the size of the data type for each buffer
     * @param remoteMpiRank
     * What remote MPI rank this/these halo region/s will be sent to.
     *
     * @return
     * This function returns the halo id (needed for referencing this halo later-on).
     */
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
        recvHaloCommunicationStrategy.push_back(Communication::Default);
        recvHaloRemoteRank.push_back(remoteMpiRank);

        void *newbuf = NULL;
        posix_memalign(&newbuf, 64, totalHaloSize*sizeof(unsigned char));
        unsigned char *newbuf_buft = reinterpret_cast<unsigned char*>(newbuf);
        unsigned char zero = 0;
        std::fill_n(newbuf_buft, totalHaloSize, zero);
        recvBuffer.push_back(std::unique_ptr<unsigned char[]>(std::move(newbuf_buft)));

#ifdef TAUSCH_CUDA
        cudaRecvBuffer.push_back(nullptr);
#endif

        std::vector<MPI_Request> perBufRequests;
        std::vector<bool> perBufSetup;
        for(size_t iBuf = 0; iBuf < haloSizePerBuffer.size(); ++iBuf) {
            perBufRequests.push_back(MPI_Request());
            perBufSetup.push_back(false);
        }
        recvHaloMpiRequests.push_back(perBufRequests);
        recvHaloMpiSetup.push_back(perBufSetup);

        return recvBuffer.size()-1;

    }

    /**
     * @brief
     * Delete a recv-halo of a given halo id.
     *
     * This deletes a recv-halo with the given halo id
     *
     * @param haloId
     * The halo id returned by the addRecvHaloInfo() member function.
     */
    inline void delRecvHaloInfo(size_t haloId) {
        recvBuffer[haloId].reset(nullptr);
    }

    /***********************************************************************/
    /*                      SET COMMUNICATION STRATEGY                     */
    /***********************************************************************/

    /**
     * @brief
     * Set a communication strategy for sending a halo.
     *
     * Set a communication strategy for sending a halo. The strategy can be any one of the
     * Communication enum.
     *
     * @param haloId
     * The halo id returned by the addSendHaloInfo() member function.
     * @param strategy
     * The strategy to use, can be any one of the Communication enum.
     */
    inline void setSendCommunicationStrategy(size_t haloId, Communication strategy) {

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

#ifdef TAUSCH_CUDA
        } else if((strategy&Communication::CUDAAwareMPI) == Communication::CUDAAwareMPI) {

            cudaMalloc(&cudaSendBuffer[haloId], sendHaloIndicesSizeTotal[haloId]*sizeof(unsigned char));
#endif

        }

    }

    /**
     * @brief
     * Set a communication strategy for receiving a halo.
     *
     * Set a communication strategy for receiving a halo. The strategy can be any one of the Communication enum.
     *
     * @param haloId
     * The halo id returned by the addRecvHaloInfo() member function.
     * @param strategy
     * The strategy to use, can be any one of the Communication enum.
     */
    inline void setRecvCommunicationStrategy(size_t haloId, Communication strategy) {

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

#ifdef TAUSCH_CUDA
        } else if((strategy&Communication::CUDAAwareMPI) == Communication::CUDAAwareMPI) {

            cudaMalloc(&cudaRecvBuffer[haloId], recvHaloIndicesSizeTotal[haloId]*sizeof(unsigned char));
#endif

        }

    }



    /***********************************************************************/
    /*                          SEND HALO BUFFER                           */
    /***********************************************************************/

    /**
     * \overload
     *
     * Internally the data buffer will be recast to unsigned char.
     */
    inline void setSendHaloBuffer(int haloId, int bufferId, int* buf) {
        setSendHaloBuffer(haloId, bufferId, reinterpret_cast<unsigned char*>(buf));
    }

    /**
     * \overload
     *
     * Internally the data buffer will be recast to unsigned char.
     */
    inline void setSendHaloBuffer(int haloId, int bufferId, double* buf) {
        setSendHaloBuffer(haloId, bufferId, reinterpret_cast<unsigned char*>(buf));
    }

    /**
     * @brief
     * Set a pointer to the sending halo data buffer.
     *
     * This sets a pointer to the sending halo data buffer of type unsigned char. This is a
     * necessary step for certain communication strategies, e.g., using MPI derived data types.
     *
     * @param haloId
     * The halo id returned by the addSendHaloInfo() member function.
     * @param bufferId
     * The id of the current buffer (numbered starting at 0).
     * @param buf
     * Pointer to the data buffer of type unsigned char.
     */
    inline void setSendHaloBuffer(int haloId, int bufferId, unsigned char* buf) {
        sendHaloBuffer[haloId][bufferId] = buf;
    }



    /***********************************************************************/
    /*                          RECV HALO BUFFER                           */
    /***********************************************************************/

    /**
     * \overload
     *
     * Internally the data buffer will be recast to unsigned char.
     */
    inline void setRecvHaloBuffer(int haloId, int bufferId, int* buf) {
        setRecvHaloBuffer(haloId, bufferId, reinterpret_cast<unsigned char*>(buf));
    }

    /**
     * \overload
     *
     * Internally the data buffer will be recast to unsigned char.
     */
    inline void setRecvHaloBuffer(int haloId, int bufferId, double* buf) {
        setRecvHaloBuffer(haloId, bufferId, reinterpret_cast<unsigned char*>(buf));
    }

    /**
     * @brief
     * Set a pointer to the receiving halo data buffer.
     *
     * This sets a pointer to the receiving halo data buffer of type int. This is a necessary step
     * for certain communication strategies, e.g., using MPI derived data types.
     *
     * @param haloId
     * The halo id returned by the addRecvHaloInfo() member function.
     * @param bufferId
     * The id of the current buffer (numbered starting at 0).
     * @param buf
     * Pointer to the data buffer of type unsigned char.
     */
    inline void setRecvHaloBuffer(int haloId, int bufferId, unsigned char* buf) {
        recvHaloBuffer[haloId][bufferId] = buf;
    }



    /***********************************************************************/
    /*                             PACK BUFFER                             */
    /***********************************************************************/

    /**
     * \overload
     *
     * Internally the data buffer will be recast to unsigned char.
     */
    inline void packSendBuffer(const size_t haloId, const size_t bufferId, const double *buf) {
        packSendBuffer(haloId, bufferId, reinterpret_cast<const unsigned char*>(buf));
    }

    /**
     * \overload
     *
     * Internally the data buffer will be recast to unsigned char.
     */
    inline void packSendBuffer(const size_t haloId, const size_t bufferId, const int *buf) {
        packSendBuffer(haloId, bufferId, reinterpret_cast<const unsigned char*>(buf));
    }

    /**
     * @brief
     * Packs a data buffer for the given halo and buffer id.
     *
     * Packs a data buffer for the given halo and buffer id. Once this function has been called the
     * data buffer is free to be used and changed as desired.
     *
     * @param haloId
     * The halo id returned by the addSendHaloInfo() member function.
     * @param bufferId
     * The id of the current buffer (numbered starting at 0).
     * @param buf
     * Pointer to the data buffer.
     */
    inline void packSendBuffer(const size_t haloId, const size_t bufferId, const unsigned char *buf) {

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

    /**
     * @brief
     * Send a given halo data off.
     *
     * Send a given halo data off.
     *
     * @param haloId
     * The halo id returned by the addSendHaloInfo() member function.
     * @param msgtag
     * The message tag to be used by this communication. For CPU-CPU communication this is the same
     * as an MPI tag.
     * @param remoteMpiRank
     * The receiving MPI rank.
     * @param bufferId
     * The id of the current buffer (numbered starting at 0).
     * @param blocking
     * If set to true Tausch will block until the send routine has been fully received. If set to
     * false this routine will return immediately and the data will be sent in the background.
     * @param communicator
     * If a communicator is specified here then Tausch will ignore the global communicator set
     * during construction.
     *
     * @return
     * Returns the MPI_Request used for this communication.
     */
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
        if(useRemoteMpiRank == myRank && (sendHaloCommunicationStrategy[haloId]&Communication::TryDirectCopy) == Communication::TryDirectCopy) {
            msgtagToHaloId[myRank*1000000 + msgtag] = haloId;
            return MPI_REQUEST_NULL;
        }

        int useBufferId = 0;
        if((sendHaloCommunicationStrategy[haloId]&Communication::DerivedMpiDatatype) == Communication::DerivedMpiDatatype)
            useBufferId = bufferId;

        // Take this path if we are to use MPI persistent communication
        // Depending on the implementation this can be a good or bad idea.
        // Using persistent communication could pin the communication protocol to eager (tends to bad performance for larger messages)
        if((sendHaloCommunicationStrategy[haloId]&Communication::MPIPersistent) == Communication::MPIPersistent) {

            if(!sendHaloMpiSetup[haloId][useBufferId]) {

                sendHaloMpiSetup[haloId][useBufferId] = true;

#ifdef TAUSCH_CUDA
                if((sendHaloCommunicationStrategy[haloId]&Communication::CUDAAwareMPI) == Communication::CUDAAwareMPI) {

                    MPI_Send_init(cudaSendBuffer[haloId], sendHaloIndicesSizeTotal[haloId], MPI_CHAR,
                                  useRemoteMpiRank, msgtag, communicator,
                                  &sendHaloMpiRequests[haloId][0]);

                } else
#endif
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

        // Take this path to use normal Isend/Irecv communication
        // This is the default.
        } else {

#ifdef TAUSCH_CUDA
            if((sendHaloCommunicationStrategy[haloId]&Communication::CUDAAwareMPI) == Communication::CUDAAwareMPI) {

                MPI_Isend(cudaSendBuffer[haloId], sendHaloIndicesSizeTotal[haloId], MPI_CHAR,
                          useRemoteMpiRank, msgtag, communicator,
                          &sendHaloMpiRequests[haloId][0]);

            } else
#endif

                if((sendHaloCommunicationStrategy[haloId]&Communication::DerivedMpiDatatype) == Communication::DerivedMpiDatatype)
                    MPI_Isend(sendHaloBuffer[haloId][useBufferId], 1, sendHaloDerivedDatatype[haloId][useBufferId],
                              useRemoteMpiRank, msgtag, communicator,
                              &sendHaloMpiRequests[haloId][useBufferId]);
                else
                    MPI_Isend(sendBuffer[haloId].get(), sendHaloIndicesSizeTotal[haloId], MPI_CHAR,
                              useRemoteMpiRank, msgtag, communicator,
                              &sendHaloMpiRequests[haloId][0]);

        }

        if(blocking)
            MPI_Wait(&sendHaloMpiRequests[haloId][useBufferId], MPI_STATUS_IGNORE);

        return sendHaloMpiRequests[haloId][useBufferId];

    }

    /***********************************************************************/
    /*                         RECEIVE MESSAGE                             */
    /***********************************************************************/

    /**
     * @brief
     * Receives a given halo.
     *
     * Receives a given halo.
     *
     * @param haloId
     * The halo id returned by the addRecvHaloInfo() member function.
     * @param msgtag
     * The message tag to be used by this communication. For CPU-CPU communication this is the same
     * as an MPI tag.
     * @param remoteMpiRank
     * The sending MPI rank.
     * @param bufferId
     * The id of the current buffer (numbered starting at 0).
     * @param blocking
     * If set to true Tausch will block until the receive routine has been fully completed. If set
     * to false this routine will return immediately and the data will be received in the
     * background (the MPI_Request can be tested to check for completion).
     * @param communicator
     * If a communicator is specified here then Tausch will ignore the global communicator set
     * during construction.
     *
     * @return
     * Returns the MPI_Request used for this communication.
     */
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
        if(useRemoteMpiRank == myRank && (recvHaloCommunicationStrategy[haloId]&Communication::TryDirectCopy) == Communication::TryDirectCopy) {
            const int remoteHaloId = msgtagToHaloId[myRank*1000000 + msgtag];
            std::memcpy(recvBuffer[haloId].get(), sendBuffer[remoteHaloId].get(), recvHaloIndicesSizeTotal[haloId]);
            return MPI_REQUEST_NULL;
        }

        int useBufferId = 0;
        if((recvHaloCommunicationStrategy[haloId]&Communication::DerivedMpiDatatype) == Communication::DerivedMpiDatatype)
            useBufferId = bufferId;

        // Take this path if we are to use MPI persistent communication
        // Depending on the implementation this can be a good or bad idea.
        // Using persistent communication could pin the communication protocol to eager (tends to bad performance for larger messages)
        if((recvHaloCommunicationStrategy[haloId]&Communication::MPIPersistent) == Communication::MPIPersistent) {

            if(!recvHaloMpiSetup[haloId][useBufferId]) {

                recvHaloMpiSetup[haloId][useBufferId] = true;

#ifdef TAUSCH_CUDA
                if((recvHaloCommunicationStrategy[haloId]&Communication::CUDAAwareMPI) == Communication::CUDAAwareMPI) {

                    MPI_Recv_init(cudaRecvBuffer[haloId], recvHaloIndicesSizeTotal[haloId], MPI_CHAR,
                                  useRemoteMpiRank, msgtag, communicator,
                                  &recvHaloMpiRequests[haloId][0]);

                } else
#endif
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

        // Take this path to use normal Isend/Irecv communication
        // This is the default.
        } else {

#ifdef TAUSCH_CUDA
            if((recvHaloCommunicationStrategy[haloId]&Communication::CUDAAwareMPI) == Communication::CUDAAwareMPI) {

                MPI_Irecv(cudaRecvBuffer[haloId], recvHaloIndicesSizeTotal[haloId], MPI_CHAR,
                          useRemoteMpiRank, msgtag, communicator,
                          &recvHaloMpiRequests[haloId][0]);

            } else
#endif
            if((recvHaloCommunicationStrategy[haloId]&Communication::DerivedMpiDatatype) == Communication::DerivedMpiDatatype)
                MPI_Irecv(recvHaloBuffer[haloId][useBufferId], 1, recvHaloDerivedDatatype[haloId][useBufferId],
                          useRemoteMpiRank, msgtag, communicator,
                          &recvHaloMpiRequests[haloId][useBufferId]);
            else
                MPI_Irecv(recvBuffer[haloId].get(), recvHaloIndicesSizeTotal[haloId], MPI_CHAR,
                          useRemoteMpiRank, msgtag, communicator,
                          &recvHaloMpiRequests[haloId][0]);

        }

        if(blocking)
            MPI_Wait(&recvHaloMpiRequests[haloId][useBufferId], MPI_STATUS_IGNORE);

        return recvHaloMpiRequests[haloId][useBufferId];

    }

    /***********************************************************************/
    /*                           UNPACK BUFFER                             */
    /***********************************************************************/

    /**
     * \overload
     *
     * Internally the data buffer will be recast to unsigned char.
     */
    inline void unpackRecvBuffer(const size_t haloId, const size_t bufferId, double *buf) {
        unpackRecvBuffer(haloId, bufferId, reinterpret_cast<unsigned char*>(buf));
    }

    /**
     * \overload
     *
     * Internally the data buffer will be recast to unsigned char.
     */
    inline void unpackRecvBuffer(const size_t haloId, const size_t bufferId, int *buf) {
        unpackRecvBuffer(haloId, bufferId, reinterpret_cast<unsigned char*>(buf));
    }

    /**
     * @brief
     * Unpacks a data buffer for the given halo and buffer id.
     *
     * Unpacks a data buffer for the given halo and buffer id. Once this function has been called
     * the received data can be immediately used.
     *
     * @param haloId
     * The halo id returned by the addRecvHaloInfo() member function.
     * @param bufferId
     * The id of the current buffer (numbered starting at 0).
     * @param buf
     * Pointer to the data buffer.
     */
    inline void unpackRecvBuffer(const size_t haloId, const size_t bufferId, unsigned char *buf) {

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

#ifdef TAUSCH_OPENCL

    /** @name OpenCL
     * The OpenCL routines. In order to use these the macro TAUSCH_OPENCL needs to be defined before including the tausch header.
     */
    /**@{*/

    /***********************************************************************/
    /*                          SET OPENCL SETUP                           */
    /***********************************************************************/

    /**
     * @brief
     * Tells Tausch about an existing OpenCL environment.
     *
     * Tells Tausch about an existing OpenCL environment. The environment needs to be set up by the user before calling this function.
     *
     * @param ocl_device
     * The OpenCL device.
     * @param ocl_context
     * The OpenCL context.
     * @param ocl_queue
     * The OpenCL CommandQueue.
     */
    inline void setOpenCL(cl::Device ocl_device, cl::Context ocl_context, cl::CommandQueue ocl_queue) {
        this->ocl_device = ocl_device;
        this->ocl_context = ocl_context;
        this->ocl_queue = ocl_queue;
    }

    /**
     * @brief
     * Tells Tausch to set up an OpenCL environment.
     *
     * Tells Tausch to set up an OpenCL environment. Tausch will use this environment for all subsequent OpenCL operations.
     *
     * @param deviceNumber
     * Choose a device number.
     *
     * @return
     * String containing the name of the chosen OpenCL device.
     */
    inline std::string enableOpenCL(size_t deviceNumber = 0) {
        try {

            std::vector<cl::Platform> all_platforms;
            cl::Platform::get(&all_platforms);
            cl::Platform ocl_platform = all_platforms[0];

            std::vector<cl::Device> all_devices;
            ocl_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
            if(deviceNumber >= all_devices.size())
                deviceNumber = 0;
            this->ocl_device = all_devices[deviceNumber];

            std::string deviceName = ocl_device.getInfo<CL_DEVICE_NAME>();

            // Create context and queue
            this->ocl_context = cl::Context({ocl_device});
            this->ocl_queue = cl::CommandQueue(ocl_context,ocl_device);

            return deviceName;

        } catch(cl::Error &error) {
            std::cout << "[enableOpenCL] OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
            exit(1);
        }

        return "";

    }

    /**
     * @brief
     * Get OpenCL device.
     *
     * Get OpenCL device.
     *
     * @return
     * OpenCL device in use by Tausch.
     */
    inline cl::Device getOclDevice() {
        return ocl_device;
    }

    /**
     * @brief
     * Get OpenCL context.
     *
     * Get OpenCL context.
     *
     * @return
     * OpenCL context in use by Tausch.
     */
    inline cl::Context getOclContext() {
        return ocl_context;
    }

    /**
     * @brief
     * Get OpenCL command queue.
     *
     * Get OpenCL command queue.
     *
     * @return
     * OpenCL command queue in use by Tausch.
     */
    inline cl::CommandQueue getOclQueue() {
        return ocl_queue;
    }

    /**
     * @brief
     * Packs an OpenCL buffer for the given halo and buffer id.
     *
     * Packs an OpenCL buffer for the given halo and buffer id. Once this function has
     * been called the OpenCL buffer is free to be used and changed as desired.
     *
     * @param haloId
     * The halo id returned by the addSendHaloInfo() member function.
     * @param bufferId
     * The id of the current buffer (numbered starting at 0).
     * @param buf
     * Handler of the OpenCL buffer.
     */
    inline void packSendBufferOCL(const size_t haloId, const size_t bufferId, cl::Buffer buf) {

        size_t bufferOffset = 0;
        for(size_t i = 0; i < bufferId; ++i)
            bufferOffset += sendHaloIndicesSizePerBuffer[haloId][i];

        size_t mpiSendBufferIndex = 0;
        for(auto const & region : sendHaloIndices[haloId][bufferId]) {

            const size_t &region_start = region[0];
            const size_t &region_howmanycols = region[1];
            const size_t &region_howmanyrows = region[2];
            const size_t &region_striderow = region[3];

            cl::size_t<3> buffer_offset;
            buffer_offset[0] = region_start; // dest starting index in bytes
            buffer_offset[1] = 0; buffer_offset[2] = 0;
            cl::size_t<3> host_offset;
            host_offset[0] = bufferOffset+mpiSendBufferIndex; // host starting index in bytes
            host_offset[1] = 0; host_offset[2] = 0;

            cl::size_t<3> reg;
            reg[0] = region_howmanycols; // how many bytes in one row
            reg[1] = region_howmanyrows; // how many rows
            reg[2] = 1; // leave at 1

            ocl_queue.enqueueReadBufferRect(buf, true, buffer_offset, host_offset, reg,
                                             region_striderow, // dest stride
                                             0,
                                             region_howmanycols,  // host stride
                                             0,
                                             sendBuffer[haloId].get());


            mpiSendBufferIndex += region_howmanyrows*region_howmanycols;

        }

    }

    /**
     * @brief
     * Unpacks an OpenCL buffer for the given halo and buffer id.
     *
     * Unpacks an OpenCL buffer for the given halo and buffer id. Once this function has
     * been called the received data can be immediately used.
     *
     * @param haloId
     * The halo id returned by the addRecvHaloInfo() member function.
     * @param bufferId
     * The id of the current buffer (numbered starting at 0).
     * @param buf
     * Handler of the OpenCL buffer.
     */
    inline void unpackRecvBufferOCL(const size_t haloId, const size_t bufferId, cl::Buffer buf) {

        size_t bufferOffset = 0;
        for(size_t i = 0; i < bufferId; ++i)
            bufferOffset += recvHaloIndicesSizePerBuffer[haloId][i];

        size_t mpiRecvBufferIndex = 0;

        for(auto const & region : recvHaloIndices[haloId][bufferId]) {

            const size_t &region_start = region[0];
            const size_t &region_howmanycols = region[1];
            const size_t &region_howmanyrows = region[2];
            const size_t &region_striderow = region[3];

            cl::size_t<3> buffer_offset;
            buffer_offset[0] = region_start; // dest starting index in bytes
            buffer_offset[1] = 0; buffer_offset[2] = 0;
            cl::size_t<3> host_offset;
            host_offset[0] = bufferOffset+mpiRecvBufferIndex; // host starting index in bytes
            host_offset[1] = 0; host_offset[2] = 0;

            cl::size_t<3> reg;
            reg[0] = region_howmanycols; // how many bytes in one row
            reg[1] = region_howmanyrows; // how many rows
            reg[2] = 1; // leave at 1

            ocl_queue.enqueueWriteBufferRect(buf, true, buffer_offset, host_offset, reg,
                                             region_striderow, // dest stride
                                             0,
                                             region_howmanycols,  // host stride
                                             0,
                                             recvBuffer[haloId].get());


            mpiRecvBufferIndex += region_howmanyrows*region_howmanycols;

        }

    }

    /**@}*/

#endif

    /***********************************************************************/
    /***********************************************************************/

#ifdef TAUSCH_CUDA

    ~Tausch() {

        for(unsigned char* ele : cudaSendBuffer) {
            cudaError_t err = cudaFree(ele);
            if(err != cudaSuccess)
                std::cout << "Tausch::~Tausch(): Error when freeing send CUDA memory: " << cudaGetErrorString(err) << " (" << err << ")" << std::endl;
        }
        for(unsigned char* ele : cudaRecvBuffer) {
            cudaError_t err = cudaFree(ele);
            if(err != cudaSuccess)
                std::cout << "Tausch::~Tausch(): Error when freeing recv CUDA memory: " << cudaGetErrorString(err) << " (" << err << ")" << std::endl;
        }

    }

    /** @name CUDA
     * The CUDA routines. In order to use these the macro TAUSCH_CUDA needs to be defined before including the tausch header.
     */
    /**@{*/

    /***********************************************************************/
    /*                         PACK BUFFER (CUDA)                          */
    /***********************************************************************/

    /**
     * \overload
     *
     * Internally the CUDA buffer will be recast to unsigned char.
     */
    inline void packSendBufferCUDA(const size_t haloId, const size_t bufferId, const double *buf) {
        packSendBufferCUDA(haloId, bufferId, reinterpret_cast<const unsigned char*>(buf));
    }

    /**
     * \overload
     *
     * Internally the CUDA buffer will be recast to unsigned char.
     */
    inline void packSendBufferCUDA(const size_t haloId, const size_t bufferId, const int *buf) {
        packSendBufferCUDA(haloId, bufferId, reinterpret_cast<const unsigned char*>(buf));
    }

    /**
     * @brief
     * Packs a CUDA buffer for the given halo and buffer id.
     *
     * Packs a CUDA buffer for the given halo and buffer id. Once this function has been called the
     * CUDA buffer is free to be used and changed as desired.
     *
     * @param haloId
     * The halo id returned by the addSendHaloInfo() member function.
     * @param bufferId
     * The id of the current buffer (numbered starting at 0).
     * @param buf
     * Pointer to the CUDA buffer.
     */
    inline void packSendBufferCUDA(const size_t haloId, const size_t bufferId, const unsigned char *buf) {

        size_t bufferOffset = 0;
        for(size_t i = 0; i < bufferId; ++i)
            bufferOffset += sendHaloIndicesSizePerBuffer[haloId][i];

        if((sendHaloCommunicationStrategy[haloId]&Communication::CUDAAwareMPI) == Communication::CUDAAwareMPI) {

            size_t mpiSendBufferIndex = 0;
            for(auto const & region : sendHaloIndices[haloId][bufferId]) {

                const size_t &region_start = region[0];
                const size_t &region_howmanycols = region[1];
                const size_t &region_howmanyrows = region[2];
                const size_t &region_striderow = region[3];

                cudaError_t err = cudaMemcpy2D(&cudaSendBuffer[haloId][bufferOffset + mpiSendBufferIndex], region_howmanycols*sizeof(unsigned char),
                                               &buf[region_start], region_striderow*sizeof(unsigned char),
                                               region_howmanycols*sizeof(unsigned char), region_howmanyrows,
                                               cudaMemcpyDeviceToDevice);

                mpiSendBufferIndex += region_howmanyrows*region_howmanycols;

                if(err != cudaSuccess)
                    std::cout << "Tausch::packSendBufferCUDA(): CUDA error detected: " << cudaGetErrorString(err) << " (" << err << ")" << std::endl;

            }

        } else {

            size_t mpiSendBufferIndex = 0;
            for(auto const & region : sendHaloIndices[haloId][bufferId]) {

                const size_t &region_start = region[0];
                const size_t &region_howmanycols = region[1];
                const size_t &region_howmanyrows = region[2];
                const size_t &region_striderow = region[3];

                cudaError_t err = cudaMemcpy2D(&sendBuffer[haloId][bufferOffset + mpiSendBufferIndex], region_howmanycols*sizeof(unsigned char),
                                               &buf[region_start], region_striderow*sizeof(unsigned char),
                                               region_howmanycols*sizeof(unsigned char), region_howmanyrows,
                                               cudaMemcpyDeviceToHost);

                mpiSendBufferIndex += region_howmanyrows*region_howmanycols;

                if(err != cudaSuccess)
                    std::cout << "Tausch::packSendBufferCUDA(): CUDA error detected: " << cudaGetErrorString(err) << " (" << err << ")" << std::endl;

            }

        }

    }

    /***********************************************************************/
    /*                        UNPACK BUFFER (CUDA)                         */
    /***********************************************************************/

    /**
     * \overload
     *
     * Internally the data buffer will be recast to unsigned char.
     */
    inline void unpackRecvBufferCUDA(const size_t haloId, const size_t bufferId, double *buf) {
        unpackRecvBufferCUDA(haloId, bufferId, reinterpret_cast<unsigned char*>(buf));
    }

    /**
     * \overload
     *
     * Internally the data buffer will be recast to unsigned char.
     */
    inline void unpackRecvBufferCUDA(const size_t haloId, const size_t bufferId, int *buf) {
        unpackRecvBufferCUDA(haloId, bufferId, reinterpret_cast<unsigned char*>(buf));
    }

    /**
     * @brief
     * Unpacks a CUDA buffer for the given halo and buffer id.
     *
     * Unpacks a CUDA buffer for the given halo and buffer id. Once this function has been called
     * the received data can be immediately used.
     *
     * @param haloId
     * The halo id returned by the addRecvHaloInfo() member function.
     * @param bufferId
     * The id of the current buffer (numbered starting at 0).
     * @param buf
     * Pointer to the CUDA buffer.
     */
    inline void unpackRecvBufferCUDA(const size_t haloId, const size_t bufferId, unsigned char *buf) {

        size_t bufferOffset = 0;
        for(size_t i = 0; i < bufferId; ++i)
            bufferOffset += recvHaloIndicesSizePerBuffer[haloId][i];

        if((recvHaloCommunicationStrategy[haloId]&Communication::CUDAAwareMPI) == Communication::CUDAAwareMPI) {

            size_t mpiRecvBufferIndex = 0;
            for(auto const & region : recvHaloIndices[haloId][bufferId]) {

                const size_t &region_start = region[0];
                const size_t &region_howmanycols = region[1];
                const size_t &region_howmanyrows = region[2];
                const size_t &region_striderow = region[3];

                cudaError_t err = cudaMemcpy2D(&buf[region_start], region_striderow*sizeof(unsigned char),
                                               &cudaRecvBuffer[haloId][bufferOffset + mpiRecvBufferIndex], region_howmanycols*sizeof(unsigned char),
                                               region_howmanycols*sizeof(unsigned char), region_howmanyrows,
                                               cudaMemcpyDeviceToDevice);

                if(err != cudaSuccess)
                    std::cout << "Tausch::unpackRecvBufferCUDA(): CUDA error detected: " << cudaGetErrorString(err) << " (" << err << ")" << std::endl;

                mpiRecvBufferIndex += region_howmanyrows*region_howmanycols;

            }

        } else {

            size_t mpiRecvBufferIndex = 0;
            for(auto const & region : recvHaloIndices[haloId][bufferId]) {

                const size_t &region_start = region[0];
                const size_t &region_howmanycols = region[1];
                const size_t &region_howmanyrows = region[2];
                const size_t &region_striderow = region[3];

                cudaError_t err = cudaMemcpy2D(&buf[region_start], region_striderow*sizeof(unsigned char),
                                               &recvBuffer[haloId][bufferOffset + mpiRecvBufferIndex], region_howmanycols*sizeof(unsigned char),
                                               region_howmanycols*sizeof(unsigned char), region_howmanyrows,
                                               cudaMemcpyHostToDevice);

                if(err != cudaSuccess)
                    std::cout << "Tausch::unpackRecvBufferCUDA(): CUDA error detected: " << cudaGetErrorString(err) << " (" << err << ")" << std::endl;

                mpiRecvBufferIndex += region_howmanyrows*region_howmanycols;

            }

        }

    }

    /**@}*/

#endif



    /***********************************************************************/
    /***********************************************************************/

    /**
     * @brief
     * Converts a list of halo indices to rectangular subregions.
     *
     * Converts a list of halo indices to rectangular subregions. Each subregion is represented by
     * four integers.
     *
     * @param indices
     * A list of indices specifying the location of halo data.
     *
     * @return
     * Returns the encoded halo information.
     */
    inline std::vector<std::array<int, 4> > extractHaloIndicesWithStride(std::vector<int> indices) {

        // nothing to do
        if(indices.size() == 0)
            return std::vector<std::array<int, 4> >();

        // first we build a collection of all consecutive rows
        std::vector<std::array<int, 2> > rows;

        size_t curIndex = 1;
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

    /**
     * @brief
     * Converts indices based on a certain data type into indices for unsigned char.
     *
     * Converts indices based on a certain data type into indices for unsigned char.
     *
     * @param indices
     * The halo indices based on given data type.
     * @param typeSize
     * The size of the data type.
     *
     * @return
     * Returns a list of indices based on unsigned char.
     */
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

#ifdef TAUSCH_CUDA
    std::vector<unsigned char*> cudaSendBuffer;
    std::vector<unsigned char*> cudaRecvBuffer;
#endif

#ifdef TAUSCH_OPENCL

    cl::Device ocl_device;
    cl::Context ocl_context;
    cl::CommandQueue ocl_queue;

#endif

};


#endif
