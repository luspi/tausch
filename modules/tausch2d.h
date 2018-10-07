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

#include "tauschdefs.h"
#include <mpi.h>
#include <iostream>
#include <fstream>
#ifdef TAUSCH_OPENCL
#include <atomic>
#endif
#include <vector>
#include <algorithm>
#include <cstring>

#ifdef TAUSCH_OPENCL
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#endif

/*!
 *
 * \brief
 *  A library providing a clean and efficient interface for halo exchange in two dimensions.
 *
 * %Tausch2D is a library that provides a clean and efficient C and C++ API for halo exchange for two dimensional domains. It doesn't assume
 * anything about the grid, except that the data is stored in one contiguous buffer. After specifying the local and remote halo regions, it takes care
 * of extracting the data for each halo from as many buffers as there are, sends the data for each halo combined into a single message each, and
 * unpacks the received data into the same number of buffers again. This is a template class and can be used with any of the common
 * C++ data types
 */
template <class buf_t>
class Tausch2D {

public:

    /*!
     *
     * The constructor, initiating the 2D Tausch object.
     *
     * \param mpiDataType
     *  The MPI_Datatype corresponding to the data type used for the template.
     * \param numBuffers
     *  The number of buffers that will be used. If more than one, they are all combined into one message for halo exchanges. Default value: 1
     * \param valuesPerPointPerBuffer
     *  How many values are stored consecutively per point in the same buffer. Each buffer can have different number of values stored per point. This
     *  is expected to be an array of the same size as the number of buffers. If set to NULL, all buffers are assumed to store 1 value per point.
     * \param comm
     *  The MPI Communictor to be used. %Tausch1D will duplicate the communicator, thus it is safe to have multiple instances of %Tausch1D working
     *  with the same communicator. By default, MPI_COMM_WORLD will be used.
     *
     */
    Tausch2D(MPI_Datatype mpiDataType, size_t numBuffers = 1, size_t *valuesPerPointPerBuffer = NULL, MPI_Comm comm = MPI_COMM_WORLD);

    /*!
     * The destructor cleaning up all memory.
     */
    ~Tausch2D();

    /*!
     *
     * Set the info about all local halos that need to be sent to remote MPI ranks.
     *
     * Function for CPU-CPU communication. The equivalent functions for CPU/GPU communication vary only in the ending of the name. Possible variants
     * are GwC, CwG, and GwG.
     *
     * \param numHaloParts
     *  How many different parts there are to the halo.
     * \param haloSpecs
     *  The specification of the different halo parts. This is done using an array of the struct TauschHaloSpec, with each instance in the array
     *  containing the specification of one halo region. %Tausch2D expects the following variables to be set for each region:
     *  variable | description
     *  :-------: | -------
     *   haloX | The starting x coordinate of the halo region
     *   haloY | The starting y coordinate of the halo region
     *   haloWidth | The width of the halo region
     *   haloHeight | The height of the halo region
     *   bufferWidth | The width of the underlying buffer
     *   bufferHeight | The height of the underlying buffer
     *   remoteMpiRank | The receiving processor
     * %Tausch2D internally copies all the data out of the haloSpecs (and sets up some buffers along the way), so it is safe to delete this array
     * after calling this function.
     *
     */
    size_t addLocalHaloInfoCwC(TauschHaloSpec haloSpec);
#ifdef TAUSCH_OPENCL /*! \cond DoxygenHideThis */
    void setLocalHaloInfoCwG(size_t numHaloParts, TauschHaloSpec *haloSpecs);
    void setLocalHaloInfoGwC(size_t numHaloParts, TauschHaloSpec *haloSpecs);
    void setLocalHaloInfoGwG(size_t numHaloParts, TauschHaloSpec *haloSpecs);
#endif /*! \endcond */

    /*!
     *
     * Frees up the memory associated with the given haloId. This does not remove the entry completely from the vector and thus doesn't change any
     * other haloId's!
     *
     * Function for CPU-CPU communication. The equivalent functions for CPU/GPU communication vary only in the ending of the name. Possible variants
     * are GwC, CwG, and GwG.
     *
     * \param haloId
     *  The id of the halo region that is to be deleted.
     *
     */
    void delLocalHaloInfoCwC(size_t haloId);

    /*!
     *
     * Set the info about all remote halos that are needed by this MPI rank.
     *
     * Function for CPU-CPU communication. The equivalent functions for CPU/GPU communication vary only in the ending of the name. Possible variants
     * are GwC, CwG, and GwG.
     *
     * \param numHaloParts
     *  How many different parts there are to the halo.
     * \param haloSpecs
     *  The specification of the different halo parts. This is done using an array of the struct TauschHaloSpec, with each instance in the array
     *  containing the specification of one halo region. %Tausch2D expects the following variables to be set for each region:
     *  variable | description
     *  :-------: | -------
     *   haloX | The starting x coordinate of the halo region
     *   haloY | The starting y coordinate of the halo region
     *   haloWidth | The width of the halo region
     *   haloHeight | The height of the halo region
     *   bufferWidth | The width of the underlying buffer
     *   bufferHeight | The height of the underlying buffer
     *   remoteMpiRank | The sending processor
     * %Tausch2D internally copies all the data out of the haloSpecs (and sets up some buffers along the way), so it is safe to delete this array
     * after calling this function.
     *
     * \return
     *  The haloId of the created halo region.
     *
     */
    size_t addRemoteHaloInfoCwC(TauschHaloSpec haloSpec);
#ifdef TAUSCH_OPENCL /*! \cond DoxygenHideThis */
    void setRemoteHaloInfoCwG(size_t numHaloParts, TauschHaloSpec *haloSpecs);
    void setRemoteHaloInfoGwC(size_t numHaloParts, TauschHaloSpec *haloSpecs);
    void setRemoteHaloInfoGwG(size_t numHaloParts, TauschHaloSpec *haloSpecs);
#endif /*! \endcond */

    /*!
     *
     * Frees up the memory associated with the given haloId. This does not remove the entry completely from the vector and thus doesn't change any
     * other haloId's!
     *
     * Function for CPU-CPU communication. The equivalent functions for CPU/GPU communication vary only in the ending of the name. Possible variants
     * are GwC, CwG, and GwG.
     *
     * \param haloId
     *  The id of the halo region that is to be deleted.
     *
     */
    void delRemoteHaloInfoCwC(size_t haloId);

    /*!
     *
     * Post the receive for the specified remote halo region of the current MPI rank.
     *
     * Function for CPU-CPU communication. The equivalent functions for CPU/GPU communication vary only in the ending of the name. Possible variants
     * are GwC, CwG, and GwG.
     *
     * \param haloId
     *  The id of the halo region. This is the index of this halo region in the local halo specification provided with setRemoteHaloInfo().
     * \param msgtag
     *  The message tag to be used for this receive. This information only has to be specified the first time the receive for the halo region with
     *  the specified id is posted. Each subsequent call, the msgtag that was passed the very first call will be re-used. This works equivalently to
     *  an MPI tag (and is, in fact, identical to it for MPI communication).
     *
     */
    void postReceiveCwC(size_t haloId, int msgtag = -1, int remoteMpiRank = -1);
#ifdef TAUSCH_OPENCL /*! \cond DoxygenHideThis */
    void postReceiveCwG(size_t haloId, int msgtag = -1);
    void postReceiveGwC(size_t haloId, int msgtag = -1);
    void postReceiveGwG(size_t haloId, int msgtag = -1);
#endif /*! \endcond */

    /*!
     *
     * Post all receives for the current MPI rank. This doesn't do anything else but post the MPI_Recv for each remote halo region.
     *
     * Function for CPU-CPU communication. The equivalent functions for CPU/GPU communication vary only in the ending of the name. Possible variants
     * are GwC, CwG, and GwG.
     *
     * \param msgtag
     *  An array containing the message tags, one for each halo region. This information only has to be specified the first time the receives are
     *  posted. Each subsequent call, the message tags that were passed the very first call will be re-used. This works equivalently to
     *  an MPI tag (and is, in fact, identical to it for MPI communication).
     *
     */
    void postAllReceivesCwC(int *msgtag = NULL);
#ifdef TAUSCH_OPENCL /*! \cond DoxygenHideThis */
    void postAllReceivesCwG(int *msgtag = NULL);
    void postAllReceivesGwC(int *msgtag = NULL);
    void postAllReceivesGwG(int *msgtag = NULL);
#endif /*! \endcond */

    /*!
     *
     * This packs the specified region of the specified halo area of the specified buffer for a send.
     *
     * Function for CPU-CPU communication. The equivalent functions for CPU/GPU communication vary only in the ending of the name. Possible variants
     * are GwC, CwG, and GwG.
     *
     * \param haloId
     *  The id of the halo region. This is the index of this halo region in the local halo specification provided with setLocalHaloInfo().
     * \param bufferId
     *  The id of the buffer. The order of the buffers will be preserved, i.e., packing buffer with id 1 required unpacking that buffer with id 1.
     *  The numbering of the buffers has to start with 0!
     * \param buf
     *  The buffer from which the data is to be extracted according to the local halo specification.
     * \param region
     *  Specification of the area of the current halo that is to be packed. This is specified relativce to the current halo, i.e., (x,y) = (0,0) is
     *  the bottom left corner of the halo region. %Tausch2D expects the following variables to be set:
     *  variable | description
     *  :-------: | -------
     *   x | The starting x coordinate of the region to be packed
     *   y | The starting y coordinate of the region to be packed
     *   width | The width of the region to be packed
     *   height | The height of the region to be packed
     *
     */
    void packSendBufferCwC(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region);
    /*!
     *
     * Overloaded function, taking the full halo region for packing.
     *
     * Function for CPU-CPU communication. The equivalent functions for CPU/GPU communication vary only in the ending of the name. Possible variants
     * are GwC, CwG, and GwG.
     *
     * \param haloId
     *  The id of the halo region. This is the index of this halo region in the local halo specification provided with setLocalHaloInfo().
     * \param bufferId
     *  The id of the buffer. The order of the buffers will be preserved, i.e., packing buffer with id 1 required unpacking that buffer with id 1.
     *  The numbering of the buffers has to start with 0!
     * \param buf
     *  The buffer from which the data is to be extracted according to the local halo specification.
     *
     */
    void packSendBufferCwC(size_t haloId, size_t bufferId, buf_t *buf);
#ifdef TAUSCH_OPENCL /*! \cond DoxygenHideThis */
    void packSendBufferCwG(size_t haloId, size_t bufferId, buf_t *buf);
    void packSendBufferCwG(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region);
    void packSendBufferGwC(size_t haloId, size_t bufferId, cl::Buffer buf);
    void packSendBufferGwG(size_t haloId, size_t bufferId, cl::Buffer buf);
#endif /*! \endcond */

    /*!
     *
     * Sends off the send buffer for the specified halo region. This starts the respective MPI_Send.
     *
     * Function for CPU-CPU communication. The equivalent functions for CPU/GPU communication vary only in the ending of the name. Possible variants
     * are GwC, CwG, and GwG.
     *
     * \param haloId
     *  The id of the halo region. This is the index of this halo region in the local halo specification provided with setLocalHaloInfo().
     * \param msgtag
     *  The message tag to be used for this send. This information only has to be specified the first time the send for the halo region with
     *  the specified id is started. Each subsequent call, the message tag that was passed the very first call will be re-used. This works
     *  equivalently to an MPI tag (and is, in fact, identical to it for MPI communication).
     * \param remoteMpiRank
     *  Where to send it to. If this is set to -1, then Tausch will take the value stored in the respective halo spec.
     *
     */
    void sendCwC(size_t haloId, int msgtag = -1, int remoteMpiRank = -1, MPI_Comm communicator = MPI_COMM_WORLD);
#ifdef TAUSCH_OPENCL /*! \cond DoxygenHideThis */
    void sendCwG(size_t haloId, int msgtag);
    void sendGwC(size_t haloId, int msgtag);
    void sendGwG(size_t haloId, int msgtag);
#endif /*! \endcond */

    /*!
     *
     * Makes sure the MPI message for the specified halo is received by this buffer. It does not do anything with that message!
     *
     * Function for CPU-CPU communication. The equivalent functions for CPU/GPU communication vary only in the ending of the name. Possible variants
     * are GwC, CwG, and GwG.
     *
     * \param haloId
     *  The id of the halo region. This is the index of this halo region in the remote halo specification provided with setRemoteHaloInfo().
     *
     */
    void recvCwC(size_t haloId);
    void recvAllCwC();
#ifdef TAUSCH_OPENCL /*! \cond DoxygenHideThis */
    void recvCwG(size_t haloId);
    void recvGwC(size_t haloId);
    void recvGwG(size_t haloId);
#endif /*! \endcond */

    /*!
     *
     * This unpacks the halo with the specified id from the received message into the provided buffer.
     *
     * Function for CPU-CPU communication. The equivalent functions for CPU/GPU communication vary only in the ending of the name. Possible variants
     * are GwC, CwG, and GwG.
     *
     * \param haloId
     *  The id of the halo region. This is the index of this halo region in the remote halo specification provided with setRemoteHaloInfo().
     * \param bufferId
     *  The id of the buffer. The order of the buffers will be preserved, i.e., packing buffer with id 1 required unpacking that buffer with id 1.
     *  The numbering of the buffers has to start with 0!
     * \param[out] buf
     *  The buffer to which the extracted data is to be written to according to the remote halo specification
     * \param region
     *  Specification of the area of the current halo that is to be packed. This is specified relativce to the current halo, i.e., (x,y) = (0,0) is
     *  the bottom left corner of the halo region. %Tausch2D expects the following variables to be set:
     *  variable | description
     *  :-------: | -------
     *   x | The starting x coordinate of the region to be packed
     *   y | The starting y coordinate of the region to be packed
     *   width | The width of the region to be packed
     *   height | The height of the region to be packed
     *
     */
    void unpackRecvBufferCwC(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region);
    /*!
     *
     * Overloaded function, taking the full halo region for unpacking.
     *
     * Function for CPU-CPU communication. The equivalent functions for CPU/GPU communication vary only in the ending of the name. Possible variants
     * are GwC, CwG, and GwG.
     *
     * \param haloId
     *  The id of the halo region. This is the index of this halo region in the remote halo specification provided with setRemoteHaloInfo().
     * \param bufferId
     *  The id of the buffer. The order of the buffers will be preserved, i.e., packing buffer with id 1 required unpacking that buffer with id 1.
     *  The numbering of the buffers has to start with 0!
     * \param[out] buf
     *  The buffer to which the extracted data is to be written to according to the remote halo specification
     *
     */
    void unpackRecvBufferCwC(size_t haloId, size_t bufferId, buf_t *buf);
#ifdef TAUSCH_OPENCL /*! \cond DoxygenHideThis */
    void unpackRecvBufferCwG(size_t haloId, size_t bufferId, buf_t *buf);
    void unpackRecvBufferCwG(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region);
    void unpackRecvBufferGwC(size_t haloId, size_t bufferId, cl::Buffer buf);
    void unpackRecvBufferGwG(size_t haloId, size_t bufferId, cl::Buffer buf);
#endif /*! \endcond */

    /*!
     *
     * Shortcut function. If only one buffer is used, this will both pack the data out of the provided buffer and send it off, all with one call.
     *
     * Function for CPU-CPU communication. The equivalent functions for CPU/GPU communication vary only in the ending of the name. Possible variants
     * are GwC, CwG, and GwG.
     *
     * \param haloId
     *  The id of the halo region. This is the index of this halo region in the local halo specification provided with setLocalHaloInfo().
     * \param buf
     *  The buffer from which the data is to be extracted according to the local halo specification.
     * \param msgtag
     *  The message tag to be used for this send. This information only has to be specified the first time the send for the halo region with
     *  the specified id is started. Each subsequent call, the message tag that was passed the very first call will be re-used. This works
     *  equivalently to an MPI tag (and is, in fact, identical to it for MPI communication).
     * \param region
     *  Specification of the area of the current halo that is to be packed. This is specified relativce to the current halo, i.e., (x,y) = (0,0) is
     *  the bottom left corner of the halo region. %Tausch2D expects the following variables to be set:
     *  variable | description
     *  :-------: | -------
     *   x | The starting x coordinate of the region to be packed
     *   y | The starting y coordinate of the region to be packed
     *   width | The width of the region to be packed
     *   height | The height of the region to be packed
     *
     */
    void packAndSendCwC(size_t haloId, buf_t *buf, TauschPackRegion region, int msgtag = -1);
    void packAndSendCwC(size_t haloId, buf_t *buf, int msgtag = -1);
#ifdef TAUSCH_OPENCL /*! \cond DoxygenHideThis */
    void packAndSendCwG(size_t haloId, buf_t *buf, TauschPackRegion region, int msgtag = -1);
    void packAndSendGwC(size_t haloId, cl::Buffer buf, int msgtag = -1);
    void packAndSendGwG(size_t haloId, cl::Buffer buf, int msgtag = -1);
#endif /*! \endcond */

    /*!
     *
     * Shortcut function. If only one buffer is used, this will both receive the MPI message and unpack the received data into the provided buffer,
     * all with one call.
     *
     * Function for CPU-CPU communication. The equivalent functions for CPU/GPU communication vary only in the ending of the name.
     * Possible variants are GwC, CwG, and GwG.
     *
     * \param haloId
     *  The id of the halo region. This is the index of this halo region in the remote halo specification provided with setRemoteHaloInfo().
     * \param[out] buf
     *  The buffer to which the extracted data is to be written to according to the remote halo specification
     * \param region
     *  Specification of the area of the current halo that is to be packed. This is specified relativce to the current halo, i.e., (x,y) = (0,0) is
     *  the bottom left corner of the halo region. %Tausch2D expects the following variables to be set:
     *  variable | description
     *  :-------: | -------
     *   x | The starting x coordinate of the region to be packed
     *   y | The starting y coordinate of the region to be packed
     *   width | The width of the region to be packed
     *   height | The height of the region to be packed
     *
     */
    void recvAndUnpackCwC(size_t haloId, buf_t *buf, TauschPackRegion region);
    void recvAndUnpackCwC(size_t haloId, buf_t *buf);
#ifdef TAUSCH_OPENCL /*! \cond DoxygenHideThis */
    void recvAndUnpackCwG(size_t haloId, buf_t *buf, TauschPackRegion region);
    void recvAndUnpackGwC(size_t haloId, cl::Buffer buf);
    void recvAndUnpackGwG(size_t haloId, cl::Buffer buf);
#endif /*! \endcond */

    /*!
     *
     * A simple shortcut for creating a filled TauschPackRegion object. Allows to create one such object using a single function call instead of
     * specifying each entry individually.
     *
     * \param x
     *  The starting x coordinate of the region, relative to the halo area.
     * \param y
     *  The starting y coordinate of the region, relative to the halo area.
     * \param width
     *  The width of the region.
     * \param height
     *  The height of the region.
     * \return
     *  The filled TauschPackRegion object.
     *
     */
    TauschPackRegion createFilledPackRegion(size_t x, size_t y, size_t width, size_t height);
    TauschPackRegion createFilledPackRegion(size_t startAtIndex, size_t endAtIndex);

    /*!
     *
     * A simple shortcut for creating a filled TauschHaloSpec object. Allows to create one such object using a single function call instead of
     * specifying each entry individually.
     *
     * \param bufferWidth
     *  The width of the underlying buffer.
     * \param bufferHeight
     *  The height of the underlying buffer.
     * \param haloX
     *  The starting x coordinate of the halo region.
     * \param haloY
     *  The starting y coordinate of the halo region.
     * \param haloWidth
     *  The width of the halo region.
     * \param haloHeight
     *  The height of the halo region.
     * \param remoteMpiRank
     *  The remote MPI rank associated with this halo region. This is either the sending or receiving MPI rank, depending on if the halo region
     * specifies a local or remote halo.
     * \return
     *  The filled TauschHaloSpec object.
     *
     */
    TauschHaloSpec createFilledHaloSpec(size_t bufferWidth, size_t bufferHeight, size_t haloX, size_t haloY,
                                        size_t haloWidth, size_t haloHeight, int remoteMpiRank);
    TauschHaloSpec createFilledHaloSpec(std::vector<size_t> haloIndicesInBuffer);


#ifdef TAUSCH_OPENCL

    /*! \name Public Member Functions (OpenCL)
     * Alll member functions relating to GPUs and OpenCL in general. <b>Note:</b> These are only available if %Tausch2D was compiled
     * with OpenCL support!
     */

    ///@{

    /*!
     * Enables the support of GPUs. This has to be called before any of the GPU/OpenCL functions are called, as it sets up a few things that are
     * necessary later-on, mainly setting up an OpenCL environment that is then used by %Tausch2D and that can also be used by the user through some
     * accessor functions.
     * \param blockingSyncCpuGpu
     *  If the CPU and the GPU part are running in the <i>same</i> thread, then this has to be set to false, otherwise %Tausch2D will reach a
     *  deadlock. If, however, both parts run in the same thread, then setting this to true automatically takes care of making sure that halo data
     *  is not read by the other thread before it is completley written. This can also be handled manually by the user, in that case this boolean
     *  should be set to false.
     * \param clLocalWorkgroupSize
     *  The local workgroup size used by OpenCL. This is typically a multiple of 32, the optimal value depends on the underlying hardware.
     * \param giveOpenCLDeviceName
     *  If this is set to true, then %Tausch2D will print the name of the OpenCL device that it is using.
     * \param showOpenCLBuildLog
     *  If set, outputs the build log of the OpenCL compiler.
     */
    void enableOpenCL(bool blockingSyncCpuGpu, size_t clLocalWorkgroupSize, bool giveOpenCLDeviceName, bool showOpenCLBuildLog);
    void enableOpenCL(cl::Device cl_defaultDevice, cl::Context cl_context, cl::CommandQueue cl_queue, bool blockingSyncCpuGpu,
                      size_t clLocalWorkgroupSize, bool showOpenCLBuildLog);

    /*!
     * This provides an access to the OpenCL Context object, that is used internally by %Tausch2D. It allows the user to 'piggyback' onto the OpenCL
     * environment set up by %Tausch2D.
     * \return
     *  The OpenCL Context object.
     */
    cl::Context getOpenCLContext() { return cl_context; }
    /*!
     * This provides an access to the OpenCL CommandQueue object, that is used internally by %Tausch2D. It allows the user to 'piggyback' onto the
     * OpenCL environment set up by %Tausch2D.
     * \return
     *  The OpenCL CommandQueue object.
     */
    cl::CommandQueue getOpenCLQueue() { return cl_queue; }

    ///@}


#endif

    size_t getNumBuffers() { return numBuffers; }
    size_t getNumLocalHaloCpuWithCpu() { return mpiSendBufferCpuWithCpu.size(); }
    size_t getNumRemoteHaloCpuWithCpu() { return mpiRecvBufferCpuWithCpu.size(); }
    size_t getNumPointsLocalHaloCpuWithCpu(size_t id) { return localTotalBufferSizeCwC[id]; }
    size_t getNumPointsRemoteHaloCpuWithCpu(size_t id) { return remoteTotalBufferSizeCwC[id]; }
    buf_t *getSendBufferCpuWithCpu(size_t id) { return mpiSendBufferCpuWithCpu[id]; }
    buf_t *getRecvBufferCpuWithCpu(size_t id) { return mpiRecvBufferCpuWithCpu[id]; }

private:

    MPI_Comm TAUSCH_COMM;
    int mpiRank, mpiSize;

    std::vector<TauschHaloSpec> localHaloSpecsCpuWithCpu;
    std::vector<TauschHaloSpec> remoteHaloSpecsCpuWithCpu;

    size_t numBuffers;
    size_t *valuesPerPointPerBuffer;
    bool valuesPerPointPerBufferAllOne;

    std::vector<buf_t*> mpiRecvBufferCpuWithCpu;
    std::vector<buf_t*> mpiSendBufferCpuWithCpu;
    std::vector<MPI_Request> mpiRecvRequestsCpuWithCpu;
    std::vector<MPI_Request> mpiSendRequestsCpuWithCpu;

    buf_t **mpiRecvBufferGpuWithGpu;
    buf_t **mpiSendBufferGpuWithGpu;
    MPI_Request *mpiRecvRequestsGpuWithGpu;
    MPI_Request *mpiSendRequestsGpuWithGpu;

    MPI_Datatype mpiDataType;

    std::vector<bool> setupMpiSendCpuWithCpu;
    std::vector<bool> setupMpiRecvCpuWithCpu;
    bool *setupMpiSendGpuWithGpu;
    bool *setupMpiRecvGpuWithGpu;

    std::vector<size_t> localBufferOffsetCwC;
    std::vector<size_t> remoteBufferOffsetCwC;
    std::vector<size_t> localTotalBufferSizeCwC;
    std::vector<size_t> remoteTotalBufferSizeCwC;

    size_t *remoteBufferOffsetCwG;
    size_t *localBufferOffsetCwG;
    int *localTotalBufferSizeCwG;
    int *remoteTotalBufferSizeCwG;

    int *localBufferOffsetGwC;
    int *remoteBufferOffsetGwC;
    size_t *localTotalBufferSizeGwC;
    size_t *remoteTotalBufferSizeGwC;

    int *localBufferOffsetGwG;
    int *remoteBufferOffsetGwG;
    size_t *localTotalBufferSizeGwG;
    size_t *remoteTotalBufferSizeGwG;

    std::vector<size_t> alreadyDeletedLocalHaloIds;
    std::vector<size_t> alreadyDeletedRemoteHaloIds;

#ifdef TAUSCH_OPENCL

    std::atomic<buf_t> **sendBufferCpuWithGpu;
    std::atomic<buf_t> **sendBufferGpuWithCpu;
    buf_t **recvBufferGpuWithCpu;
    buf_t **recvBufferCpuWithGpu;
    cl::Buffer *cl_sendBufferGpuWithCpu;
    cl::Buffer *cl_recvBufferGpuWithCpu;
    cl::Buffer *cl_sendBufferGpuWithGpu;
    cl::Buffer *cl_recvBufferGpuWithGpu;

    // gpu with cpu def
    size_t localHaloNumPartsGpuWithCpu;
    TauschHaloSpec *localHaloSpecsGpuWithCpu;
    cl::Buffer *cl_localHaloSpecsGpuWithCpu;
    size_t remoteHaloNumPartsGpuWithCpu;
    TauschHaloSpec *remoteHaloSpecsGpuWithCpu;
    cl::Buffer *cl_remoteHaloSpecsGpuWithCpu;

    // cpu with gpu def
    size_t localHaloNumPartsCpuWithGpu;
    TauschHaloSpec *localHaloSpecsCpuWithGpu;
    cl::Buffer *cl_localHaloSpecsCpuWithGpu;
    size_t remoteHaloNumPartsCpuWithGpu;
    TauschHaloSpec *remoteHaloSpecsCpuWithGpu;
    cl::Buffer *cl_remoteHaloSpecsCpuWithGpu;

    // gpu with gpu def
    size_t localHaloNumPartsGpuWithGpu;
    TauschHaloSpec *localHaloSpecsGpuWithGpu;
    cl::Buffer *cl_localHaloSpecsGpuWithGpu;
    size_t remoteHaloNumPartsGpuWithGpu;
    TauschHaloSpec *remoteHaloSpecsGpuWithGpu;
    cl::Buffer *cl_remoteHaloSpecsGpuWithGpu;


    cl::Buffer cl_valuesPerPointPerBuffer;

    cl::Device cl_defaultDevice;
    cl::Context cl_context;
    cl::CommandQueue cl_queue;
    cl::Platform cl_platform;
    cl::Program cl_programs;

    void setupOpenCL(bool giveOpenCLDeviceName);
    void compileKernels();
    void syncTwoThreads();

    size_t obtainRemoteId(int msgtag);

    bool blockingSyncCpuGpu;
    size_t cl_kernelLocalSize;
    bool showOpenCLBuildLog;

    std::atomic<int> *msgtagsCpuToGpu;
    std::atomic<int> *msgtagsGpuToCpu;

    std::atomic<int> sync_counter[2];
    std::atomic<int> sync_lock[2];

    bool setupCpuWithGpu;
    bool setupGpuWithCpu;
    bool setupGpuWithGpu;

#endif

};


#endif // TAUSCH2D_H
