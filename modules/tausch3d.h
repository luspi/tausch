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
#include <fstream>
#include <atomic>

#ifdef TAUSCH_OPENCL
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#endif

/*!
 *
 * \brief
 *  A library providing a clean and efficient interface for halo exchange in two dimensions.
 *
 * %Tausch3D is a library that provides a clean and efficient C and C++ API for halo exchange for three dimensional domains. It doesn't assume
 * anything about the grid, except that the data is stored in one contiguous buffer. After specifying the local and remote halo regions, it takes care
 * of extracting the data for each halo from as many buffers as there are, sends the data for each halo combined into a single message each, and
 * unpacks the received data into the same number of buffers again. This is a template class and can be used with any of the common
 * C++ data types
 */
template <class buf_t>
class Tausch3D : public Tausch<buf_t> {

public:

    /*!
     *
     * The constructor, initiating the 3D Tausch object.
     *
     * \param mpiDataType
     *  The MPI_Datatype corresponding to the data type used for the template.
     * \param numBuffers
     *  The number of buffers that will be used. If more than one, they are all combined into one message for halo exchanges. Default value: 1
     * \param valuesPerPointPerBuffer
     *  How many values are stored consecutively per point in the same buffer. Each buffer can have different number of values stored per point. This
     *  is expected to be an array of the same size as the number of buffers. If set to nullptr, all buffers are assumed to store 1 value per point.
     * \param comm
     *  The MPI Communictor to be used. %Tausch1D will duplicate the communicator, thus it is safe to have multiple instances of %Tausch1D working
     *  with the same communicator. By default, MPI_COMM_WORLD will be used.
     *
     */
    Tausch3D(MPI_Datatype mpiDataType, int numBuffers = 1, size_t *valuesPerPointPerBuffer = nullptr, MPI_Comm comm = MPI_COMM_WORLD);

    /*!
     * The destructor cleaning up all memory.
     */
    ~Tausch3D();

    /*!
     *
     * Set the info about all local halos that need to be sent to remote MPI ranks.
     *
     * \param flags
     *  This is expected to be a bit wise combination of two flags. First choose one of TAUSCH_CPU or TAUSCH_GPU and combine it with either of
     *  TAUSCH_WITHCPU or TAUSCH_WITHGPU.
     * \param numHaloParts
     *  How many different parts there are to the halo
     * \param haloSpecs
     *  The specification of the different halo parts. This is done using the simple struct TauschHaloSpec, containing variables for all the necessary
     *  entries. %Tausch3D expects the following variables to be set:
     *  variable | description
     *  :-------: | -------
     *   haloX | The starting x coordinate of the halo region
     *   haloY | The starting y coordinate of the halo region
     *   haloZ | The starting z coordinate of the halo region
     *   haloWidth | The width of the halo region (x direction)
     *   haloHeight | The height of the halo region (y direction)
     *   haloDepth | The height of the halo region (z direction)
     *   bufferWidth | The width of the underlying buffer (x direction)
     *   bufferHeight | The height of the underlying buffer (y direction)
     *   bufferDepth | The depth of the underlying buffer (z direction)
     *   remoteMpiRank | The receiving processor
     *
     */
    void setLocalHaloInfo(TauschDeviceDirection flags, size_t numHaloParts, TauschHaloSpec *haloSpecs);

    /*!
     *
     * Set the info about all remote halos that are needed by this MPI rank.
     *
     * \param flags
     *  This is expected to be a bit wise combination of two flags. First choose one of TAUSCH_CPU or TAUSCH_GPU and combine it with either of
     *  TAUSCH_WITHCPU or TAUSCH_WITHGPU.
     * \param numHaloParts
     *  How many different parts there are to the halo
     * \param haloSpecs
     *  The specification of the different halo parts. This is done using the simple struct TauschHaloSpec, containing variables for all the necessary
     *  entries. %Tausch3D expects the following variables to be set:
     *  variable | description
     *  :-------: | -------
     *   haloX | The starting x coordinate of the halo region
     *   haloY | The starting y coordinate of the halo region
     *   haloZ | The starting z coordinate of the halo region
     *   haloWidth | The width of the halo region (x direction)
     *   haloHeight | The height of the halo region (y direction)
     *   haloDepth | The height of the halo region (z direction)
     *   bufferWidth | The width of the underlying buffer (x direction)
     *   bufferHeight | The height of the underlying buffer (y direction)
     *   bufferDepth | The depth of the underlying buffer (z direction)
     *   remoteMpiRank | The sending processor
     *
     */
    void setRemoteHaloInfo(TauschDeviceDirection flags, size_t numHaloParts, TauschHaloSpec *haloSpecs);

    /*!
     *
     * Post the receive for the specified remote halo region of the current MPI rank. This doesn't do anything else but post the MPI_Recv.
     *
     * \param flags
     *  This is expected to be a bit wise combination of two flags. First choose one of TAUSCH_CPU or TAUSCH_GPU and combine it with either of
     *  TAUSCH_WITHCPU or TAUSCH_WITHGPU.
     * \param haloId
     *  The id of the halo region. This is the index of this halo region in the local halo specification provided with setRemoteHaloInfo().
     * \param msgtag
     *  The message tag to be used for this receive. This information only has to be specified the first time the receive for the halo region with
     *  the specified id is posted. Each subsequent call, the msgtag that was passed the very first call will be re-used. This works equivalently to
     *  an MPI tag (and is, in fact, identical to it for MPI communication).
     *
     */
    void postReceive(TauschDeviceDirection flags, size_t haloId, int msgtag = -1);

    /*!
     *
     * Post all receives for the current MPI rank. This doesn't do anything else but post the MPI_Recv for each remote halo region.
     *
     * \param flags
     *  This is expected to be a bit wise combination of two flags. First choose one of TAUSCH_CPU or TAUSCH_GPU and combine it with either of
     *  TAUSCH_WITHCPU or TAUSCH_WITHGPU.
     * \param msgtag
     *  An array containing the message tags, one for each halo region. This information only has to be specified the first time the receives are
     *  posted. Each subsequent call, the message tags that were passed the very first call will be re-used. This works equivalently to
     *  an MPI tag (and is, in fact, identical to it for MPI communication).
     *
     */
    void postAllReceives(TauschDeviceDirection flags, int *msgtag = nullptr);

    /*!
     *
     * This packs the specified region of the specified halo area of the specified buffer for a send. This has to be called for all buffers before
     * sending the message.
     *
     * \param flags
     *  This is expected to be a bit wise combination of two flags. First choose one of TAUSCH_CPU or TAUSCH_GPU and combine it with either of
     *  TAUSCH_WITHCPU or TAUSCH_WITHGPU.
     * \param haloId
     *  The id of the halo region. This is the index of this halo region in the local halo specification provided with setLocalHaloInfo().
     * \param bufferId
     *  The id of the buffer. The order of the buffers will be preserved, i.e., packing buffer with id 1 required unpacking that buffer with id 1.
     *  The numbering of the buffers has to start with 0!
     * \param buf
     *  The buffer from which the data is to be extracted according to the local halo specification.
     * \param region
     *  Specification of the area of the current halo that is to be packed. This is specified relativce to the current halo, i.e., (x,y,z) = (0,0,0)
     *  is the bottom left corner of the halo region. %Tausch3D expects the following variables to be set:
     *  variable | description
     *  :-------: | -------
     *   x | The starting x coordinate of the region to be packed
     *   y | The starting y coordinate of the region to be packed
     *   z | The starting z coordinate of the region to be packed
     *   width | The width of the region to be packed
     *   height | The height of the region to be packed
     *   depth | The depth of the region to be packed
     */
    void packSendBuffer(TauschDeviceDirection flags, size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region);

    /*!
     *
     * Overloaded function, taking the full halo region for packing.
     *
     * \param flags
     *  This is expected to be a bit wise combination of two flags. First choose one of TAUSCH_CPU or TAUSCH_GPU and combine it with either of
     *  TAUSCH_WITHCPU or TAUSCH_WITHGPU.
     * \param haloId
     *  The id of the halo region. This is the index of this halo region in the local halo specification provided with setLocalHaloInfo().
     * \param bufferId
     *  The id of the buffer. The order of the buffers will be preserved, i.e., packing buffer with id 1 required unpacking that buffer with id 1.
     *  The numbering of the buffers has to start with 0!
     * \param buf
     *  The buffer from which the data is to be extracted according to the local halo specification.
     *
     */
    void packSendBuffer(TauschDeviceDirection flags, size_t haloId, size_t bufferId, buf_t *buf);

#ifdef TAUSCH_OPENCL
    /*!
     *
     * Overloaded function, taking the full halo region of an OpenCL buffer for packing.
     *
     * \param flags
     *  This is expected to be a bit wise combination of two flags. First choose one of TAUSCH_CPU or TAUSCH_GPU and combine it with either of
     *  TAUSCH_WITHCPU or TAUSCH_WITHGPU.
     * \param haloId
     *  The id of the halo region. This is the index of this halo region in the local halo specification provided with setLocalHaloInfo().
     * \param bufferId
     *  The id of the buffer. The order of the buffers will be preserved, i.e., packing buffer with id 1 required unpacking that buffer with id 1.
     *  The numbering of the buffers has to start with 0!
     * \param buf
     *  The OpenCL buffer from which the data is to be extracted according to the local halo specification.
     *
     */
    void packSendBuffer(TauschDeviceDirection flags, size_t haloId, size_t bufferId, cl::Buffer buf);
#endif

    /*!
     *
     * Sends off the send buffer for the specified halo region. This starts the respective MPI_Send.
     *
     * \param flags
     *  This is expected to be a bit wise combination of two flags. First choose one of TAUSCH_CPU or TAUSCH_GPU and combine it with either of
     *  TAUSCH_WITHCPU or TAUSCH_WITHGPU.
     * \param haloId
     *  The id of the halo region. This is the index of this halo region in the local halo specification provided with setLocalHaloInfo().
     * \param msgtag
     *  The message tag to be used for this send. This information only has to be specified the first time the send for the halo region with
     *  the specified id is started. Each subsequent call, the message tag that was passed the very first call will be re-used. This works
     *  equivalently to an MPI tag (and is, in fact, identical to it for MPI communication).
     *
     */
    void send(TauschDeviceDirection flags, size_t haloId, int msgtag = -1);

    /*!
     *
     * Makes sure the MPI message for the specified halo is received by this buffer. It does not do anything with that message!
     *
     * \param flags
     *  This is expected to be a bit wise combination of two flags. First choose one of TAUSCH_CPU or TAUSCH_GPU and combine it with either of
     *  TAUSCH_WITHCPU or TAUSCH_WITHGPU.
     * \param haloId
     *  The id of the halo region. This is the index of this halo region in the remote halo specification provided with setRemoteHaloInfo().
     *
     */
    void recv(TauschDeviceDirection flags, size_t haloId);

    /*!
     *
     * This unpacks the next halo from the received message into the specified and provided buffer. This has to be called for all buffers.
     *
     * \param flags
     *  This is expected to be a bit wise combination of two flags. First choose one of TAUSCH_CPU or TAUSCH_GPU and combine it with either of
     *  TAUSCH_WITHCPU or TAUSCH_WITHGPU.
     * \param haloId
     *  The id of the halo region. This is the index of this halo region in the remote halo specification provided with setRemoteHaloInfo().
     * \param bufferId
     *  The id of the buffer. The order of the buffers will be preserved, i.e., packing buffer with id 1 required unpacking that buffer with id 1.
     *  The numbering of the buffers has to start with 0!
     * \param[out] buf
     *  The buffer to which the extracted data is to be written to according to the remote halo specification
     * \param region
     *  Specification of the area of the current halo that is to be packed. This is specified relativce to the current halo, i.e., (x,y,z) = (0,0,0)
     *  is the bottom left corner of the halo region. %Tausch3D expects the following variables to be set:
     *  variable | description
     *  :-------: | -------
     *   x | The starting x coordinate of the region to be packed
     *   y | The starting y coordinate of the region to be packed
     *   z | The starting z coordinate of the region to be packed
     *   width | The width of the region to be packed
     *   height | The height of the region to be packed
     *   depth | The depth of the region to be packed
     *
     */
    void unpackRecvBuffer(TauschDeviceDirection flags, size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region);

    /*!
     *
     * Overloaded function, taking the full halo region for unpacking.
     *
     * \param flags
     *  This is expected to be a bit wise combination of two flags. First choose one of TAUSCH_CPU or TAUSCH_GPU and combine it with either of
     *  TAUSCH_WITHCPU or TAUSCH_WITHGPU.
     * \param haloId
     *  The id of the halo region. This is the index of this halo region in the remote halo specification provided with setRemoteHaloInfo().
     * \param bufferId
     *  The id of the buffer. The order of the buffers will be preserved, i.e., packing buffer with id 1 required unpacking that buffer with id 1.
     *  The numbering of the buffers has to start with 0!
     * \param[out] buf
     *  The buffer to which the extracted data is to be written to according to the remote halo specification
     *
     */
    void unpackRecvBuffer(TauschDeviceDirection flags, size_t haloId, size_t bufferId, buf_t *buf);

#ifdef TAUSCH_OPENCL
    /*!
     *
     * Overloaded function, taking the full halo region of an OpenCL buffer for unpacking.
     *
     * \param flags
     *  This is expected to be a bit wise combination of two flags. First choose one of TAUSCH_CPU or TAUSCH_GPU and combine it with either of
     *  TAUSCH_WITHCPU or TAUSCH_WITHGPU.
     * \param haloId
     *  The id of the halo region. This is the index of this halo region in the remote halo specification provided with setRemoteHaloInfo().
     * \param bufferId
     *  The id of the buffer. The order of the buffers will be preserved, i.e., packing buffer with id 1 required unpacking that buffer with id 1.
     *  The numbering of the buffers has to start with 0!
     * \param[out] buf
     *  The OpenCL buffer to which the extracted data is to be written to according to the remote halo specification
     *
     */
    void unpackRecvBuffer(TauschDeviceDirection flags, size_t haloId, size_t bufferId, cl::Buffer buf);
#endif

    /*!
     *
     * Shortcut function. If only one buffer is used, this will both pack the data out of the provided buffer and send it off, all with one call.
     *
     * \param flags
     *  This is expected to be a bit wise combination of two flags. First choose one of TAUSCH_CPU or TAUSCH_GPU and combine it with either of
     *  TAUSCH_WITHCPU or TAUSCH_WITHGPU.
     * \param haloId
     *  The id of the halo region. This is the index of this halo region in the local halo specification provided with setLocalHaloInfo().
     * \param buf
     *  The buffer from which the data is to be extracted according to the local halo specification.
     * \param msgtag
     *  The message tag to be used for this send. This information only has to be specified the first time the send for the halo region with
     *  the specified id is started. Each subsequent call, the message tag that was passed the very first call will be re-used. This works
     *  equivalently to an MPI tag (and is, in fact, identical to it for MPI communication).
     * \param region
     *  Specification of the area of the current halo that is to be packed. This is specified relativce to the current halo, i.e., (x,y,z) = (0,0,0)
     *  is the bottom left corner of the halo region. %Tausch3D expects the following variables to be set:
     *  variable | description
     *  :-------: | -------
     *   x | The starting x coordinate of the region to be packed
     *   y | The starting y coordinate of the region to be packed
     *   z | The starting z coordinate of the region to be packed
     *   width | The width of the region to be packed
     *   height | The height of the region to be packed
     *   depth | The depth of the region to be packed
     */
    void packAndSend(TauschDeviceDirection flags, size_t haloId, buf_t *buf, TauschPackRegion region, int msgtag = -1);

    /*!
     *
     * Overloaded function, taking the full halo region for packing and sending.
     *
     * \param flags
     *  This is expected to be a bit wise combination of two flags. First choose one of TAUSCH_CPU or TAUSCH_GPU and combine it with either of
     *  TAUSCH_WITHCPU or TAUSCH_WITHGPU.
     * \param haloId
     *  The id of the halo region. This is the index of this halo region in the local halo specification provided with setLocalHaloInfo().
     * \param buf
     *  The buffer from which the data is to be extracted according to the local halo specification.
     * \param msgtag
     *  The message tag to be used for this send. This information only has to be specified the first time the send for the halo region with
     *  the specified id is started. Each subsequent call, the message tag that was passed the very first call will be re-used. This works
     *  equivalently to an MPI tag (and is, in fact, identical to it for MPI communication).
     *
     */
    void packAndSend(TauschDeviceDirection flags, size_t haloId, buf_t *buf, int msgtag = -1);

#ifdef TAUSCH_OPENCL
    /*!
     *
     * Overloaded function, taking the full halo region of an OpenCL buffer for packing and sending.
     *
     * \param flags
     *  This is expected to be a bit wise combination of two flags. First choose one of TAUSCH_CPU or TAUSCH_GPU and combine it with either of
     *  TAUSCH_WITHCPU or TAUSCH_WITHGPU.
     * \param haloId
     *  The id of the halo region. This is the index of this halo region in the local halo specification provided with setLocalHaloInfo().
     * \param buf
     *  The OpenCL buffer from which the data is to be extracted according to the local halo specification.
     * \param msgtag
     *  The message tag to be used for this send. This information only has to be specified the first time the send for the halo region with
     *  the specified id is started. Each subsequent call, the message tag that was passed the very first call will be re-used. This works
     *  equivalently to an MPI tag (and is, in fact, identical to it for MPI communication).
     *
     */
    void packAndSend(TauschDeviceDirection flags, size_t haloId, cl::Buffer buf, int msgtag = -1);
#endif

    /*!
     *
     * Shortcut function. If only one buffer is used, this will both receive the MPI message and unpack the received data into the provided buffer,
     * all with one call.
     *
     * \param flags
     *  This is expected to be a bit wise combination of two flags. First choose one of TAUSCH_CPU or TAUSCH_GPU and combine it with either of
     *  TAUSCH_WITHCPU or TAUSCH_WITHGPU.
     * \param haloId
     *  The id of the halo region. This is the index of this halo region in the remote halo specification provided with setRemoteHaloInfo().
     * \param[out] buf
     *  The buffer to which the extracted data is to be written to according to the remote halo specification
     * \param region
     *  Specification of the area of the current halo that is to be packed. This is specified relativce to the current halo, i.e., (x,y,z) = (0,0,0)
     *  is the bottom left corner of the halo region. %Tausch3D expects the following variables to be set:
     *  variable | description
     *  :-------: | -------
     *   x | The starting x coordinate of the region to be packed
     *   y | The starting y coordinate of the region to be packed
     *   z | The starting z coordinate of the region to be packed
     *   width | The width of the region to be packed
     *   height | The height of the region to be packed
     *   depth | The depth of the region to be packed
     *
     */
    void recvAndUnpack(TauschDeviceDirection flags, size_t haloId, buf_t *buf, TauschPackRegion region);

    /*!
     *
     * Overloaded function, taking the full halo region for receiving and unpacking.
     *
     * \param flags
     *  This is expected to be a bit wise combination of two flags. First choose one of TAUSCH_CPU or TAUSCH_GPU and combine it with either of
     *  TAUSCH_WITHCPU or TAUSCH_WITHGPU.
     * \param haloId
     *  The id of the halo region. This is the index of this halo region in the remote halo specification provided with setRemoteHaloInfo().
     * \param[out] buf
     *  The buffer to which the extracted data is to be written to according to the remote halo specification
     *
     */
    void recvAndUnpack(TauschDeviceDirection flags, size_t haloId, buf_t *buf);

#ifdef TAUSCH_OPENCL
    /*!
     *
     * Overloaded function, taking the full halo region of an OpenCL buffer for receiving and unpacking.
     *
     * \param flags
     *  This is expected to be a bit wise combination of two flags. First choose one of TAUSCH_CPU or TAUSCH_GPU and combine it with either of
     *  TAUSCH_WITHCPU or TAUSCH_WITHGPU.
     * \param haloId
     *  The id of the halo region. This is the index of this halo region in the remote halo specification provided with setRemoteHaloInfo().
     * \param[out] buf
     *  The OpenCL buffer to which the extracted data is to be written to according to the remote halo specification
     *
     */
    void recvAndUnpack(TauschDeviceDirection flags, size_t haloId, cl::Buffer buf);
#endif

#ifdef TAUSCH_OPENCL

    /*! \name Public Member Functions (OpenCL)
     * Alll member functions relating to GPUs and OpenCL in general. <b>Note:</b> These are only available if %Tausch3D was compiled
     * with OpenCL support!
     */

    ///@{

    /*!
     * Enables the support of GPUs. This has to be called before any of the GPU/OpenCL functions are called, as it sets up a few things that are
     * necessary later-on, mainly setting up an OpenCL environment that is then used by %Tausch3D and that can also be used by the user through some
     * accessor functions.
     * \param blockingSyncCpuGpu
     *  If the CPU and the GPU part are running in the <i>same</i> thread, then this has to be set to false, otherwise %Tausch3D will reach a
     *  deadlock. If, however, both parts run in the same thread, then setting this to true automatically takes care of making sure that halo data
     *  is not read by the other thread before it is completley written. This can also be handled manually by the user, in that case this boolean
     *  should be set to false.
     * \param clLocalWorkgroupSize
     *  The local workgroup size used by OpenCL. This is typically a multiple of 32, the optimal value depends on the underlying hardware.
     * \param giveOpenCLDeviceName
     *  If this is set to true, then %Tausch3D will print the name of the OpenCL device that it is using.
     * \param showOpenCLBuildLog
     *  If set, outputs the build log of the OpenCL compiler.
     */
    void enableOpenCL(bool blockingSyncCpuGpu, int clLocalWorkgroupSize, bool giveOpenCLDeviceName, bool showOpenCLBuildLog);
    void enableOpenCL(cl::Device cl_defaultDevice, cl::Context cl_context, cl::CommandQueue cl_queue, bool blockingSyncCpuGpu, int clLocalWorkgroupSize, bool showOpenCLBuildLog);

    /*!
     * This provides an access to the OpenCL Context object, that is used internally by %Tausch3D. It allows the user to 'piggyback' onto the OpenCL
     * environment set up by %Tausch3D.
     * \return
     *  The OpenCL Context object.
     */
    cl::Context getOpenCLContext() { return cl_context; }
    /*!
     * This provides an access to the OpenCL CommandQueue object, that is used internally by %Tausch3D. It allows the user to 'piggyback' onto the
     * OpenCL environment set up by %Tausch3D.
     * \return
     *  The OpenCL CommandQueue object.
     */
    cl::CommandQueue getOpenCLQueue() { return cl_queue; }

    ///@}

#endif

    /*!
     * \cond DoxygenHideThis
     */

    void setLocalHaloInfoCpu(size_t numHaloParts, TauschHaloSpec *haloSpecs);
#ifdef TAUSCH_OPENCL
    void setLocalHaloInfoCpuForGpu(size_t numHaloParts, TauschHaloSpec *haloSpecs);
    void setLocalHaloInfoGpu(size_t numHaloParts, TauschHaloSpec *haloSpecs);
#endif

    void setRemoteHaloInfoCpu(size_t numHaloParts, TauschHaloSpec *haloSpecs);
#ifdef TAUSCH_OPENCL
    void setRemoteHaloInfoCpuForGpu(size_t numHaloParts, TauschHaloSpec *haloSpecs);
    void setRemoteHaloInfoGpu(size_t numHaloParts, TauschHaloSpec *haloSpecs);
#endif

    void postReceiveCpu(size_t haloId, int mpitag = -1);
#ifdef TAUSCH_OPENCL
    void postReceiveCpuForGpu(size_t haloId, int msgtag = -1);
    void postReceiveGpu(size_t haloId, int msgtag = -1);
#endif

    void postAllReceivesCpu(int *mpitag = nullptr);
#ifdef TAUSCH_OPENCL
    void postAllReceivesCpuForGpu(int *msgtag = nullptr);
    void postAllReceivesGpu(int *msgtag = nullptr);
#endif

    void packSendBufferCpu(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region);
#ifdef TAUSCH_OPENCL
    void packSendBufferCpuToGpu(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region);
    void packSendBufferGpuToCpu(size_t haloId, size_t bufferId, cl::Buffer buf);
#endif

    void sendCpu(size_t haloId, int mpitag = -1);
#ifdef TAUSCH_OPENCL
    void sendCpuToGpu(size_t haloId, int msgtag);
    void sendGpuToCpu(size_t haloId, int msgtag);
#endif

    void recvCpu(size_t haloId);
#ifdef TAUSCH_OPENCL
    void recvGpuToCpu(size_t haloId);
    void recvCpuToGpu(size_t haloId);
#endif

    void unpackRecvBufferCpu(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region);
#ifdef TAUSCH_OPENCL
    void unpackRecvBufferGpuToCpu(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region);
    void unpackRecvBufferCpuToGpu(size_t haloId, size_t bufferId, cl::Buffer buf);
#endif

    void packAndSendCpu(size_t haloId, buf_t *buf, TauschPackRegion region, int msgtag = -1);
#ifdef TAUSCH_OPENCL
    void packAndSendCpuForGpu(size_t haloId, buf_t *buf, TauschPackRegion region, int msgtag = -1);
    void packAndSendGpu(size_t haloId, cl::Buffer buf, int msgtag = -1);
#endif

    void recvAndUnpackCpu(size_t haloId, buf_t *buf, TauschPackRegion region);
#ifdef TAUSCH_OPENCL
    void recvAndUnpackCpuForGpu(size_t haloId, buf_t *buf, TauschPackRegion region);
    void recvAndUnpackGpu(size_t haloId, cl::Buffer buf);
#endif

    /*!
     * \endcond
     */

private:

    MPI_Comm TAUSCH_COMM;
    int mpiRank, mpiSize;

    size_t localHaloNumParts;
    TauschHaloSpec *localHaloSpecsCpu;
    size_t remoteHaloNumParts;
    TauschHaloSpec *remoteHaloSpecsCpu;

    size_t numBuffers;
    size_t *valuesPerPointPerBuffer;

    buf_t **mpiRecvBuffer;
    buf_t **mpiSendBuffer;
    MPI_Request *mpiRecvRequests;
    MPI_Request *mpiSendRequests;
    MPI_Datatype mpiDataType;

    bool *setupMpiSend;
    bool *setupMpiRecv;

#ifdef TAUSCH_OPENCL

    std::atomic<buf_t> **cpuToGpuSendBuffer;
    std::atomic<buf_t> **gpuToCpuSendBuffer;
    buf_t **cpuToGpuRecvBuffer;
    buf_t **gpuToCpuRecvBuffer;
    cl::Buffer *cl_gpuToCpuSendBuffer;
    cl::Buffer *cl_cpuToGpuRecvBuffer;

    // gpu def
    size_t localHaloNumPartsGpu;
    TauschHaloSpec *localHaloSpecsGpu;
    cl::Buffer *cl_localHaloSpecsGpu;
    size_t remoteHaloNumPartsGpu;
    TauschHaloSpec *remoteHaloSpecsGpu;
    cl::Buffer *cl_remoteHaloSpecsGpu;
    // cpu def
    size_t localHaloNumPartsCpuForGpu;
    TauschHaloSpec *localHaloSpecsCpuForGpu;
    cl::Buffer *cl_localHaloSpecsCpuForGpu;
    size_t remoteHaloNumPartsCpuForGpu;
    TauschHaloSpec *remoteHaloSpecsCpuForGpu;
    cl::Buffer *cl_remoteHaloSpecsCpuForGpu;


    cl::Buffer cl_valuesPerPointPerBuffer;

    cl::Device cl_defaultDevice;
    cl::Context cl_context;
    cl::CommandQueue cl_queue;
    cl::Platform cl_platform;
    cl::Program cl_programs;

    void setupOpenCL(bool giveOpenCLDeviceName);
    void compileKernels();
    void syncCpuAndGpu();

    int obtainRemoteId(int msgtag);

    bool blockingSyncCpuGpu;
    int cl_kernelLocalSize;
    bool showOpenCLBuildLog;

    std::atomic<int> *msgtagsCpuToGpu;
    std::atomic<int> *msgtagsGpuToCpu;

    std::atomic<int> sync_counter[2];
    std::atomic<int> sync_lock[2];

#endif

};


#endif // TAUSCH3D_H
