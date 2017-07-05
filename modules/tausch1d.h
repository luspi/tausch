/*!
 * \file
 * \author  Lukas Spies <LSpies@illinois.edu>
 * \version 1.0
 *
 * \brief
 *  One-dimensional halo exchange library.
 *
 *  A library providing a clean and efficient interface for halo exchange in one dimension.
 *
 */

#ifndef TAUSCH1D_H
#define TAUSCH1D_H

#include "tausch.h"
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <atomic>

#ifdef TAUSCH_OPENCL
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#endif

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
 *  A library providing a clean and efficient interface for halo exchange in one dimension.
 *
 * %Tausch1D is a library that provides a clean and efficient C and C++ API for halo exchange for one dimensional domains. It doesn't assume
 * anything about the grid, except that the data is stored in one contiguous buffer (including halo). After specifying the local and remote halo
 * regions, it takes care of extracting the data for each halo from as many buffers as there are, sends the data for each halo combined into a single
 * message each, and unpacks the received data into the same number of buffers again. This is a template class and can be used with any of the
 * following data types: char, char16_t, char32_t, wchar_t, signed char, short int, int, long, long long, unsigned char, unsigned short int, unsigned
 * int, unsigned long, unsigned long long, float, double, long double, bool.
 */
template <class buf_t>
class Tausch1D : public Tausch<buf_t> {

public:

    /*!
     *
     * The constructor, initiating the 1D Tausch object.
     *
     * \param mpiDataType
     *  The MPI_Datatype corresponding to the datatype used for the template.
     * \param numBuffers
     *  The number of buffers that will be used. If more than one, they are all combined into one message. All buffers will have to use the same
     *  discretisation! Default value: 1
     * \param valuesPerPointPerBuffer
     *  How many values are stored consecutively per point in the same buffer. Each buffer can have different number of values stored per point. This
     *  is expected to be an array of the same size as the number of buffers. If set to nullptr, all buffers are assumed to store 1 value per point.
     * \param comm
     *  The MPI Communictor to be used. %Tausch1D will duplicate the communicator, thus it is safe to have multiple instances of %Tausch1D working
     *  with the same communicator. By default, MPI_COMM_WORLD will be used.
     *
     */
    Tausch1D(MPI_Datatype mpiDataType, size_t numBuffers = 1, size_t *valuesPerPointPerBuffer = nullptr, MPI_Comm comm = MPI_COMM_WORLD);

    /*!
     *
     * The destructor cleaning up all memory.
     *
     */
    ~Tausch1D();

    /*!
     *
     * Set the info about all local halos that need to be sent to remote MPI ranks.
     *
     * \param numHaloParts
     *  How many different parts there are to the halo
     * \param haloSpecs
     *  The specification of the different halo parts. This is done using the simple struct TauschHaloSpec, containing variables for all the necessary
     *  entries. %Tausch1D expects the following variables to be set:
     *  variable | description
     *  :-------: | -------
     *   x | The starting x coordinate of the halo region
     *   width | The width of the halo region
     *   remoteMpiRank | The receiving processor
     *
     */
    void setLocalHaloInfoCpu(size_t numHaloParts, TauschHaloSpec *haloSpecs);

    /*!
     *
     * Set the info about all remote halos that are needed by this MPI rank.
     *
     * \param numHaloParts
     *  How many different parts there are to the halo
     * \param haloSpecs
     *  The specification of the different halo parts. This is done using the simple struct TauschHaloSpec, containing variables for all the necessary
     *  entries. %Tausch1D expects the following variables to be set:
     *  variable | description
     *  :-------: | -------
     *   x | The starting x coordinate of the halo region
     *   width | The width of the halo region
     *   remoteMpiRank | The sending processor
     *
     */
    void setRemoteHaloInfoCpu(size_t numHaloParts, TauschHaloSpec *haloSpecs);

    /*!
     * Post the receive for the specified remote halo region of the current MPI rank. This doesn't do anything else but post the MPI_Recv.
     * \param haloId
     *  The id of the halo region. This is the index of this halo region in the local halo specification provided with setRemoteHaloInfo().
     * \param mpitag
     *  The mpitag to be used for this MPI_Recv. This information only has to be specified the first time the MPI_Recv for the halo region with
     *  the specified id is posted. Each subsequent call, the mpitag that was passed the very first call will be re-used.
     */
    void postReceiveCpu(size_t haloId, int mpitag = -1);

    /*!
     * Post all receives for the current MPI rank. This doesn't do anything else but post the MPI_Recv for each remote halo region.
     * \param mpitag
     *  An array containing the MPI tags, one for each halo region. This information only has to be specified the first time the MPI_Recvs are posted.
     *  Each subsequent call, the mpitags that were passed the very first call will be re-used.
     */
    void postAllReceivesCpu(int *mpitag = nullptr);

    /*!
     *
     * This packs the specified region of the specified halo area for the specified buffer for a send. This has to be called for all buffers before
     * sending the message.
     * \param haloId
     *  The id of the halo region. This is the index of this halo region in the local halo specification provided with setLocalHaloInfo().
     * \param bufferId
     *  The id of the buffer. The order of the buffers will be preserved, i.e., packing buffer with id 1 required unpacking that buffer with id 1.
     *  The numbering of the buffers has to start with 0!
     * \param buf
     *  The buffer from which the data is to be extracted according to the local halo specification.
     * \param region
     *  Specification of the area of the current halo that is to be packed. This is specified relativce to the current halo, i.e., x=0 is
     *  the left edge of the halo region. %Tausch1D expects the following variables to be set:
     *  variable | description
     *  :-------: | -------
     *   x | The starting x coordinate of the region to be packed
     *   width | The width of the region to be packed
     *
     */
    void packSendBufferCpu(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region);

    /*!
     *
     * Overloaded function, packs the full region of the specified halo area.
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
    void packSendBufferCpu(size_t haloId, size_t bufferId, buf_t *buf);

    /*!
     *
     * Sends off the send buffer for the specified halo region. This starts the respective MPI_Send.
     *
     * \param haloId
     *  The id of the halo region. This is the index of this halo region in the local halo specification provided with setLocalHaloInfo().
     * \param mpitag
     *  The mpitag to be used for this MPI_Send. This information only has to be specified the first time the MPI_Send for the halo region with
     *  the specified id is started. Each subsequent call, the mpitag that was passed the very first call will be re-used.
     *
     */
    void sendCpu(size_t haloId, int mpitag = -1);

    /*!
     *
     * Makes sure the MPI message for the specified halo is received by this buffer. It does not do anything with that message!
     *
     * \param haloId
     *  The id of the halo region. This is the index of this halo region in the remote halo specification provided with setRemoteHaloInfo().
     *
     */
    void recvCpu(size_t haloId);

    /*!
     *
     * This unpacks the next halo from the received message into the provided buffer. This has to be called for all buffers.
     *
     * \param haloId
     *  The id of the halo region. This is the index of this halo region in the remote halo specification provided with setRemoteHaloInfo().
     * \param bufferId
     *  The id of the buffer. The order of the buffers will be preserved, i.e., packing buffer with id 1 required unpacking that buffer with id 1.
     *  The numbering of the buffers has to start with 0!
     * \param[out] buf
     *  The buffer to which the extracted data is to be written to according to the remote halo specification
     * \param region
     *  Specification of the area of the current halo that is to be packed. This is specified relative to the current halo, i.e., x=0 is
     *  the left edge of the halo region. %Tausch1D expects the following variables to be set:
     *  variable | description
     *  :-------: | -------
     *   x | The starting x coordinate of the region to be packed
     *   width | The width of the region to be packed
     *
     */
    void unpackRecvBufferCpu(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region);

    /*!
     *
     * This unpacks the next halo from the received message into the provided buffer. This has to be called for all buffers.
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
    void unpackRecvBufferCpu(size_t haloId, size_t bufferId, buf_t *buf);

    /*!
     *
     * Shortcut function. If only one buffer is used, this will both pack the data out of the provided buffer and send it off, all with one call.
     *
     * \param haloId
     *  The id of the halo region. This is the index of this halo region in the local halo specification provided with setLocalHaloInfo().
     * \param buf
     *  The buffer from which the data is to be extracted according to the local halo specification.
     * \param mpitag
     *  The mpitag to be used for this MPI_Send. This information only has to be specified the first time the MPI_Send for the halo region with
     *  the specified id is started. Each subsequent call, the mpitag that was passed the very first call will be re-used.
     * \param region
     *  Specification of the area of the current halo that is to be packed. This is specified relative to the current halo, i.e., x=0 is
     *  the left edge of the halo region. %Tausch1D expects the following variables to be set:
     *  variable | description
     *  :-------: | -------
     *   x | The starting x coordinate of the region to be packed
     *   width | The width of the region to be packed
     *
     */
    void packAndSendCpu(size_t haloId, buf_t *buf, TauschPackRegion region, int mpitag = -1);

    /*!
     *
     * Shortcut function. If only one buffer is used, this will both pack the data out of the provided buffer and send it off, all with one call.
     *
     * \param haloId
     *  The id of the halo region. This is the index of this halo region in the local halo specification provided with setLocalHaloInfo().
     * \param buf
     *  The buffer from which the data is to be extracted according to the local halo specification.
     * \param mpitag
     *  The mpitag to be used for this MPI_Send. This information only has to be specified the first time the MPI_Send for the halo region with
     *  the specified id is started. Each subsequent call, the mpitag that was passed the very first call will be re-used.
     *
     */
    void packAndSendCpu(size_t haloId, buf_t *buf, int mpitag = -1);

    /*!
     *
     * Shortcut function. If only one buffer is used, this will both receive the MPI message and unpack the received data into the provided buffer,
     * all with one call.
     *
     * \param haloId
     *  The id of the halo region. This is the index of this halo region in the remote halo specification provided with setRemoteHaloInfo().
     * \param[out] buf
     *  The buffer to which the extracted data is to be written to according to the remote halo specification
     * \param region
     *  Specification of the area of the current halo that is to be packed. This is specified relative to the current halo, i.e., x=0 is
     *  the left edge of the halo region. %Tausch1D expects the following variables to be set:
     *  variable | description
     *  :-------: | -------
     *   x | The starting x coordinate of the region to be packed
     *   width | The width of the region to be packed
     *
     */
    void recvAndUnpackCpu(size_t haloId, buf_t *buf, TauschPackRegion region);

    /*!
     *
     * Shortcut function. If only one buffer is used, this will both receive the MPI message and unpack the received data into the provided buffer,
     * all with one call.
     *
     * \param haloId
     *  The id of the halo region. This is the index of this halo region in the remote halo specification provided with setRemoteHaloInfo().
     * \param[out] buf
     *  The buffer to which the extracted data is to be written to according to the remote halo specification
     *
     */
    void recvAndUnpackCpu(size_t haloId, buf_t *buf);

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
    void enableOpenCL(bool blockingSyncCpuGpu, int clLocalWorkgroupSize, bool giveOpenCLDeviceName, bool showOpenCLBuildLog);

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

    /*! \name Public Member Functions (CPU -> GPU)
     * All member functions about communicating halo data between the CPU and GPU - CPU side of things. <b>Note:</b> These are only available if
     * %Tausch2D was compiled with OpenCL support!
     */

    ///@{

    /*!
     * Set the info about all local CPU halos that need to be sent to the GPU partition. Note that the <i>remoteMpiRank</i> field of TauschHaloSpec
     * does <b>not</b> need to be set as the data is not sent through MPI!
     * \param numHaloParts
     *  How many different parts there are to the halo.
     * \param haloSpecs
     *  The specification of the different halo parts. This is done using the simple struct TauschHaloSpec, containing variables for all the necessary
     *  entries. %Tausch2D expects the following 4 variables to be set:
     *  variable | description
     *  :-------: | -------
     *   x | The starting x coordinate of the halo region
     *   y | The starting y coordinate of the halo region
     *   width | The width of the halo region
     *   height | The height of the halo region
     */
    void setLocalHaloInfoCpuForGpu(size_t numHaloParts, TauschHaloSpec *haloSpecs);

    /*!
     * Set the info about all remote CPU halos that live on the GPU and that are needed by the CPU of this partition.  Note that the
     * <i>remoteMpiRank</i> field of TauschHaloSpec does <b>not</b> need to be set as the data is not sent through MPI!
     * \param numHaloParts
     *  How many different parts there are to the halo.
     * \param haloSpecs
     *  The specification of the different halo parts. This is done using the simple struct TauschHaloSpec, containing variables for all the necessary
     *  entries. %Tausch2D expects the following 4 variables to be set:
     *  variable | description
     *  :-------: | -------
     *   x | The starting x coordinate of the halo region
     *   y | The starting y coordinate of the halo region
     *   width | The width of the halo region
     *   height | The height of the halo region
     */
    void setRemoteHaloInfoCpuForGpu(size_t numHaloParts, TauschHaloSpec *haloSpecs);

    /*!
     * This packs the next buffer to be sent. This has to be called for all buffers before sending the message.
     * \param haloId
     *  The id of the halo region. This is the index of this halo region in the local halo specification provided with setLocalHaloInfoCpuForGpu().
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
     */
    void packSendBufferCpuToGpu(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region);

    /*!
     * Overloaded function, packs the full region of the specified halo area.
     * \param haloId
     *  The id of the halo region. This is the index of this halo region in the local halo specification provided with setLocalHaloInfoCpuForGpu().
     * \param bufferId
     *  The id of the buffer. The order of the buffers will be preserved, i.e., packing buffer with id 1 required unpacking that buffer with id 1.
     *  The numbering of the buffers has to start with 0!
     * \param buf
     *  The buffer from which the data is to be extracted according to the local halo specification.
     */
    void packSendBufferCpuToGpu(size_t haloId, size_t bufferId, buf_t *buf);

    /*!
     * Sends off the send buffer for the specified halo region. This does <b>NOT</b> use MPI, but takes advantage of shared memory.
     * \param haloId
     *  The id of the halo region. This is the index of this halo region in the local halo specification provided with setLocalHaloInfoCpuForGpu().
     * \param msgtag
     *  The tag for the current message. This works equivalently to an MPI tag, the corresponding receive has to be called with the same msgtag.
     */
    void sendCpuToGpu(size_t haloId, int msgtag);

    /*!
     * Makes sure that writing the remote halo data to shared memory has completed for the specified halo id. It does not do anything with
     * that message!
     * \param haloId
     *  The id of the halo region. This is the index of this halo region in the remote halo specification provided with setRemoteHaloInfoCpuForGpu().
     * \param msgtag
     *  The tag for the current message. This works equivalently to an MPI tag, the corresponding receive has to be called with the same msgtag.
     */
    void recvGpuToCpu(size_t haloId, int msgtag);

    /*!
     * This unpacks the next halo from the data in shared memory into the provided buffer. This has to be called for all buffers.
     * \param haloId
     *  The id of the halo region. This is the index of this halo region in the remote halo specification provided with setRemoteHaloInfoCpuForGpu().
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
     */
    void unpackRecvBufferGpuToCpu(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region);

    /*!
     * Overloaded function, unpacks the full region of the specified halo area.
     * \param haloId
     *  The id of the halo region. This is the index of this halo region in the remote halo specification provided with setRemoteHaloInfoCpuForGpu().
     * \param bufferId
     *  The id of the buffer. The order of the buffers will be preserved, i.e., packing buffer with id 1 required unpacking that buffer with id 1.
     *  The numbering of the buffers has to start with 0!
     * \param[out] buf
     *  The buffer to which the extracted data is to be written to according to the remote halo specification
     */
    void unpackRecvBufferGpuToCpu(size_t haloId, size_t bufferId, buf_t *buf);

    ///@}

    /*! \name Public Member Functions (GPU -> CPU)
     * All member functions about communicating halo data between the CPU and GPU - GPU side of things. <b>Note:</b> These are only available if
     * %Tausch2D was compiled with OpenCL support!
     */
    //!@{

    /*!
     * Set the info about all local GPU halos that need to be sent to the surrounding CPU partition. Note that the <i>remoteMpiRank</i> field of TauschHaloSpec
     * does <b>not</b> need to be set as the data is not sent through MPI!
     * \param numHaloParts
     *  How many different parts there are to the halo.
     * \param haloSpecs
     *  The specification of the different halo parts. This is done using the simple struct TauschHaloSpec, containing variables for all the necessary
     *  entries. %Tausch2D expects the following 4 variables to be set:
     *  variable | description
     *  :-------: | -------
     *   x | The starting x coordinate of the halo region
     *   y | The starting y coordinate of the halo region
     *   width | The width of the halo region
     *   height | The height of the halo region
     */
    void setLocalHaloInfoGpu(size_t numHaloParts, TauschHaloSpec *haloSpecs);

    /*!
     * Set the info about all remote CPU halos that live on the GPU and that are needed by the CPU of this partition.  Note that the
     * <i>remoteMpiRank</i> field of TauschHaloSpec does <b>not</b> need to be set as the data is not sent through MPI!
     * \param numHaloParts
     *  How many different parts there are to the halo.
     * \param haloSpecs
     *  The specification of the different halo parts. This is done using the simple struct TauschHaloSpec, containing variables for all the necessary
     *  entries. %Tausch2D expects the following 4 variables to be set:
     *  variable | description
     *  :-------: | -------
     *   x | The starting x coordinate of the halo region
     *   y | The starting y coordinate of the halo region
     *   width | The width of the halo region
     *   height | The height of the halo region
     */
    void setRemoteHaloInfoGpu(size_t numHaloParts, TauschHaloSpec *haloSpecs);

    /*!
     * This packs the next buffer to be sent. This has to be called for all buffers before sending the message.
     * \param haloId
     *  The id of the halo region. This is the index of this halo region in the local halo specification provided with setLocalHaloInfoGpu().
     * \param bufferId
     *  The id of the buffer. The order of the buffers will be preserved, i.e., packing buffer with id 1 required unpacking that buffer with id 1.
     *  The numbering of the buffers has to start with 0!
     * \param buf
     *  The buffer from which the data is to be extracted according to the local halo specification.
     */
    void packSendBufferGpuToCpu(size_t haloId, size_t bufferId, cl::Buffer buf);

    /*!
     * Sends off the send buffer for the specified halo region. This does <b>NOT</b> use MPI, but takes advantage of shared memory.
     * \param haloId
     *  The id of the halo region. This is the index of this halo region in the local halo specification provided with setLocalHaloInfoGpu().
     * \param msgtag
     *  The tag for the current message. This works equivalently to an MPI tag, the corresponding receive has to be called with the same msgtag.
     */
    void sendGpuToCpu(size_t haloId, int msgtag);
    /*!
     * Makes sure that writing the remote halo data to shared memory has completed for the specified halo id. It does not do anything with
     * that message!
     * \param haloId
     *  The id of the halo region. This is the index of this halo region in the remote halo specification provided with setRemoteHaloInfoGpu().
     * \param msgtag
     *  The tag for the current message. This works equivalently to an MPI tag, the corresponding receive has to be called with the same msgtag.
     */
    void recvCpuToGpu(size_t haloId, int msgtag);
    /*!
     * This unpacks the next halo from the data in shared memory into the provided buffer. This has to be called for all buffers.
     * \param haloId
     *  The id of the halo region. This is the index of this halo region in the remote halo specification provided with setRemoteHaloInfoGpu().
     * \param bufferId
     *  The id of the buffer. The order of the buffers will be preserved, i.e., packing buffer with id 1 required unpacking that buffer with id 1.
     *  The numbering of the buffers has to start with 0!
     * \param[out] buf
     *  The buffer to which the extracted data is to be written to according to the remote halo specification
     */
    void unpackRecvBufferCpuToGpu(size_t haloId, size_t bufferId, cl::Buffer buf);
    //!@}

#endif

private:

    MPI_Comm TAUSCH_COMM;
    int mpiRank, mpiSize;

    size_t localHaloNumParts;
    TauschHaloSpec *localHaloSpecs;
    size_t remoteHaloNumParts;
    TauschHaloSpec *remoteHaloSpecs;

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


#endif // TAUSCH1D_H
