/*!
 * \file
 * \author  Lukas Spies <LSpies@illinois.edu>
 * \version 1.0
 *
 * \brief
 *  Virtual API, allowing runtime choice of 1D, 2D or 3D version.
 *
 *  Virtual API, allowing runtime choice of 1D, 2D or 3D version. For more details on the implementation of the various functions in the API refer to
 * the respective documentation of Tausch1D, Tausch2D, or Tausch3D.
 */
#ifndef TAUSCHBASE_H
#define TAUSCHBASE_H

#include <cstddef>

// either one of tausch_opencl_yes.h or tausch_opencl_no.h will ne copied to this file when configuring the project with cmake
// This takes care of making sure TAUSCH_OPENCL is defined (or not).
#include "tausch_opencl.h"

#ifdef TAUSCH_OPENCL
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#endif

/*!
 * A struct simplifying the specification of halo regions.
 */
struct TauschHaloSpec {
    TauschHaloSpec() : bufferWidth(0), bufferHeight(0), bufferDepth(0),
                       haloX(0), haloY(0), haloZ(0),
                       haloWidth(0), haloHeight(0), haloDepth(0),
                       remoteMpiRank(0) {}
    /*!
     * The width of the underlying buffer.
     */
    size_t bufferWidth;
    /*!
     * The height of the underlying buffer.
     */
    size_t bufferHeight;
    /*!
     * The depth of the underlying buffer.
     */
    size_t bufferDepth;
    /*!
     * The starting x coordinate of the halo region.
     */
    size_t haloX;
    /*!
     * The starting y coordinate of the halo region.
     */
    size_t haloY;
    /*!
     * The starting z coordinate of the halo region.
     */
    size_t haloZ;
    /*!
     * The width of the halo region.
     */
    size_t haloWidth;
    /*!
     * The height of the halo region.
     */
    size_t haloHeight;
    /*!
     * The depth of the halo region.
     */
    size_t haloDepth;
    /*!
     * The remote MPI rank associated with this halo region. This is either the sending or receiving MPI rank, depending on if the halo region
     * specifies a local or remote halo.
     */
    int remoteMpiRank;
};

/*!
 * A struct for specifying which region of a halo area to pack.
 */
struct TauschPackRegion {
    /*!
     * The starting x coordinate of the region, relative to the halo area.
     */
    size_t x;
    /*!
     * The starting y coordinate of the region, relative to the halo area.
     */
    size_t y;
    /*!
     * The starting z coordinate of the region, relative to the halo area.
     */
    size_t z;
    /*!
     * The width of the region.
     */
    size_t width;
    /*!
     * The height of the region.
     */
    size_t height;
    /*!
     * The depth of the region.
     */
    size_t depth;
};

/*!
 *
 * \internal
 *
 */
template <class buf_t>
class Tausch {
public:
    virtual ~Tausch() {}
    virtual void setLocalHaloInfoCpu(size_t numHaloParts, TauschHaloSpec *haloSpecs) = 0;
    virtual void setRemoteHaloInfoCpu(size_t numHaloParts, TauschHaloSpec *haloSpecs) = 0;
    virtual void postReceiveCpu(size_t haloId, int mpitag = -1) = 0;
    virtual void postAllReceivesCpu(int *mpitag = nullptr) = 0;
    virtual void packSendBufferCpu(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region) = 0;
    virtual void packSendBufferCpu(size_t haloId, size_t bufferId, buf_t *buf) = 0;
    virtual void sendCpu(size_t haloId, int mpitag = -1) = 0;
    virtual void recvCpu(size_t haloId) = 0;
    virtual void unpackRecvBufferCpu(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region) = 0;
    virtual void unpackRecvBufferCpu(size_t haloId, size_t bufferId, buf_t *buf) = 0;
    virtual void packAndSendCpu(size_t haloId, buf_t *buf, TauschPackRegion region, int mpitag = -1) = 0;
    virtual void packAndSendCpu(size_t haloId, buf_t *buf, int mpitag = -1) = 0;
    virtual void recvAndUnpackCpu(size_t haloId, buf_t *buf, TauschPackRegion region) = 0;
    virtual void recvAndUnpackCpu(size_t haloId, buf_t *buf) = 0;

#ifdef TAUSCH_OPENCL

    virtual void enableOpenCL(bool blockingSyncCpuGpu, int clLocalWorkgroupSize, bool giveOpenCLDeviceName, bool showOpenCLBuildLog) = 0;
    virtual cl::Context getOpenCLContext() = 0;
    virtual cl::CommandQueue getOpenCLQueue() = 0;
    virtual void setLocalHaloInfoCpuForGpu(size_t numHaloParts, TauschHaloSpec *haloSpecs) = 0;
    virtual void setRemoteHaloInfoCpuForGpu(size_t numHaloParts, TauschHaloSpec *haloSpecs) = 0;
    virtual void packSendBufferCpuToGpu(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region) = 0;
    virtual void packSendBufferCpuToGpu(size_t haloId, size_t bufferId, buf_t *buf) = 0;
    virtual void sendCpuToGpu(size_t haloId, int msgtag) = 0;
    virtual void recvGpuToCpu(size_t haloId, int msgtag) = 0;
    virtual void unpackRecvBufferGpuToCpu(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region) = 0;
    virtual void unpackRecvBufferGpuToCpu(size_t haloId, size_t bufferId, buf_t *buf) = 0;
    virtual void setLocalHaloInfoGpu(size_t numHaloParts, TauschHaloSpec *haloSpecs) = 0;
    virtual void setRemoteHaloInfoGpu(size_t numHaloParts, TauschHaloSpec *haloSpecs) = 0;
    virtual void packSendBufferGpuToCpu(size_t haloId, size_t bufferId, cl::Buffer buf) = 0;
    virtual void sendGpuToCpu(size_t haloId, int msgtag) = 0;
    virtual void recvCpuToGpu(size_t haloId, int msgtag) = 0;
    virtual void unpackRecvBufferCpuToGpu(size_t haloId, size_t bufferId, cl::Buffer buf) = 0;

#endif

};

#endif // TAUSCHBASE_H
