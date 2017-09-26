/*!
 * \file
 * \author  Lukas Spies <LSpies@illinois.edu>
 * \version 1.0
 *
 * \brief
 *  Virtual API, allowing runtime choice of 1D, 2D or 3D version. Here are also some enums and structs defined.
 *
 *  Virtual API, allowing runtime choice of 1D, 2D or 3D version. For more details on the implementation of the various functions in the API refer to
 * the respective documentation of Tausch1D, Tausch2D, or Tausch3D.
 *
 *  This header file also defines some enums and structs to be used by and with Tausch.
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
 * These are the three dimensions that can be used with Tausch, providing better clarity as to which array entry is which dimension: X, Y, Z.
 */
enum TauschDimensions { TAUSCH_X, TAUSCH_Y, TAUSCH_Z };

/*!
 * These values specify which device is commication with which other device. They need to be specified for most calls to Tausch functions.
 * Valid combinations are:
 *
 * Communication | flags | shorthand
 * :----------: | ---------------- | -------------
 * CPU with CPU | TAUSCH_CPU\|TAUSCH_WITHCPU | TAUSCH_CwC
 * CPU with GPU | TAUSCH_CPU\|TAUSCH_WITHGPU | TAUSCH_CwG
 * GPU with CPU | TAUSCH_GPU\|TAUSCH_WITHCPU | TAUSCH_GwC
 */
enum TauschDeviceDirection {
    TAUSCH_CPU = 1,
    TAUSCH_GPU = 2,
    TAUSCH_WITHCPU = 4,
    TAUSCH_WITHGPU = 8,
    TAUSCH_CwC = 5,
    TAUSCH_CwG = 9,
    TAUSCH_GwC = 6
};

/*!
 * \cond DoxygenHideThis
 */
inline TauschDeviceDirection operator|(TauschDeviceDirection a, TauschDeviceDirection b) {
    return static_cast<TauschDeviceDirection>(static_cast<int>(a) | static_cast<int>(b));
}
/*!
 * \endcond
 */

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
    TauschPackRegion() : x(0), y(0), z(0),
                       width(0), height(0), depth(0) {}
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
    virtual void setLocalHaloInfo(TauschDeviceDirection flags, size_t numHaloParts, TauschHaloSpec *haloSpecs) = 0;
    virtual void setRemoteHaloInfo(TauschDeviceDirection flags, size_t numHaloParts, TauschHaloSpec *haloSpecs) = 0;
    virtual void postReceive(TauschDeviceDirection flags, size_t haloId, int mpitag = -1) = 0;
    virtual void postAllReceives(TauschDeviceDirection flags, int *mpitag = nullptr) = 0;
    virtual void packSendBuffer(TauschDeviceDirection flags, size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region) = 0;
    virtual void packSendBuffer(TauschDeviceDirection flags, size_t haloId, size_t bufferId, buf_t *buf) = 0;
#ifdef TAUSCH_OPENCL
    virtual void packSendBuffer(TauschDeviceDirection flags, size_t haloId, size_t bufferId, cl::Buffer buf) = 0;
#endif
    virtual void send(TauschDeviceDirection flags, size_t haloId, int mpitag = -1) = 0;
    virtual void recv(TauschDeviceDirection flags, size_t haloId) = 0;
    virtual void unpackRecvBuffer(TauschDeviceDirection flags, size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region) = 0;
    virtual void unpackRecvBuffer(TauschDeviceDirection flags, size_t haloId, size_t bufferId, buf_t *buf) = 0;
#ifdef TAUSCH_OPENCL
    virtual void unpackRecvBuffer(TauschDeviceDirection flags, size_t haloId, size_t bufferId, cl::Buffer buf) = 0;
#endif
    virtual void packAndSend(TauschDeviceDirection flags, size_t haloId, buf_t *buf, TauschPackRegion region, int mpitag = -1) = 0;
    virtual void packAndSend(TauschDeviceDirection flags, size_t haloId, buf_t *buf, int mpitag = -1) = 0;
#ifdef TAUSCH_OPENCL
    virtual void packAndSend(TauschDeviceDirection flags, size_t haloId, cl::Buffer buf, int mpitag = -1) = 0;
#endif
    virtual void recvAndUnpack(TauschDeviceDirection flags, size_t haloId, buf_t *buf, TauschPackRegion region) = 0;
    virtual void recvAndUnpack(TauschDeviceDirection flags, size_t haloId, buf_t *buf) = 0;
#ifdef TAUSCH_OPENCL
    virtual void recvAndUnpack(TauschDeviceDirection flags, size_t haloId, cl::Buffer buf) = 0;
#endif

    virtual TauschPackRegion createFilledPackRegion(size_t x, size_t width) = 0;
    virtual TauschPackRegion createFilledPackRegion(size_t x, size_t y, size_t width, size_t height) = 0;
    virtual TauschPackRegion createFilledPackRegion(size_t x, size_t y, size_t z, size_t width, size_t height, size_t depth) = 0;

    virtual TauschHaloSpec createFilledHaloSpec(size_t bufferWidth, size_t haloX, size_t haloWidth, int remoteMpiRank) = 0;
    virtual TauschHaloSpec createFilledHaloSpec(size_t bufferWidth, size_t bufferHeight, size_t haloX, size_t haloY,
                                                size_t haloWidth, size_t haloHeight, int remoteMpiRank) = 0;
    virtual TauschHaloSpec createFilledHaloSpec(size_t bufferWidth, size_t bufferHeight, size_t bufferDepth, size_t haloX, size_t haloY, size_t haloZ,
                                                size_t haloWidth, size_t haloHeight, size_t haloDepth, int remoteMpiRank) = 0;

#ifdef TAUSCH_OPENCL

    virtual void enableOpenCL(bool blockingSyncCpuGpu, int clLocalWorkgroupSize, bool giveOpenCLDeviceName, bool showOpenCLBuildLog) = 0;
    virtual void enableOpenCL(cl::Device cl_defaultDevice, cl::Context cl_context, cl::CommandQueue cl_queue, bool blockingSyncCpuGpu,
                              int clLocalWorkgroupSize, bool showOpenCLBuildLog) = 0;
    virtual cl::Context getOpenCLContext() = 0;
    virtual cl::CommandQueue getOpenCLQueue() = 0;

#endif

};

#endif // TAUSCHBASE_H
