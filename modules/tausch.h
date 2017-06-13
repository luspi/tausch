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

/*!
 * A struct simplifying the specification of halo regions.
 */
struct TauschHaloSpec {
    TauschHaloSpec() : x(0), y(0), z(0), width(0), height(0), depth(0), remoteMpiRank(0) {}
    /*!
     * The starting x coordinate of the halo region.
     */
    size_t x;
    /*!
     * The starting y coordinate of the halo region.
     */
    size_t y;
    /*!
     * The starting z coordinate of the halo region.
     */
    size_t z;
    /*!
     * The width of the halo region.
     */
    size_t width;
    /*!
     * The height of the halo region.
     */
    size_t height;
    /*!
     * The depth of the halo region.
     */
    size_t depth;
    /*!
     * The remote MPI rank associated with this halo region. This is either the sending or receiving MPI rank, depending on if the halo region
     * specifies a local or remote halo.
     */
    int remoteMpiRank;
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
    virtual void postReceiveCpu(size_t id, int mpitag = -1) = 0;
    virtual void postAllReceivesCpu(int *mpitag = nullptr) = 0;
    virtual void packNextSendBufferCpu(size_t id, buf_t *buf) = 0;
    virtual void sendCpu(size_t id, int mpitag = -1) = 0;
    virtual void recvCpu(size_t id) = 0;
    virtual void unpackNextRecvBufferCpu(size_t id, buf_t *buf) = 0;
    virtual void packAndSendCpu(size_t id, buf_t *buf, int mpitag = -1) = 0;
    virtual void recvAndUnpackCpu(size_t id, buf_t *buf) = 0;

};

#endif // TAUSCHBASE_H
