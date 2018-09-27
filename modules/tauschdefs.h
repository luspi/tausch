#ifndef TAUSCHDEFS_H
#define TAUSCHDEFS_H

#include <cstddef>
#include <vector>

// either one of tausch_opencl_yes.h or tausch_opencl_no.h will ne copied to this file when configuring the project with cmake
// This takes care of making sure TAUSCH_OPENCL is defined (or not).
#include "tausch_opencl.h"

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
    TAUSCH_GwC = 6,
    TAUSCH_GwG = 10
};

/*!
 * A struct simplifying the specification of halo regions.
 */
struct TauschHaloSpec {
    TauschHaloSpec() : bufferWidth(0), bufferHeight(0), bufferDepth(0),
                       haloX(0), haloY(0), haloZ(0),
                       haloWidth(0), haloHeight(0), haloDepth(0),
                       remoteMpiRank(0) {}

    /*!
     * The halo indices in the actual buffer. This in addition to remoteMpiRank is enough to specify a full halo region.
     */
    std::vector<size_t> haloIndicesInBuffer;
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
    TauschPackRegion() : startAtIndex(0), endAtIndex(0),
                         x(0), y(0), z(0),
                         width(0), height(0), depth(0) {}
    /*!
     * If specified, this takes the passed on list of indices and packs starting at this position, up to (but not including) endAtIndex.
     * The (region)x/y/z parameters are not used in that case and do not need to be specified!
     */
    size_t startAtIndex;
    /*!
     * If specified, this takes the passed on list of indices and packs starting at startAtIndex, up to (but not including) this value.
     * The (region)x/y/z parameters are not used in that case and do not need to be specified!
     */
    size_t endAtIndex;
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

#endif // TAUSCHDEFS_H
