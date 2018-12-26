#ifndef TAUSCHDEFS_H
#define TAUSCHDEFS_H

#include <cstdlib>

struct TauschHaloRegion {
    TauschHaloRegion() : bufferWidth(0), bufferHeight(0), bufferDepth(0),
                       haloX(0), haloY(0), haloZ(0),
                       haloWidth(0), haloHeight(0), haloDepth(0), remoteMpiRank(-1) {}
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
     * Optional: Receiving/Sending MPI rank
     */
    int remoteMpiRank;
};

#endif
