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

#ifdef TAUSCH_OPENCL
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#endif

/*!
 * A struct simplifying the specification of halo regions.
 */
struct TauschHaloSpec {
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
 * \brief
 * Virtual API, allowing runtime choice of 1D, 2D or 3D version.
 *
 * Virtual API, allowing runtime choice of 1D, 2D or 3D version. For more details on the implementation of the various functions in the API refer to
 * the respective documentation of Tausch1D, Tausch2D, or Tausch3D.
 */
template <class buf_t>
class Tausch {
public:
    virtual ~Tausch() {}
    /*!
     * Virtual member pointing to respective function of Tausch1D, Tausch2D, or Tausch3D. More details can be found in the documentation for the
     * respective class.
     * \param numHaloParts
     *  Into how many regions the local halo is divided.
     * \param haloSpecs
     *  The specifications of the local halo regions.
     */
    virtual void setLocalHaloInfoCpu(size_t numHaloParts, TauschHaloSpec *haloSpecs) = 0;
    /*!
     * Virtual member pointing to respective function of Tausch1D, Tausch2D, or Tausch3D. More details can be found in the documentation for the
     * respective class.
     * \param numHaloParts
     *  Into how many regions the remote halo is divided.
     * \param haloSpecs
     *  The specifications of the remote halo regions.
     */
    virtual void setRemoteHaloInfoCpu(size_t numHaloParts, TauschHaloSpec *haloSpecs) = 0;
    /*!
     * Virtual member pointing to respective function of Tausch1D, Tausch2D, or Tausch3D. More details can be found in the documentation for the
     * respective class.
     */
    virtual void postReceiveCpu(size_t id, int mpitag = -1) = 0;
    virtual void postAllReceivesCpu(int *mpitag = nullptr) = 0;
    /*!
     * Virtual member pointing to respective function of Tausch1D, Tausch2D, or Tausch3D. More details can be found in the documentation for the
     * respective class.
     * \param id
     *  The id of the local halo region.
     * \param buf
     *  The buffer from where to extract the halo data from.
     */
    virtual void packNextSendBufferCpu(size_t id, buf_t *buf) = 0;
    /*!
     * Virtual member pointing to respective function of Tausch1D, Tausch2D, or Tausch3D. More details can be found in the documentation for the
     * respective class.
     * \param id
     *  The id of the local halo region.
     */
    virtual void sendCpu(size_t id, int mpitag = -1) = 0;
    /*!
     * Virtual member pointing to respective function of Tausch1D, Tausch2D, or Tausch3D. More details can be found in the documentation for the
     * respective class.
     * \param id
     *  The id of the remote halo region.
     */
    virtual void recvCpu(size_t id) = 0;
    /*!
     * Virtual member pointing to respective function of Tausch1D, Tausch2D, or Tausch3D. More details can be found in the documentation for the
     * respective class.
     * \param id
     *  The id of the remote halo region.
     * \param buf
     *  The buffer to where to extract the halo data to.
     */
    virtual void unpackNextRecvBufferCpu(size_t id, buf_t *buf) = 0;
    /*!
     * Virtual member pointing to respective function of Tausch1D, Tausch2D, or Tausch3D. More details can be found in the documentation for the
     * respective class.
     * \param id
     *  The id of the local halo region.
     * \param buf
     *  The buffer from where to extract the halo data from.
     */
    virtual void packAndSendCpu(size_t id, buf_t *buf, int mpitag = -1) = 0;
    /*!
     * Virtual member pointing to respective function of Tausch1D, Tausch2D, or Tausch3D. More details can be found in the documentation for the
     * respective class.
     * \param id
     *  The id of the remote halo region.
     * \param buf
     *  The buffer to where to extract the halo data to.
     */
    virtual void recvAndUnpackCpu(size_t id, buf_t *buf) = 0;

};

#endif // TAUSCHBASE_H
