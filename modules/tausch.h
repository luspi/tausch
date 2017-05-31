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

#ifdef TAUSCH_OPENCL
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#endif

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
template <class real_t>
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
    virtual void setLocalHaloInfoCpu(int numHaloParts, int **haloSpecs) = 0;
    /*!
     * Virtual member pointing to respective function of Tausch1D, Tausch2D, or Tausch3D. More details can be found in the documentation for the
     * respective class.
     * \param numHaloParts
     *  Into how many regions the remote halo is divided.
     * \param haloSpecs
     *  The specifications of the remote halo regions.
     */
    virtual void setRemoteHaloInfoCpu(int numHaloParts, int **haloSpecs) = 0;
    /*!
     * Virtual member pointing to respective function of Tausch1D, Tausch2D, or Tausch3D. More details can be found in the documentation for the
     * respective class.
     */
    virtual void postReceivesCpu() = 0;
    /*!
     * Virtual member pointing to respective function of Tausch1D, Tausch2D, or Tausch3D. More details can be found in the documentation for the
     * respective class.
     * \param id
     *  The id of the local halo region.
     * \param buf
     *  The buffer from where to extract the halo data from.
     */
    virtual void packNextSendBufferCpu(int id, real_t *buf) = 0;
    /*!
     * Virtual member pointing to respective function of Tausch1D, Tausch2D, or Tausch3D. More details can be found in the documentation for the
     * respective class.
     * \param id
     *  The id of the local halo region.
     */
    virtual void sendCpu(int id) = 0;
    /*!
     * Virtual member pointing to respective function of Tausch1D, Tausch2D, or Tausch3D. More details can be found in the documentation for the
     * respective class.
     * \param id
     *  The id of the remote halo region.
     */
    virtual void recvCpu(int id) = 0;
    /*!
     * Virtual member pointing to respective function of Tausch1D, Tausch2D, or Tausch3D. More details can be found in the documentation for the
     * respective class.
     * \param id
     *  The id of the remote halo region.
     * \param buf
     *  The buffer to where to extract the halo data to.
     */
    virtual void unpackNextRecvBufferCpu(int id, real_t *buf) = 0;
    /*!
     * Virtual member pointing to respective function of Tausch1D, Tausch2D, or Tausch3D. More details can be found in the documentation for the
     * respective class.
     * \param id
     *  The id of the local halo region.
     * \param buf
     *  The buffer from where to extract the halo data from.
     */
    virtual void packAndSendCpu(int id, real_t *buf) = 0;
    /*!
     * Virtual member pointing to respective function of Tausch1D, Tausch2D, or Tausch3D. More details can be found in the documentation for the
     * respective class.
     * \param id
     *  The id of the remote halo region.
     * \param buf
     *  The buffer to where to extract the halo data to.
     */
    virtual void recvAndUnpackCpu(int id, real_t *buf) = 0;

};

#endif // TAUSCHBASE_H
