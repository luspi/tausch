/*!
 * \file
 * \author  Lukas Spies <LSpies@illinois.edu>
 * \version 1.0
 *
 * \brief
 *  Multi-dimensional API, providing access to all versions of communications for all 3 dimensions. Here are also some enums and structs defined.
 *
 *  Multi-dimensional API, providing access to all versions of communications for all 3 dimensions. For more details on the implementation of the
 *  various functions in the API refer to the respective documentation of Tausch1D, Tausch2D, or Tausch3D.
 *
 *  This header file also defines some enums and structs to be used by and with Tausch.
 */
#ifndef TAUSCH_H
#define TAUSCH_H

#ifdef TAUSCH_OPENCL
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#endif

#include "tausch1d.h"
#include "tausch2d.h"
#include "tausch3d.h"


template <class buf_t>
class Tausch {

public:
    Tausch(MPI_Datatype mpiDataType, size_t numBuffers = 1, size_t *valuesPerPointPerBuffer = NULL, MPI_Comm comm = MPI_COMM_WORLD) {
        tausch1 = new Tausch1D<buf_t>(mpiDataType, numBuffers, valuesPerPointPerBuffer, comm);
        tausch2 = new Tausch2D<buf_t>(mpiDataType, numBuffers, valuesPerPointPerBuffer, comm);
        tausch3 = new Tausch3D<buf_t>(mpiDataType, numBuffers, valuesPerPointPerBuffer, comm);
    }
    ~Tausch() {
        delete tausch1;
        delete tausch2;
        delete tausch3;
    }

    int addLocalHaloInfo1D_CwC(TauschHaloSpec haloSpec) { return tausch1->addLocalHaloInfoCwC(haloSpec); }
    int addLocalHaloInfo2D_CwC(TauschHaloSpec haloSpec) { return tausch2->addLocalHaloInfoCwC(haloSpec); }
    int addLocalHaloInfo3D_CwC(TauschHaloSpec haloSpec) { return tausch3->addLocalHaloInfoCwC(haloSpec); }
#ifdef TAUSCH_OPENCL
    void setLocalHaloInfo1D_CwG(size_t numHaloParts, TauschHaloSpec *haloSpecs) { tausch1->setLocalHaloInfoCwG(numHaloParts, haloSpecs); }
    void setLocalHaloInfo2D_CwG(size_t numHaloParts, TauschHaloSpec *haloSpecs) { tausch2->setLocalHaloInfoCwG(numHaloParts, haloSpecs); }
    void setLocalHaloInfo3D_CwG(size_t numHaloParts, TauschHaloSpec *haloSpecs) { tausch3->setLocalHaloInfoCwG(numHaloParts, haloSpecs); }

    void setLocalHaloInfo1D_GwC(size_t numHaloParts, TauschHaloSpec *haloSpecs) { tausch1->setLocalHaloInfoGwC(numHaloParts, haloSpecs); }
    void setLocalHaloInfo2D_GwC(size_t numHaloParts, TauschHaloSpec *haloSpecs) { tausch2->setLocalHaloInfoGwC(numHaloParts, haloSpecs); }
    void setLocalHaloInfo3D_GwC(size_t numHaloParts, TauschHaloSpec *haloSpecs) { tausch3->setLocalHaloInfoGwC(numHaloParts, haloSpecs); }

//    void setLocalHaloInfo1D_GwG(size_t numHaloParts, TauschHaloSpec *haloSpecs);
    void setLocalHaloInfo2D_GwG(size_t numHaloParts, TauschHaloSpec *haloSpecs) { tausch2->setLocalHaloInfoGwG(numHaloParts, haloSpecs); }
//    void setLocalHaloInfo3D_GwG(size_t numHaloParts, TauschHaloSpec *haloSpecs);
#endif

    void delLocalHaloInfo1D_CwC(size_t haloId) { tausch1->delLocalHaloInfoCwC(haloId); }
    void delLocalHaloInfo2D_CwC(size_t haloId) { tausch2->delLocalHaloInfoCwC(haloId); }
    void delLocalHaloInfo3D_CwC(size_t haloId) { tausch3->delLocalHaloInfoCwC(haloId); }

    int addRemoteHaloInfo1D_CwC(TauschHaloSpec haloSpec) { return tausch1->addRemoteHaloInfoCwC(haloSpec); }
    int addRemoteHaloInfo2D_CwC(TauschHaloSpec haloSpec) { return tausch2->addRemoteHaloInfoCwC(haloSpec); }
    int addRemoteHaloInfo3D_CwC(TauschHaloSpec haloSpec) { return tausch3->addRemoteHaloInfoCwC(haloSpec); }
#ifdef TAUSCH_OPENCL
    void setRemoteHaloInfo1D_CwG(size_t numHaloParts, TauschHaloSpec *haloSpecs) { tausch1->setRemoteHaloInfoCwG(numHaloParts, haloSpecs); }
    void setRemoteHaloInfo2D_CwG(size_t numHaloParts, TauschHaloSpec *haloSpecs) { tausch2->setRemoteHaloInfoCwG(numHaloParts, haloSpecs); }
    void setRemoteHaloInfo3D_CwG(size_t numHaloParts, TauschHaloSpec *haloSpecs) { tausch3->setRemoteHaloInfoCwG(numHaloParts, haloSpecs); }

    void setRemoteHaloInfo1D_GwC(size_t numHaloParts, TauschHaloSpec *haloSpecs) { tausch1->setRemoteHaloInfoGwC(numHaloParts, haloSpecs); }
    void setRemoteHaloInfo2D_GwC(size_t numHaloParts, TauschHaloSpec *haloSpecs) { tausch2->setRemoteHaloInfoGwC(numHaloParts, haloSpecs); }
    void setRemoteHaloInfo3D_GwC(size_t numHaloParts, TauschHaloSpec *haloSpecs) { tausch3->setRemoteHaloInfoGwC(numHaloParts, haloSpecs); }

//    void setRemoteHaloInfo1D_GwG(size_t numHaloParts, TauschHaloSpec *haloSpecs);
    void setRemoteHaloInfo2D_GwG(size_t numHaloParts, TauschHaloSpec *haloSpecs) { tausch2->setRemoteHaloInfoGwG(numHaloParts, haloSpecs); }
//    void setRemoteHaloInfo3D_GwG(size_t numHaloParts, TauschHaloSpec *haloSpecs);
#endif

    void delRemoteHaloInfo1D_CwC(size_t haloId) { tausch1->delRemoteHaloInfoCwC(haloId); }
    void delRemoteHaloInfo2D_CwC(size_t haloId) { tausch2->delRemoteHaloInfoCwC(haloId); }
    void delRemoteHaloInfo3D_CwC(size_t haloId) { tausch3->delRemoteHaloInfoCwC(haloId); }

    void postReceive1D_CwC(size_t haloId, int mpitag = -1) { tausch1->postReceiveCwC(haloId, mpitag); }
    void postReceive2D_CwC(size_t haloId, int mpitag = -1) { tausch2->postReceiveCwC(haloId, mpitag); }
    void postReceive3D_CwC(size_t haloId, int mpitag = -1) { tausch3->postReceiveCwC(haloId, mpitag); }
#ifdef TAUSCH_OPENCL
    void postReceive1D_CwG(size_t haloId, int msgtag = -1) { tausch1->postReceiveCwG(haloId, msgtag); }
    void postReceive2D_CwG(size_t haloId, int msgtag = -1) { tausch2->postReceiveCwG(haloId, msgtag); }
    void postReceive3D_CwG(size_t haloId, int msgtag = -1) { tausch3->postReceiveCwG(haloId, msgtag); }

    void postReceive1D_GwC(size_t haloId, int msgtag = -1) { tausch1->postReceiveGwC(haloId, msgtag); }
    void postReceive2D_GwC(size_t haloId, int msgtag = -1) { tausch2->postReceiveGwC(haloId, msgtag); }
    void postReceive3D_GwC(size_t haloId, int msgtag = -1) { tausch3->postReceiveGwC(haloId, msgtag); }

//    void postReceive1D_GwG(size_t haloId, int msgtag = -1);
    void postReceive2D_GwG(size_t haloId, int msgtag = -1) { tausch2->postReceiveGwG(haloId, msgtag); }
//    void postReceive3D_GwG(size_t haloId, int msgtag = -1);
#endif

    void postAllReceives1D_CwC(int *mpitag = NULL) { tausch1->postAllReceivesCwC(mpitag); }
    void postAllReceives2D_CwC(int *mpitag = NULL) { tausch2->postAllReceivesCwC(mpitag); }
    void postAllReceives3D_CwC(int *mpitag = NULL) { tausch3->postAllReceivesCwC(mpitag); }
#ifdef TAUSCH_OPENCL
    void postAllReceives1D_CwG(int *msgtag = NULL) { tausch1->postAllReceivesCwG(msgtag); }
    void postAllReceives2D_CwG(int *msgtag = NULL) { tausch2->postAllReceivesCwG(msgtag); }
    void postAllReceives3D_CwG(int *msgtag = NULL) { tausch3->postAllReceivesCwG(msgtag); }

    void postAllReceives1D_GwC(int *msgtag = NULL) { tausch1->postAllReceivesGwC(msgtag); }
    void postAllReceives2D_GwC(int *msgtag = NULL) { tausch2->postAllReceivesGwC(msgtag); }
    void postAllReceives3D_GwC(int *msgtag = NULL) { tausch3->postAllReceivesGwC(msgtag); }

//    void postAllReceives1D_GwG(int *msgtag = NULL);
    void postAllReceives2D_GwG(int *msgtag = NULL) { tausch2->postAllReceivesGwG(msgtag); }
//    void postAllReceives3D_GwG(int *msgtag = NULL);
#endif

    void packSendBuffer1D_CwC(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region) { tausch1->packSendBufferCwC(haloId, bufferId, buf, region); }
    void packSendBuffer2D_CwC(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region) { tausch2->packSendBufferCwC(haloId, bufferId, buf, region); }
    void packSendBuffer3D_CwC(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region) { tausch3->packSendBufferCwC(haloId, bufferId, buf, region); }

    void packSendBuffer1D_CwC(size_t haloId, size_t bufferId, buf_t *buf) { tausch1->packSendBufferCwC(haloId, bufferId, buf); }
    void packSendBuffer2D_CwC(size_t haloId, size_t bufferId, buf_t *buf) { tausch2->packSendBufferCwC(haloId, bufferId, buf); }
    void packSendBuffer3D_CwC(size_t haloId, size_t bufferId, buf_t *buf) { tausch3->packSendBufferCwC(haloId, bufferId, buf); }
#ifdef TAUSCH_OPENCL
    void packSendBuffer1D_CwG(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region) { tausch1->packSendBufferCwG(haloId, bufferId, buf, region); }
    void packSendBuffer2D_CwG(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region) { tausch2->packSendBufferCwG(haloId, bufferId, buf, region); }
    void packSendBuffer3D_CwG(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region) { tausch3->packSendBufferCwG(haloId, bufferId, buf, region); }

    void packSendBuffer1D_CwG(size_t haloId, size_t bufferId, buf_t *buf) { tausch1->packSendBufferCwG(haloId, bufferId, buf); }
    void packSendBuffer2D_CwG(size_t haloId, size_t bufferId, buf_t *buf) { tausch2->packSendBufferCwG(haloId, bufferId, buf); }
    void packSendBuffer3D_CwG(size_t haloId, size_t bufferId, buf_t *buf) { tausch3->packSendBufferCwG(haloId, bufferId, buf); }

    void packSendBuffer1D_GwC(size_t haloId, size_t bufferId, cl::Buffer buf) { tausch1->packSendBufferGwC(haloId, bufferId, buf); }
    void packSendBuffer2D_GwC(size_t haloId, size_t bufferId, cl::Buffer buf) { tausch2->packSendBufferGwC(haloId, bufferId, buf); }
    void packSendBuffer3D_GwC(size_t haloId, size_t bufferId, cl::Buffer buf) { tausch3->packSendBufferGwC(haloId, bufferId, buf); }

//    void packSendBuffer1D_GwG(size_t haloId, size_t bufferId, cl::Buffer buf);
    void packSendBuffer2D_GwG(size_t haloId, size_t bufferId, cl::Buffer buf) { tausch2->packSendBufferGwG(haloId, bufferId, buf); }
//    void packSendBuffer3D_GwG(size_t haloId, size_t bufferId, cl::Buffer buf);
#endif

    void send1D_CwC(size_t haloId, int mpitag = -1) { tausch1->sendCwC(haloId, mpitag); }
    void send2D_CwC(size_t haloId, int mpitag = -1) { tausch2->sendCwC(haloId, mpitag); }
    void send3D_CwC(size_t haloId, int mpitag = -1) { tausch3->sendCwC(haloId, mpitag); }
#ifdef TAUSCH_OPENCL
    void send1D_CwG(size_t haloId, int msgtag) { tausch1->sendCwG(haloId, msgtag); }
    void send2D_CwG(size_t haloId, int msgtag) { tausch2->sendCwG(haloId, msgtag); }
    void send3D_CwG(size_t haloId, int msgtag) { tausch3->sendCwG(haloId, msgtag); }

    void send1D_GwC(size_t haloId, int msgtag) { tausch1->sendGwC(haloId, msgtag); }
    void send2D_GwC(size_t haloId, int msgtag) { tausch2->sendGwC(haloId, msgtag); }
    void send3D_GwC(size_t haloId, int msgtag) { tausch3->sendGwC(haloId, msgtag); }

//    void send1D_GwG(size_t haloId, int msgtag);
    void send2D_GwG(size_t haloId, int msgtag) { tausch2->sendGwG(haloId, msgtag); }
//    void send3D_GwG(size_t haloId, int msgtag);
#endif

    void recv1D_CwC(size_t haloId) { tausch1->recvCwC(haloId); }
    void recv2D_CwC(size_t haloId) { tausch2->recvCwC(haloId); }
    void recv3D_CwC(size_t haloId) { tausch3->recvCwC(haloId); }
#ifdef TAUSCH_OPENCL
    void recv1D_CwG(size_t haloId) { tausch1->recvCwG(haloId); }
    void recv2D_CwG(size_t haloId) { tausch2->recvCwG(haloId); }
    void recv3D_CwG(size_t haloId) { tausch3->recvCwG(haloId); }

    void recv1D_GwC(size_t haloId) { tausch1->recvGwC(haloId); }
    void recv2D_GwC(size_t haloId) { tausch2->recvGwC(haloId); }
    void recv3D_GwC(size_t haloId) { tausch3->recvGwC(haloId); }

//    void recv1D_GwG(size_t haloId);
    void recv2D_GwG(size_t haloId) { tausch2->recvGwG(haloId); }
//    void recv3D_GwG(size_t haloId);
#endif

    void unpackRecvBuffer1D_CwC(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region) { tausch1->unpackRecvBufferCwC(haloId, bufferId, buf, region); }
    void unpackRecvBuffer2D_CwC(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region) { tausch2->unpackRecvBufferCwC(haloId, bufferId, buf, region); }
    void unpackRecvBuffer3D_CwC(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region) { tausch3->unpackRecvBufferCwC(haloId, bufferId, buf, region); }

    void unpackRecvBuffer1D_CwC(size_t haloId, size_t bufferId, buf_t *buf) { tausch1->unpackRecvBufferCwC(haloId, bufferId, buf); }
    void unpackRecvBuffer2D_CwC(size_t haloId, size_t bufferId, buf_t *buf) { tausch2->unpackRecvBufferCwC(haloId, bufferId, buf); }
    void unpackRecvBuffer3D_CwC(size_t haloId, size_t bufferId, buf_t *buf) { tausch3->unpackRecvBufferCwC(haloId, bufferId, buf); }
#ifdef TAUSCH_OPENCL
    void unpackRecvBuffer1D_CwG(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region) { tausch1->unpackRecvBufferCwG(haloId, bufferId, buf, region); }
    void unpackRecvBuffer2D_CwG(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region) { tausch2->unpackRecvBufferCwG(haloId, bufferId, buf, region); }
    void unpackRecvBuffer3D_CwG(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region) { tausch3->unpackRecvBufferCwG(haloId, bufferId, buf, region); }

    void unpackRecvBuffer1D_CwG(size_t haloId, size_t bufferId, buf_t *buf) { tausch1->unpackRecvBufferCwG(haloId, bufferId, buf); }
    void unpackRecvBuffer2D_CwG(size_t haloId, size_t bufferId, buf_t *buf) { tausch2->unpackRecvBufferCwG(haloId, bufferId, buf); }
    void unpackRecvBuffer3D_CwG(size_t haloId, size_t bufferId, buf_t *buf) { tausch3->unpackRecvBufferCwG(haloId, bufferId, buf); }

    void unpackRecvBuffer1D_GwC(size_t haloId, size_t bufferId, cl::Buffer buf) { tausch1->unpackRecvBufferGwC(haloId, bufferId, buf); }
    void unpackRecvBuffer2D_GwC(size_t haloId, size_t bufferId, cl::Buffer buf) { tausch2->unpackRecvBufferGwC(haloId, bufferId, buf); }
    void unpackRecvBuffer3D_GwC(size_t haloId, size_t bufferId, cl::Buffer buf) { tausch3->unpackRecvBufferGwC(haloId, bufferId, buf); }

//    void unpackRecvBuffer1D_GwG(size_t haloId, size_t bufferId, cl::Buffer buf);
    void unpackRecvBuffer2D_GwG(size_t haloId, size_t bufferId, cl::Buffer buf) { tausch2->unpackRecvBufferGwG(haloId, bufferId, buf); }
//    void unpackRecvBuffer3D_GwG(size_t haloId, size_t bufferId, cl::Buffer buf);
#endif

    void packAndSend1D_CwC(size_t haloId, buf_t *buf, TauschPackRegion region, int msgtag = -1) { tausch1->packAndSendCwC(haloId, buf, region, msgtag); }
    void packAndSend2D_CwC(size_t haloId, buf_t *buf, TauschPackRegion region, int msgtag = -1) { tausch2->packAndSendCwC(haloId, buf, region, msgtag); }
    void packAndSend3D_CwC(size_t haloId, buf_t *buf, TauschPackRegion region, int msgtag = -1) { tausch3->packAndSendCwC(haloId, buf, region, msgtag); }

    void packAndSend1D_CwC(size_t haloId, buf_t *buf, int msgtag = -1) { tausch1->packAndSendCwC(haloId, buf, msgtag); }
    void packAndSend2D_CwC(size_t haloId, buf_t *buf, int msgtag = -1) { tausch2->packAndSendCwC(haloId, buf, msgtag); }
    void packAndSend3D_CwC(size_t haloId, buf_t *buf, int msgtag = -1) { tausch3->packAndSendCwC(haloId, buf, msgtag); }

#ifdef TAUSCH_OPENCL
    void packAndSend1D_CwG(size_t haloId, buf_t *buf, TauschPackRegion region, int msgtag = -1) { tausch1->packAndSendCwG(haloId, buf, region, msgtag); }
    void packAndSend2D_CwG(size_t haloId, buf_t *buf, TauschPackRegion region, int msgtag = -1) { tausch2->packAndSendCwG(haloId, buf, region, msgtag); }
    void packAndSend3D_CwG(size_t haloId, buf_t *buf, TauschPackRegion region, int msgtag = -1) { tausch3->packAndSendCwG(haloId, buf, region, msgtag); }

    void packAndSend1D_GwC(size_t haloId, cl::Buffer buf, int msgtag = -1) { tausch1->packAndSendGwC(haloId, buf, msgtag); }
    void packAndSend2D_GwC(size_t haloId, cl::Buffer buf, int msgtag = -1) { tausch2->packAndSendGwC(haloId, buf, msgtag); }
    void packAndSend3D_GwC(size_t haloId, cl::Buffer buf, int msgtag = -1) { tausch3->packAndSendGwC(haloId, buf, msgtag); }

//    void packAndSend1D_GwG(size_t haloId, cl::Buffer buf, int msgtag = -1);
    void packAndSend2D_GwG(size_t haloId, cl::Buffer buf, int msgtag = -1) { tausch2->packAndSendGwG(haloId, buf, msgtag); }
//    void packAndSend3D_GwG(size_t haloId, cl::Buffer buf, int msgtag = -1);
#endif

    void recvAndUnpack1D_CwC(size_t haloId, buf_t *buf, TauschPackRegion region) { tausch1->recvAndUnpackCwC(haloId, buf, region); }
    void recvAndUnpack2D_CwC(size_t haloId, buf_t *buf, TauschPackRegion region) { tausch2->recvAndUnpackCwC(haloId, buf, region); }
    void recvAndUnpack3D_CwC(size_t haloId, buf_t *buf, TauschPackRegion region) { tausch3->recvAndUnpackCwC(haloId, buf, region); }

    void recvAndUnpack1D_CwC(size_t haloId, buf_t *buf) { tausch1->recvAndUnpackCwC(haloId, buf); }
    void recvAndUnpack2D_CwC(size_t haloId, buf_t *buf) { tausch2->recvAndUnpackCwC(haloId, buf); }
    void recvAndUnpack3D_CwC(size_t haloId, buf_t *buf) { tausch3->recvAndUnpackCwC(haloId, buf); }
#ifdef TAUSCH_OPENCL
    void recvAndUnpack1D_CwG(size_t haloId, buf_t *buf, TauschPackRegion region) { tausch1->recvAndUnpackCwG(haloId, buf, region); }
    void recvAndUnpack2D_CwG(size_t haloId, buf_t *buf, TauschPackRegion region) { tausch2->recvAndUnpackCwG(haloId, buf, region); }
    void recvAndUnpack3D_CwG(size_t haloId, buf_t *buf, TauschPackRegion region) { tausch3->recvAndUnpackCwG(haloId, buf, region); }

    void recvAndUnpack1D_GwC(size_t haloId, cl::Buffer buf) { tausch1->recvAndUnpackGwC(haloId, buf); }
    void recvAndUnpack2D_GwC(size_t haloId, cl::Buffer buf) { tausch2->recvAndUnpackGwC(haloId, buf); }
    void recvAndUnpack3D_GwC(size_t haloId, cl::Buffer buf) { tausch3->recvAndUnpackGwC(haloId, buf); }

//    void recvAndUnpack1D_GwG(size_t haloId, cl::Buffer buf);
    void recvAndUnpack2D_GwG(size_t haloId, cl::Buffer buf) { tausch2->recvAndUnpackGwG(haloId, buf); }
//    void recvAndUnpack3D_GwG(size_t haloId, cl::Buffer buf);
#endif

#ifdef TAUSCH_OPENCL

    void enableOpenCL1D(bool blockingSyncCpuGpu, int clLocalWorkgroupSize, bool giveOpenCLDeviceName, bool showOpenCLBuildLog) {
        tausch1->enableOpenCL(blockingSyncCpuGpu, clLocalWorkgroupSize, giveOpenCLDeviceName, showOpenCLBuildLog);
    }
    void enableOpenCL2D(bool blockingSyncCpuGpu, int clLocalWorkgroupSize, bool giveOpenCLDeviceName, bool showOpenCLBuildLog) {
        tausch2->enableOpenCL(blockingSyncCpuGpu, clLocalWorkgroupSize, giveOpenCLDeviceName, showOpenCLBuildLog);
    }
    void enableOpenCL3D(bool blockingSyncCpuGpu, int clLocalWorkgroupSize, bool giveOpenCLDeviceName, bool showOpenCLBuildLog) {
        tausch3->enableOpenCL(blockingSyncCpuGpu, clLocalWorkgroupSize, giveOpenCLDeviceName, showOpenCLBuildLog);
    }

    void enableOpenCL1D(cl::Device cl_defaultDevice, cl::Context cl_context, cl::CommandQueue cl_queue, bool blockingSyncCpuGpu,
                         int clLocalWorkgroupSize, bool showOpenCLBuildLog) {
        tausch1->enableOpenCL(cl_defaultDevice, cl_context, cl_queue, blockingSyncCpuGpu, clLocalWorkgroupSize, showOpenCLBuildLog);
    }
    void enableOpenCL2D(cl::Device cl_defaultDevice, cl::Context cl_context, cl::CommandQueue cl_queue, bool blockingSyncCpuGpu,
                         int clLocalWorkgroupSize, bool showOpenCLBuildLog) {
        tausch2->enableOpenCL(cl_defaultDevice, cl_context, cl_queue, blockingSyncCpuGpu, clLocalWorkgroupSize, showOpenCLBuildLog);
    }
    void enableOpenCL3D(cl::Device cl_defaultDevice, cl::Context cl_context, cl::CommandQueue cl_queue, bool blockingSyncCpuGpu,
                         int clLocalWorkgroupSize, bool showOpenCLBuildLog) {
        tausch3->enableOpenCL(cl_defaultDevice, cl_context, cl_queue, blockingSyncCpuGpu, clLocalWorkgroupSize, showOpenCLBuildLog);
    }

    cl::Context getOpenCLContext1D() { return tausch1->getOpenCLContext(); }
    cl::Context getOpenCLContext2D() { return tausch2->getOpenCLContext(); }
    cl::Context getOpenCLContext3D() { return tausch3->getOpenCLContext(); }

    cl::CommandQueue getOpenCLQueue1D() { return tausch1->getOpenCLQueue(); }
    cl::CommandQueue getOpenCLQueue2D() { return tausch2->getOpenCLQueue(); }
    cl::CommandQueue getOpenCLQueue3D() { return tausch3->getOpenCLQueue(); }

#endif

    TauschPackRegion createFilledPackRegion1D(size_t x, size_t width) { return tausch1->createFilledPackRegion(x, width); }
    TauschPackRegion createFilledPackRegion2D(size_t x, size_t y, size_t width, size_t height) { return tausch2->createFilledPackRegion(x, y, width, height); }
    TauschPackRegion createFilledPackRegion2D(size_t startAtIndex, size_t endAtIndex) { return tausch2->createFilledPackRegion(startAtIndex, endAtIndex); }
    TauschPackRegion createFilledPackRegion3D(size_t x, size_t y, size_t z, size_t width, size_t height, size_t depth) { return tausch3->createFilledPackRegion(x, y, z, width, height, depth); }
    TauschPackRegion createFilledPackRegion3D(size_t startAtIndex, size_t endAtIndex) { return tausch3->createFilledPackRegion(startAtIndex, endAtIndex); }

private:
    Tausch1D<buf_t> *tausch1;
    Tausch2D<buf_t> *tausch2;
    Tausch3D<buf_t> *tausch3;

};



#endif // TAUSCH_H
