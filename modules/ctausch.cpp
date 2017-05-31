#include "../tausch.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "ctausch.h"

CTausch *tausch_new(int *localDim, int *haloWidth, int numBuffers, int valuesPerPoint, MPI_Comm comm, TAUSCH_VERSION version) {
    Tausch<double> *t;
    if(version == TAUSCH_1D_VERSION)
        t = new Tausch1D<double>(localDim, haloWidth, numBuffers, valuesPerPoint, comm);
    else if(version == TAUSCH_2D_VERSION)
        t = new Tausch2D<double>(localDim, haloWidth, numBuffers, valuesPerPoint, comm);
//    else if(version == TAUSCH_3D_VERSION)
//        t = new Tausch3D<double>(localDim, haloWidth, numBuffers, valuesPerPoint, comm);
    return reinterpret_cast<CTausch*>(t);
}

void tausch_delete(CTausch *tC) {
    Tausch<double> *t = reinterpret_cast<Tausch<double>*>(tC);
    delete t;
}

void tausch_setCpuLocalHaloInfo(CTausch *tC, int numHaloParts, int **haloSpecs) {
    Tausch<double> *t = reinterpret_cast<Tausch<double>*>(tC);
    t->setCpuLocalHaloInfo(numHaloParts, haloSpecs);
}

void tausch_setCpuRemoteHaloInfo(CTausch *tC, int numHaloParts, int **haloSpecs) {
    Tausch<double> *t = reinterpret_cast<Tausch<double>*>(tC);
    t->setCpuRemoteHaloInfo(numHaloParts, haloSpecs);
}

void tausch_postMpiReceives(CTausch *tC) {
    Tausch<double> *t = reinterpret_cast<Tausch<double>*>(tC);
    t->postMpiReceives();
}

void tausch_packNextSendBuffer(CTausch *tC, int id, double *buf) {
    Tausch<double> *t = reinterpret_cast<Tausch<double>*>(tC);
    t->packNextSendBuffer(id, buf);
}

void tausch_send(CTausch *tC, int id) {
    Tausch<double> *t = reinterpret_cast<Tausch<double>*>(tC);
    t->send(id);
}

void tausch_recv(CTausch *tC, int id) {
    Tausch<double> *t = reinterpret_cast<Tausch<double>*>(tC);
    t->recv(id);
}

void tausch_unpackNextRecvBuffer(CTausch *tC, int id, double *buf) {
    Tausch<double> *t = reinterpret_cast<Tausch<double>*>(tC);
    t->unpackNextRecvBuffer(id, buf);
}

void tausch_packAndSend(CTausch *tC, int id, double *buf) {
    Tausch<double> *t = reinterpret_cast<Tausch<double>*>(tC);
    t->packAndSend(id, buf);
}

void tausch_recvAndUnpack(CTausch *tC, int id, double *buf) {
    Tausch<double> *t = reinterpret_cast<Tausch<double>*>(tC);
    t->recvAndUnpack(id, buf);
}


#ifdef __cplusplus
}
#endif

