#include "../tausch.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "ctausch.h"

CTausch *tausch_new(int *localDim, int *haloWidth, int numBuffers, int valuesPerPoint,
                    MPI_Comm comm, TAUSCH_VERSION version, TAUSCH_DATATYPE datatype) {
    if(version == TAUSCH_1D) {
        if(datatype == TAUSCH_DOUBLE){
            Tausch<double> *t;
            t = new Tausch1D<double>(localDim, haloWidth, numBuffers, valuesPerPoint, comm);
            return reinterpret_cast<CTausch*>(t);
        } else if(datatype == TAUSCH_FLOAT) {
            Tausch<float> *t;
            t = new Tausch1D<float>(localDim, haloWidth, numBuffers, valuesPerPoint, comm);
            return reinterpret_cast<CTausch*>(t);
        } else if(datatype == TAUSCH_INT) {
            Tausch<int> *t;
            t = new Tausch1D<int>(localDim, haloWidth, numBuffers, valuesPerPoint, comm);
            return reinterpret_cast<CTausch*>(t);
        } else if(datatype == TAUSCH_UNSIGNED_INT) {
            Tausch<unsigned int> *t;
            t = new Tausch1D<unsigned int>(localDim, haloWidth, numBuffers, valuesPerPoint, comm);
            return reinterpret_cast<CTausch*>(t);
        } else if(datatype == TAUSCH_LONG) {
            Tausch<long> *t;
            t = new Tausch1D<long>(localDim, haloWidth, numBuffers, valuesPerPoint, comm);
            return reinterpret_cast<CTausch*>(t);
        } else if(datatype == TAUSCH_LONG_LONG) {
            Tausch<long long> *t;
            t = new Tausch1D<long long>(localDim, haloWidth, numBuffers, valuesPerPoint, comm);
            return reinterpret_cast<CTausch*>(t);
        } else if(datatype == TAUSCH_LONG_DOUBLE) {
            Tausch<long double> *t;
            t = new Tausch1D<long double>(localDim, haloWidth, numBuffers, valuesPerPoint, comm);
            return reinterpret_cast<CTausch*>(t);
        } else {
            std::cerr << "ERROR! Invalid data type specified: " << datatype << " - Abort..." << std::endl;
            exit(1);
        }
    } else if(version == TAUSCH_2D) {
        if(datatype == TAUSCH_DOUBLE) {
            Tausch<double> *t;
            t = new Tausch2D<double>(localDim, haloWidth, numBuffers, valuesPerPoint, comm);
            return reinterpret_cast<CTausch*>(t);
        } else if(datatype == TAUSCH_FLOAT) {
            Tausch<float> *t;
            t = new Tausch2D<float>(localDim, haloWidth, numBuffers, valuesPerPoint, comm);
            return reinterpret_cast<CTausch*>(t);
        } else if(datatype == TAUSCH_INT) {
            Tausch<int> *t;
            t = new Tausch2D<int>(localDim, haloWidth, numBuffers, valuesPerPoint, comm);
            return reinterpret_cast<CTausch*>(t);
        } else if(datatype == TAUSCH_UNSIGNED_INT) {
            Tausch<unsigned int> *t;
            t = new Tausch2D<unsigned int>(localDim, haloWidth, numBuffers, valuesPerPoint, comm);
            return reinterpret_cast<CTausch*>(t);
        } else if(datatype == TAUSCH_LONG) {
            Tausch<long> *t;
            t = new Tausch2D<long>(localDim, haloWidth, numBuffers, valuesPerPoint, comm);
            return reinterpret_cast<CTausch*>(t);
        } else if(datatype == TAUSCH_LONG_LONG) {
            Tausch<long long> *t;
            t = new Tausch2D<long long>(localDim, haloWidth, numBuffers, valuesPerPoint, comm);
            return reinterpret_cast<CTausch*>(t);
        } else if(datatype == TAUSCH_LONG_DOUBLE) {
            Tausch<long double> *t;
            t = new Tausch2D<long double>(localDim, haloWidth, numBuffers, valuesPerPoint, comm);
            return reinterpret_cast<CTausch*>(t);
        } else {
            std::cerr << "ERROR! Invalid data type specified: " << datatype << " - Abort..." << std::endl;
            exit(1);
        }
    } else if(version == TAUSCH_3D) {
//        if(datatype == TAUSCH_DOUBLE) {
//            Tausch<double> *t;
//            t = new Tausch3D<double>(localDim, haloWidth, numBuffers, valuesPerPoint, comm);
//            return reinterpret_cast<CTausch*>(t);
//        } else if(datatype == TAUSCH_FLOAT) {
//            Tausch<float> *t;
//            t = new Tausch3D<float>(localDim, haloWidth, numBuffers, valuesPerPoint, comm);
//            return reinterpret_cast<CTausch*>(t);
//        } else if(datatype == TAUSCH_INT) {
//            Tausch<int> *t;
//            t = new Tausch3D<int>(localDim, haloWidth, numBuffers, valuesPerPoint, comm);
//            return reinterpret_cast<CTausch*>(t);
//        } else if(datatype == TAUSCH_UNSIGNED_INT) {
//            Tausch<unsigned int> *t;
//            t = new Tausch3D<unsigned int>(localDim, haloWidth, numBuffers, valuesPerPoint, comm);
//            return reinterpret_cast<CTausch*>(t);
//        } else if(datatype == TAUSCH_LONG) {
//            Tausch<long> *t;
//            t = new Tausch3D<long>(localDim, haloWidth, numBuffers, valuesPerPoint, comm);
//            return reinterpret_cast<CTausch*>(t);
//        } else if(datatype == TAUSCH_LONG_LONG) {
//            Tausch<long long> *t;
//            t = new Tausch3D<long long>(localDim, haloWidth, numBuffers, valuesPerPoint, comm);
//            return reinterpret_cast<CTausch*>(t);
//        } else if(datatype == TAUSCH_LONG_DOUBLE) {
//            Tausch<long double> *t;
//            t = new Tausch3D<long double>(localDim, haloWidth, numBuffers, valuesPerPoint, comm);
//            return reinterpret_cast<CTausch*>(t);
//        } else {
//            std::cerr << "ERROR! Invalid data type specified: " << datatype << " - Abort..." << std::endl;
//            exit(1);
//        }
    }
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

