#include "sample.h"

Sample::Sample(size_t localDim, size_t gpuDim, size_t loops, size_t *cpuHaloWidth, size_t *gpuHaloWidth, size_t *cpuForGpuHaloWidth, bool hybrid) {

    this->hybrid = hybrid;
    this->localDim = localDim;
    this->gpuDim = gpuDim;
    this->loops = loops;
    for(int i = 0; i < 2; ++i)
        this->cpuHaloWidth[i] = cpuHaloWidth[i];
    for(int i = 0; i < 2; ++i)
        this->gpuHaloWidth[i] = gpuHaloWidth[i];
    for(int i = 0; i < 2; ++i)
        this->cpuForGpuHaloWidth[i] = cpuForGpuHaloWidth[i];

    int mpiRank = 0, mpiSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    left = mpiRank-1, right = mpiRank+1;
    if(mpiRank == 0)
        left = mpiSize-1;
    if(mpiRank == mpiSize-1)
        right = 0;

    numBuffers = 2;

    valuesPerPoint = new size_t[numBuffers];
    for(int b = 0; b < numBuffers; ++b)
        valuesPerPoint[b] = 1;
    dat = new double*[numBuffers];
    for(int b = 0; b < numBuffers; ++b)
        dat[b] = new double[valuesPerPoint[b]*(localDim + cpuHaloWidth[0] + cpuHaloWidth[1])]{};

    if(!hybrid) {
        for(int i = 0; i < localDim; ++i) {
            for(int b = 0; b < numBuffers; ++b)
                for(int val = 0; val < valuesPerPoint[b]; ++val)
                    dat[b][valuesPerPoint[b]*(i+cpuHaloWidth[0])+val] = (b*5 + i+1)*10+val;
        }
    } else {
        for(int i = 0; i < localDim; ++i) {
            if(i >= (localDim-gpuDim)/2 && i < (localDim-gpuDim)/2+gpuDim) continue;
            for(int b = 0; b < numBuffers; ++b)
                for(int val = 0; val < valuesPerPoint[b]; ++val)
                    dat[b][valuesPerPoint[b]*(i+cpuHaloWidth[0])+val] = (b*5 + i+1)*10+val;
        }
    }

    size_t tauschLocalDim = localDim+cpuHaloWidth[0]+cpuHaloWidth[1];
    tausch = new Tausch1D<double>(MPI_DOUBLE, numBuffers, valuesPerPoint);

    // These are the (up to) 4 remote halos that are needed by this rank
    remoteHaloSpecs = new TauschHaloSpec[2];
    // These are the (up to) 4 local halos that are needed tobe sent by this rank
    localHaloSpecs = new TauschHaloSpec[2];

    localHaloSpecs[0].bufferWidth = tauschLocalDim;
    localHaloSpecs[0].haloX = cpuHaloWidth[0];
    localHaloSpecs[0].haloWidth = cpuHaloWidth[1];
    localHaloSpecs[0].remoteMpiRank = left;
    remoteHaloSpecs[0].bufferWidth = tauschLocalDim;
    remoteHaloSpecs[0].haloX = 0;
    remoteHaloSpecs[0].haloWidth = cpuHaloWidth[0];
    remoteHaloSpecs[0].remoteMpiRank = left;


    localHaloSpecs[1].bufferWidth = tauschLocalDim;
    localHaloSpecs[1].haloX = localDim;
    localHaloSpecs[1].haloWidth = cpuHaloWidth[0];
    localHaloSpecs[1].remoteMpiRank = right;
    remoteHaloSpecs[1].bufferWidth = tauschLocalDim;
    remoteHaloSpecs[1].haloX = cpuHaloWidth[0]+localDim;
    remoteHaloSpecs[1].haloWidth = cpuHaloWidth[1];
    remoteHaloSpecs[1].remoteMpiRank = right;

    tausch->setLocalHaloInfoCpu(2, localHaloSpecs);
    tausch->setRemoteHaloInfoCpu(2, remoteHaloSpecs);

}

Sample::~Sample() {

    delete[] localHaloSpecs;
    delete[] remoteHaloSpecs;
    delete tausch;
    for(int b = 0; b < numBuffers; ++b)
        delete[] dat[b];
    delete[] dat;

}

void Sample::launchCPU() {

    int sendtags[2] = {0, 1};
    int recvtags[2] = {1, 0};

    for(int iter = 0; iter < loops; ++iter) {

        tausch->postAllReceivesCpu(recvtags);

        for(int b = 0; b < numBuffers; ++b)
            tausch->packSendBufferCpu(0, b, dat[b]);
        tausch->sendCpu(0, sendtags[0]);

        for(int b = 0; b < numBuffers; ++b)
            tausch->packSendBufferCpu(1, b, dat[b]);
        tausch->sendCpu(1, sendtags[1]);

        tausch->recvCpu(0);
        for(int b = 0; b < numBuffers; ++b)
            tausch->unpackRecvBufferCpu(0, b, dat[b]);

        tausch->recvCpu(1);
        for(int b = 0; b < numBuffers; ++b)
            tausch->unpackRecvBufferCpu(1, b, dat[b]);

    }

}

void Sample::launchGPU() {

//    int sendtags[2] = {0, 1};
//    int recvtags[2] = {1, 0};

//    for(int iter = 0; iter < loops; ++iter) {

//        tausch->postAllReceivesCpu(recvtags);

//        tausch->packNextSendBufferCpu(0, dat1);
//        tausch->packNextSendBufferCpu(0, dat2);
//        tausch->sendCpu(0, sendtags[0]);

//        tausch->packNextSendBufferCpu(1, dat1);
//        tausch->packNextSendBufferCpu(1, dat2);
//        tausch->sendCpu(1, sendtags[1]);

//        tausch->recvCpu(0);
//        tausch->unpackNextRecvBufferCpu(0, dat1);
//        tausch->unpackNextRecvBufferCpu(0, dat2);

//        tausch->recvCpu(1);
//        tausch->unpackNextRecvBufferCpu(1, dat1);
//        tausch->unpackNextRecvBufferCpu(1, dat2);

//    }

}

void Sample::print() {

    for(int b = 0; b < numBuffers; ++b) {
        for(int val = 0; val < valuesPerPoint[b]; ++val) {
            for(int i = 0; i < localDim+cpuHaloWidth[0]+cpuHaloWidth[1]; ++i)
                std::cout << std::setw(3) << dat[b][valuesPerPoint[b]*i + val] << " ";
            if(val != valuesPerPoint[b]-1)
                std::cout << "   ";
        }
        if(b != numBuffers-1)
            std::cout << "          ";
    }
    std::cout << std::endl;

}
