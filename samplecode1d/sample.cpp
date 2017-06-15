#include "sample.h"

Sample::Sample(size_t localDim, size_t loops, size_t *cpuHaloWidth) {

    this->localDim = localDim;
    this->loops = loops;
    for(int i = 0; i < 2; ++i)
        this->cpuHaloWidth[i] = cpuHaloWidth[i];

    int mpiRank = 0, mpiSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    left = mpiRank-1, right = mpiRank+1;
    if(mpiRank == 0)
        left = mpiSize-1;
    if(mpiRank == mpiSize-1)
        right = 0;

    numBuffers = 2;
    valuesPerPoint[0] = 1; valuesPerPoint[1] = 1;
    dat1 = new double[valuesPerPoint[0]*(localDim + cpuHaloWidth[0] + cpuHaloWidth[1])]{};
    dat2 = new double[valuesPerPoint[1]*(localDim + cpuHaloWidth[0] + cpuHaloWidth[1])]{};
    for(int i = 0; i < localDim; ++i) {
        for(int val = 0; val < valuesPerPoint[0]; ++val)
            dat1[valuesPerPoint[0]*(i+cpuHaloWidth[0])+val] = (i+1)*10+val;
        for(int val = 0; val < valuesPerPoint[1]; ++val)
            dat2[valuesPerPoint[1]*(i+cpuHaloWidth[0])+val] = (5+i+1)*10+val;
    }

    size_t tauschLocalDim = localDim+cpuHaloWidth[0]+cpuHaloWidth[1];
    tausch = new Tausch1D<double>(MPI_DOUBLE, numBuffers, valuesPerPoint);

    // These are the (up to) 4 remote halos that are needed by this rank
    remoteHaloSpecs = new TauschHaloSpec[2];
    // These are the (up to) 4 local halos that are needed tobe sent by this rank
    localHaloSpecs = new TauschHaloSpec[2];

    localHaloSpecs[0].bufferWidth = localDim+cpuHaloWidth[0]+cpuHaloWidth[1];
    localHaloSpecs[0].haloX = cpuHaloWidth[0];
    localHaloSpecs[0].haloWidth = cpuHaloWidth[1];
    localHaloSpecs[0].remoteMpiRank = left;
    remoteHaloSpecs[0].bufferWidth = localDim+cpuHaloWidth[0]+cpuHaloWidth[1];
    remoteHaloSpecs[0].haloX = 0;
    remoteHaloSpecs[0].haloWidth = cpuHaloWidth[0];
    remoteHaloSpecs[0].remoteMpiRank = left;


    localHaloSpecs[1].bufferWidth = localDim+cpuHaloWidth[0]+cpuHaloWidth[1];
    localHaloSpecs[1].haloX = localDim;
    localHaloSpecs[1].haloWidth = cpuHaloWidth[0];
    localHaloSpecs[1].remoteMpiRank = right;
    remoteHaloSpecs[1].bufferWidth = localDim+cpuHaloWidth[0]+cpuHaloWidth[1];
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
    delete dat1;
    delete dat2;

}

void Sample::launchCPU() {

    int sendtags[2] = {0, 1};
    int recvtags[2] = {1, 0};

    for(int iter = 0; iter < loops; ++iter) {

        tausch->postAllReceivesCpu(recvtags);

        tausch->packNextSendBufferCpu(0, dat1);
        tausch->packNextSendBufferCpu(0, dat2);
        tausch->sendCpu(0, sendtags[0]);

        tausch->packNextSendBufferCpu(1, dat1);
        tausch->packNextSendBufferCpu(1, dat2);
        tausch->sendCpu(1, sendtags[1]);

        tausch->recvCpu(0);
        tausch->unpackNextRecvBufferCpu(0, dat1);
        tausch->unpackNextRecvBufferCpu(0, dat2);

        tausch->recvCpu(1);
        tausch->unpackNextRecvBufferCpu(1, dat1);
        tausch->unpackNextRecvBufferCpu(1, dat2);

    }

}

void Sample::print() {

    for(int val = 0; val < valuesPerPoint[0]; ++val) {
        for(int i = 0; i < localDim+cpuHaloWidth[0]+cpuHaloWidth[1]; ++i)
            std::cout << std::setw(3) << dat1[valuesPerPoint[0]*i + val] << " ";
        if(val != valuesPerPoint[0]-1)
            std::cout << "   ";
    }
    std::cout << "          ";
    for(int val = 0; val < valuesPerPoint[1]; ++val) {
        for(int i = 0; i < localDim+cpuHaloWidth[0]+cpuHaloWidth[1]; ++i)
            std::cout << std::setw(3) << dat2[valuesPerPoint[1]*i + val] << " ";
        if(val != valuesPerPoint[1]-1)
            std::cout << "   ";
    }
    std::cout << std::endl;

}
