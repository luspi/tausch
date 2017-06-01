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
    valuesPerPoint = 2;
    dat1 = new double[valuesPerPoint*(localDim + cpuHaloWidth[0] + cpuHaloWidth[1])]{};
    dat2 = new double[valuesPerPoint*(localDim + cpuHaloWidth[0] + cpuHaloWidth[1])]{};
    for(int i = 0; i < localDim; ++i) {
        for(int val = 0; val < valuesPerPoint; ++val) {
            dat1[valuesPerPoint*(i+cpuHaloWidth[0])+val] = (i+1)*10+val;
            dat2[valuesPerPoint*(i+cpuHaloWidth[0])+val] = (5+i+1)*10+val;
        }
    }

    size_t tauschLocalDim = localDim+cpuHaloWidth[0]+cpuHaloWidth[1];
    tausch = new Tausch1D<double>(&tauschLocalDim, MPI_DOUBLE, numBuffers, valuesPerPoint);

    // These are the (up to) 4 remote halos that are needed by this rank
    remoteHaloSpecs = new size_t*[2];
    // These are the (up to) 4 local halos that are needed tobe sent by this rank
    localHaloSpecs = new size_t*[2];

    localHaloSpecs[0] = new size_t[6]{cpuHaloWidth[0], cpuHaloWidth[1], left, 0};
    remoteHaloSpecs[0] = new size_t[6]{0, cpuHaloWidth[0], left, 1};
    localHaloSpecs[1] = new size_t[6]{localDim, cpuHaloWidth[0], right, 1};
    remoteHaloSpecs[1] = new size_t[6]{cpuHaloWidth[0]+localDim, cpuHaloWidth[1], right, 0};

    tausch->setLocalHaloInfoCpu(2, localHaloSpecs);
    tausch->setRemoteHaloInfoCpu(2, remoteHaloSpecs);

}

Sample::~Sample() {

    for(int i = 0; i < 2; ++i) {
        delete[] localHaloSpecs[i];
        delete[] remoteHaloSpecs[i];
    }
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

    for(int val = 0; val < valuesPerPoint; ++val) {
        for(int i = 0; i < localDim+cpuHaloWidth[0]+cpuHaloWidth[1]; ++i)
            std::cout << std::setw(3) << dat1[valuesPerPoint*i + val] << " ";
        if(val != valuesPerPoint-1)
            std::cout << "   ";
    }
    std::cout << "          ";
    for(int val = 0; val < valuesPerPoint; ++val) {
        for(int i = 0; i < localDim+cpuHaloWidth[0]+cpuHaloWidth[1]; ++i)
            std::cout << std::setw(3) << dat2[valuesPerPoint*i + val] << " ";
        if(val != valuesPerPoint-1)
            std::cout << "   ";
    }
    std::cout << std::endl;

}
