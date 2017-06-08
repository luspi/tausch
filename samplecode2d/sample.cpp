#include "sample.h"

Sample::Sample(size_t *localDim, size_t loops, size_t *cpuHaloWidth, size_t *mpiNum) {

    this->localDim[0] = localDim[0];
    this->localDim[1] = localDim[1];
    this->loops = loops;
    for(int i = 0; i < 4; ++i)
        this->cpuHaloWidth[i] = cpuHaloWidth[i];
    this->mpiNum[0] = mpiNum[0];
    this->mpiNum[1] = mpiNum[1];

    int mpiRank = 0, mpiSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    left = mpiRank-1, right = mpiRank+1, top = mpiRank+mpiNum[0], bottom = mpiRank-mpiNum[0];
    if(mpiRank%mpiNum[0] == 0)
        left += mpiNum[0];
    if((mpiRank+1)%mpiNum[0] == 0)
        right -= mpiNum[0];
    if(mpiRank < mpiNum[0])
        bottom += mpiSize;
    if(mpiRank >= mpiSize-mpiNum[0])
        top -= mpiSize;

    numBuffers = 2;
    valuesPerPoint[0] = 1;
    valuesPerPoint[1] = 1;
    dat1 = new double[valuesPerPoint[0]*(localDim[0] + cpuHaloWidth[0] + cpuHaloWidth[1])*(localDim[1] + cpuHaloWidth[2] + cpuHaloWidth[3])]{};
    dat2 = new double[valuesPerPoint[1]*(localDim[0] + cpuHaloWidth[0] + cpuHaloWidth[1])*(localDim[1] + cpuHaloWidth[2] + cpuHaloWidth[3])]{};
    for(int j = 0; j < localDim[1]; ++j)
        for(int i = 0; i < localDim[0]; ++i) {
            for(int val = 0; val < valuesPerPoint[0]; ++val)
                dat1[valuesPerPoint[0]*((j+cpuHaloWidth[3])*(localDim[0]+cpuHaloWidth[0]+cpuHaloWidth[1]) + i+cpuHaloWidth[0])+val] =
                        (j*localDim[0]+i+1)*10+val;
            for(int val = 0; val < valuesPerPoint[1]; ++val)
                dat2[valuesPerPoint[1]*((j+cpuHaloWidth[3])*(localDim[0]+cpuHaloWidth[0]+cpuHaloWidth[1]) + i+cpuHaloWidth[0])+val] =
                        (5+j*localDim[0]+i+1)*10+val;
        }


    size_t tauschLocalDim[2] = {localDim[0]+cpuHaloWidth[0]+cpuHaloWidth[1], localDim[1]+cpuHaloWidth[2]+cpuHaloWidth[3]};
    tausch = new Tausch2D<double>(tauschLocalDim, MPI_DOUBLE, numBuffers, valuesPerPoint);

    // These are the (up to) 4 remote halos that are needed by this rank
    remoteHaloSpecs = new TauschHaloSpec[4];
    // These are the (up to) 4 local halos that are needed tobe sent by this rank
    localHaloSpecs = new TauschHaloSpec[4];

    localHaloSpecs[0].x = cpuHaloWidth[0]; localHaloSpecs[0].y = 0;
    localHaloSpecs[0].width = cpuHaloWidth[1]; localHaloSpecs[0].height = cpuHaloWidth[3]+localDim[1]+cpuHaloWidth[2];
    localHaloSpecs[0].remoteMpiRank = left;
    remoteHaloSpecs[0].x = 0; remoteHaloSpecs[0].y = 0;
    remoteHaloSpecs[0].width = cpuHaloWidth[0]; remoteHaloSpecs[0].height = cpuHaloWidth[3]+localDim[1]+cpuHaloWidth[2];
    remoteHaloSpecs[0].remoteMpiRank = left;

    localHaloSpecs[1].x = localDim[0]; localHaloSpecs[1].y = 0;
    localHaloSpecs[1].width = cpuHaloWidth[0]; localHaloSpecs[1].height = cpuHaloWidth[3]+localDim[1]+cpuHaloWidth[2];
    localHaloSpecs[1].remoteMpiRank = right;
    remoteHaloSpecs[1].x = cpuHaloWidth[0]+localDim[0]; remoteHaloSpecs[1].y = 0;
    remoteHaloSpecs[1].width = cpuHaloWidth[1]; remoteHaloSpecs[1].height = cpuHaloWidth[3]+localDim[1]+cpuHaloWidth[2];
    remoteHaloSpecs[1].remoteMpiRank = right;

    localHaloSpecs[2].x = 0; localHaloSpecs[2].y = localDim[1];
    localHaloSpecs[2].width = cpuHaloWidth[0]+localDim[0]+cpuHaloWidth[1]; localHaloSpecs[2].height = cpuHaloWidth[3];
    localHaloSpecs[2].remoteMpiRank = top;
    remoteHaloSpecs[2].x = 0; remoteHaloSpecs[2].y = cpuHaloWidth[3]+localDim[1];
    remoteHaloSpecs[2].width = cpuHaloWidth[0]+localDim[0]+cpuHaloWidth[1]; remoteHaloSpecs[2].height = cpuHaloWidth[2];
    remoteHaloSpecs[2].remoteMpiRank = top;

    localHaloSpecs[3].x = 0; localHaloSpecs[3].y = cpuHaloWidth[3];
    localHaloSpecs[3].width = cpuHaloWidth[0]+localDim[0]+cpuHaloWidth[1]; localHaloSpecs[3].height = cpuHaloWidth[2];
    localHaloSpecs[3].remoteMpiRank = bottom;
    remoteHaloSpecs[3].x = 0; remoteHaloSpecs[3].y = 0;
    remoteHaloSpecs[3].width = cpuHaloWidth[0]+localDim[0]+cpuHaloWidth[1]; remoteHaloSpecs[3].height = cpuHaloWidth[3];
    remoteHaloSpecs[3].remoteMpiRank = bottom;

    tausch->setLocalHaloInfoCpu(4, localHaloSpecs);
    tausch->setRemoteHaloInfoCpu(4, remoteHaloSpecs);

}

Sample::~Sample() {

    delete[] localHaloSpecs;
    delete[] remoteHaloSpecs;
    delete tausch;
    delete[] dat1;
    delete[] dat2;

}

void Sample::launchCPU() {

    for(int iter = 0; iter < loops; ++iter) {

        int sendtags[4] = {0, 1, 2, 3};
        int recvtags[4] = {1, 0, 3, 2};

        tausch->postAllReceivesCpu(recvtags);

        for(int ver_hor = 0; ver_hor < 2; ++ver_hor) {

            tausch->packNextSendBufferCpu(2*ver_hor, dat1);
            tausch->packNextSendBufferCpu(2*ver_hor, dat2);
            tausch->sendCpu(2*ver_hor, sendtags[2*ver_hor]);

            tausch->packNextSendBufferCpu(2*ver_hor+1, dat1);
            tausch->packNextSendBufferCpu(2*ver_hor+1, dat2);
            tausch->sendCpu(2*ver_hor+1, sendtags[2*ver_hor +1]);

            tausch->recvCpu(2*ver_hor);
            tausch->unpackNextRecvBufferCpu(2*ver_hor, dat1);
            tausch->unpackNextRecvBufferCpu(2*ver_hor, dat2);

            tausch->recvCpu(2*ver_hor+1);
            tausch->unpackNextRecvBufferCpu(2*ver_hor+1, dat1);
            tausch->unpackNextRecvBufferCpu(2*ver_hor+1, dat2);

        }

    }

}

void Sample::print() {

    for(int j = localDim[1]+cpuHaloWidth[2]+cpuHaloWidth[3]-1; j >= 0; --j) {
        for(int val = 0; val < valuesPerPoint[0]; ++val) {
            for(int i = 0; i < localDim[0]+cpuHaloWidth[0]+cpuHaloWidth[1]; ++i)
                std::cout << std::setw(3) << dat1[valuesPerPoint[0]*(j*(localDim[0]+cpuHaloWidth[0]+cpuHaloWidth[1]) + i) + val] << " ";
            if(val != valuesPerPoint[0]-1)
                std::cout << "   ";
        }
        std::cout << "          ";
        for(int val = 0; val < valuesPerPoint[1]; ++val) {
            for(int i = 0; i < localDim[0]+cpuHaloWidth[0]+cpuHaloWidth[1]; ++i)
                std::cout << std::setw(3) << dat2[valuesPerPoint[1]*(j*(localDim[0]+cpuHaloWidth[0]+cpuHaloWidth[1]) + i) + val] << " ";
            if(val != valuesPerPoint[1]-1)
                std::cout << "   ";
        }
        std::cout << std::endl;
    }

}
