#include "sample.h"

Sample::Sample(size_t *localDim, size_t loops, size_t *cpuHaloWidth, size_t *mpiNum) {

    for(int i = 0; i < 3; ++i)
        this->localDim[i] = localDim[i];
    this->loops = loops;
    for(int i = 0; i < 6; ++i)
        this->cpuHaloWidth[i] = cpuHaloWidth[i];
    for(int i = 0; i < 3; ++i)
        this->mpiNum[i] = mpiNum[i];

    int mpiRank = 0, mpiSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    left = mpiRank-1, right = mpiRank+1;
    top = mpiRank+mpiNum[TAUSCH_X], bottom = mpiRank-mpiNum[TAUSCH_X];
    front = mpiRank-mpiNum[TAUSCH_X]*mpiNum[TAUSCH_Y]; back = mpiRank+mpiNum[TAUSCH_X]*mpiNum[TAUSCH_Y];

    if(mpiRank%mpiNum[TAUSCH_X] == 0)
        left += mpiNum[TAUSCH_X];
    if((mpiRank+1)%mpiNum[TAUSCH_X] == 0)
        right -= mpiNum[TAUSCH_X];
    if(mpiRank%(mpiNum[TAUSCH_X]*mpiNum[TAUSCH_Y]) >= (mpiNum[TAUSCH_X]*mpiNum[TAUSCH_Y]-mpiNum[TAUSCH_X]))
        top -= mpiNum[TAUSCH_X]*mpiNum[TAUSCH_Y];
    if(mpiRank%(mpiNum[TAUSCH_X]*mpiNum[TAUSCH_Y]) < mpiNum[TAUSCH_X])
        bottom += mpiNum[TAUSCH_X]*mpiNum[TAUSCH_Y];
    if(mpiRank < mpiNum[TAUSCH_X]*mpiNum[TAUSCH_Y])
        front += mpiSize;
    if(mpiRank >= mpiSize-mpiNum[TAUSCH_X]*mpiNum[TAUSCH_Y])
        back -= mpiSize;

    numBuffers = 2;
    valuesPerPoint[0] = 1;
    valuesPerPoint[1] = 1;
    dat1 = new double[valuesPerPoint[0]*(localDim[0] + cpuHaloWidth[0] + cpuHaloWidth[1])*
                                     (localDim[1] + cpuHaloWidth[2] + cpuHaloWidth[3])*
                                     (localDim[2] + cpuHaloWidth[4] + cpuHaloWidth[5])]{};
    dat2 = new double[valuesPerPoint[1]*(localDim[0] + cpuHaloWidth[0] + cpuHaloWidth[1])*
                                     (localDim[1] + cpuHaloWidth[2] + cpuHaloWidth[3])*
                                     (localDim[2] + cpuHaloWidth[4] + cpuHaloWidth[5])]{};

    for(int k = 0; k < localDim[TAUSCH_Z]; ++k)
        for(int j = 0; j < localDim[TAUSCH_Y]; ++j)
            for(int i = 0; i < localDim[TAUSCH_X]; ++i) {
                for(int val = 0; val < valuesPerPoint[0]; ++val)
                    dat1[valuesPerPoint[0]*((k+cpuHaloWidth[4])*(localDim[TAUSCH_X]+cpuHaloWidth[0]+cpuHaloWidth[1])*
                                         (localDim[TAUSCH_Y]+cpuHaloWidth[2]+cpuHaloWidth[3]) +
                                         (j+cpuHaloWidth[3])*(localDim[TAUSCH_X]+cpuHaloWidth[0]+cpuHaloWidth[1]) +
                                          i+cpuHaloWidth[0]) + val] = (k*localDim[TAUSCH_X]*localDim[TAUSCH_Y] + j*localDim[TAUSCH_X] + i + 1)*10+val;
                for(int val = 0; val < valuesPerPoint[1]; ++val)
                    dat2[valuesPerPoint[1]*((k+cpuHaloWidth[4])*(localDim[TAUSCH_X]+cpuHaloWidth[0]+cpuHaloWidth[1])*
                                         (localDim[TAUSCH_Y]+cpuHaloWidth[2]+cpuHaloWidth[3]) +
                                         (j+cpuHaloWidth[3])*(localDim[TAUSCH_X]+cpuHaloWidth[0]+cpuHaloWidth[1]) +
                                          i+cpuHaloWidth[0])+val] = (5+k*localDim[TAUSCH_X]*localDim[TAUSCH_Y] + j*localDim[TAUSCH_X] + i + 1)*10+val;
            }

    size_t tauschLocalDim[3] = {localDim[0]+cpuHaloWidth[0]+cpuHaloWidth[1],
                                localDim[1]+cpuHaloWidth[2]+cpuHaloWidth[3],
                                localDim[2]+cpuHaloWidth[4]+cpuHaloWidth[5]};
    tausch = new Tausch3D<double>(MPI_DOUBLE, numBuffers, valuesPerPoint);

    // These are the (up to) 4 remote halos that are needed by this rank
    remoteHaloSpecs = new TauschHaloSpec[6];
    // These are the (up to) 4 local halos that are needed tobe sent by this rank
    localHaloSpecs = new TauschHaloSpec[6];

    localHaloSpecs[0].bufferWidth = tauschLocalDim[0]; localHaloSpecs[0].bufferHeight = tauschLocalDim[1]; localHaloSpecs[0].bufferDepth = tauschLocalDim[2];
    localHaloSpecs[0].haloX = cpuHaloWidth[0]; localHaloSpecs[0].haloY = 0; localHaloSpecs[0].haloZ = 0;
    localHaloSpecs[0].haloWidth = cpuHaloWidth[1]; localHaloSpecs[0].haloHeight = cpuHaloWidth[3]+localDim[1]+cpuHaloWidth[2];
    localHaloSpecs[0].haloDepth = cpuHaloWidth[4]+localDim[2]+cpuHaloWidth[5]; localHaloSpecs[0].remoteMpiRank = left;

    localHaloSpecs[1].bufferWidth = tauschLocalDim[0]; localHaloSpecs[1].bufferHeight = tauschLocalDim[1]; localHaloSpecs[1].bufferDepth = tauschLocalDim[2];
    localHaloSpecs[1].haloX = localDim[0]; localHaloSpecs[1].haloY = 0; localHaloSpecs[1].haloZ = 0;
    localHaloSpecs[1].haloWidth = cpuHaloWidth[0]; localHaloSpecs[1].haloHeight = cpuHaloWidth[3]+localDim[1]+cpuHaloWidth[2];
    localHaloSpecs[1].haloDepth = cpuHaloWidth[4]+localDim[2]+cpuHaloWidth[5]; localHaloSpecs[1].remoteMpiRank = right;

    localHaloSpecs[2].bufferWidth = tauschLocalDim[0]; localHaloSpecs[2].bufferHeight = tauschLocalDim[1]; localHaloSpecs[2].bufferDepth = tauschLocalDim[2];
    localHaloSpecs[2].haloX = 0; localHaloSpecs[2].haloY = localDim[1]; localHaloSpecs[2].haloZ = 0;
    localHaloSpecs[2].haloWidth = cpuHaloWidth[0]+localDim[0]+cpuHaloWidth[1]; localHaloSpecs[2].haloHeight = cpuHaloWidth[3];
    localHaloSpecs[2].haloDepth = cpuHaloWidth[4]+localDim[2]+cpuHaloWidth[5]; localHaloSpecs[2].remoteMpiRank = top;

    localHaloSpecs[3].bufferWidth = tauschLocalDim[0]; localHaloSpecs[3].bufferHeight = tauschLocalDim[1]; localHaloSpecs[3].bufferDepth = tauschLocalDim[2];
    localHaloSpecs[3].haloX = 0; localHaloSpecs[3].haloY = cpuHaloWidth[3]; localHaloSpecs[3].haloZ = 0;
    localHaloSpecs[3].haloWidth = cpuHaloWidth[0]+localDim[0]+cpuHaloWidth[1]; localHaloSpecs[3].haloHeight = cpuHaloWidth[2];
    localHaloSpecs[3].haloDepth = cpuHaloWidth[4]+localDim[2]+cpuHaloWidth[5]; localHaloSpecs[3].remoteMpiRank = bottom;

    localHaloSpecs[4].bufferWidth = tauschLocalDim[0]; localHaloSpecs[4].bufferHeight = tauschLocalDim[1]; localHaloSpecs[4].bufferDepth = tauschLocalDim[2];
    localHaloSpecs[4].haloX = 0; localHaloSpecs[4].haloY = 0; localHaloSpecs[4].haloZ = cpuHaloWidth[4];
    localHaloSpecs[4].haloWidth = cpuHaloWidth[0]+localDim[0]+cpuHaloWidth[1]; localHaloSpecs[4].haloHeight = cpuHaloWidth[2]+localDim[1]+cpuHaloWidth[3];
    localHaloSpecs[4].haloDepth = cpuHaloWidth[5]; localHaloSpecs[4].remoteMpiRank = front;

    localHaloSpecs[5].bufferWidth = tauschLocalDim[0]; localHaloSpecs[5].bufferHeight = tauschLocalDim[1]; localHaloSpecs[5].bufferDepth = tauschLocalDim[2];
    localHaloSpecs[5].haloX = 0; localHaloSpecs[5].haloY = 0; localHaloSpecs[5].haloZ = localDim[2];
    localHaloSpecs[5].haloWidth = cpuHaloWidth[0]+localDim[0]+cpuHaloWidth[1]; localHaloSpecs[5].haloHeight = cpuHaloWidth[2]+localDim[1]+cpuHaloWidth[3];
    localHaloSpecs[5].haloDepth = cpuHaloWidth[4]; localHaloSpecs[5].remoteMpiRank = back;


    remoteHaloSpecs[0].bufferWidth = tauschLocalDim[0]; remoteHaloSpecs[0].bufferHeight = tauschLocalDim[1]; remoteHaloSpecs[0].bufferDepth = tauschLocalDim[2];
    remoteHaloSpecs[0].haloX = 0; remoteHaloSpecs[0].haloY = 0; remoteHaloSpecs[0].haloZ = 0;
    remoteHaloSpecs[0].haloWidth = cpuHaloWidth[0]; remoteHaloSpecs[0].haloHeight = cpuHaloWidth[3]+localDim[1]+cpuHaloWidth[2];
    remoteHaloSpecs[0].haloDepth = cpuHaloWidth[4]+localDim[2]+cpuHaloWidth[5]; remoteHaloSpecs[0].remoteMpiRank = left;

    remoteHaloSpecs[1].bufferWidth = tauschLocalDim[0]; remoteHaloSpecs[1].bufferHeight = tauschLocalDim[1]; remoteHaloSpecs[1].bufferDepth = tauschLocalDim[2];
    remoteHaloSpecs[1].haloX = localDim[0]+cpuHaloWidth[0]; remoteHaloSpecs[1].haloY = 0; remoteHaloSpecs[1].haloZ = 0;
    remoteHaloSpecs[1].haloWidth = cpuHaloWidth[1]; remoteHaloSpecs[1].haloHeight = cpuHaloWidth[3]+localDim[1]+cpuHaloWidth[2];
    remoteHaloSpecs[1].haloDepth = cpuHaloWidth[4]+localDim[2]+cpuHaloWidth[5]; remoteHaloSpecs[1].remoteMpiRank = right;

    remoteHaloSpecs[2].bufferWidth = tauschLocalDim[0]; remoteHaloSpecs[2].bufferHeight = tauschLocalDim[1]; remoteHaloSpecs[2].bufferDepth = tauschLocalDim[2];
    remoteHaloSpecs[2].haloX = 0; remoteHaloSpecs[2].haloY = localDim[1]+cpuHaloWidth[3]; remoteHaloSpecs[2].haloZ = 0;
    remoteHaloSpecs[2].haloWidth = cpuHaloWidth[0]+localDim[0]+cpuHaloWidth[1]; remoteHaloSpecs[2].haloHeight = cpuHaloWidth[2];
    remoteHaloSpecs[2].haloDepth = cpuHaloWidth[4]+localDim[2]+cpuHaloWidth[5]; remoteHaloSpecs[2].remoteMpiRank = top;

    remoteHaloSpecs[3].bufferWidth = tauschLocalDim[0]; remoteHaloSpecs[3].bufferHeight = tauschLocalDim[1]; remoteHaloSpecs[3].bufferDepth = tauschLocalDim[2];
    remoteHaloSpecs[3].haloX = 0; remoteHaloSpecs[3].haloY = 0; remoteHaloSpecs[3].haloZ = 0;
    remoteHaloSpecs[3].haloWidth = cpuHaloWidth[0]+localDim[0]+cpuHaloWidth[1]; remoteHaloSpecs[3].haloHeight = cpuHaloWidth[3];
    remoteHaloSpecs[3].haloDepth = cpuHaloWidth[4]+localDim[2]+cpuHaloWidth[5]; remoteHaloSpecs[3].remoteMpiRank = bottom;

    remoteHaloSpecs[4].bufferWidth = tauschLocalDim[0]; remoteHaloSpecs[4].bufferHeight = tauschLocalDim[1]; remoteHaloSpecs[4].bufferDepth = tauschLocalDim[2];
    remoteHaloSpecs[4].haloX = 0; remoteHaloSpecs[4].haloY = 0; remoteHaloSpecs[4].haloZ = 0;
    remoteHaloSpecs[4].haloWidth = cpuHaloWidth[0]+localDim[0]+cpuHaloWidth[1]; remoteHaloSpecs[4].haloHeight = cpuHaloWidth[2]+localDim[1]+cpuHaloWidth[3];
    remoteHaloSpecs[4].haloDepth = cpuHaloWidth[4]; remoteHaloSpecs[4].remoteMpiRank = front;

    remoteHaloSpecs[5].bufferWidth = tauschLocalDim[0]; remoteHaloSpecs[5].bufferHeight = tauschLocalDim[1]; remoteHaloSpecs[5].bufferDepth = tauschLocalDim[2];
    remoteHaloSpecs[5].haloX = 0; remoteHaloSpecs[5].haloY = 0; remoteHaloSpecs[5].haloZ = localDim[2]+cpuHaloWidth[4];
    remoteHaloSpecs[5].haloWidth = cpuHaloWidth[0]+localDim[0]+cpuHaloWidth[1]; remoteHaloSpecs[5].haloHeight = cpuHaloWidth[2]+localDim[1]+cpuHaloWidth[3];
    remoteHaloSpecs[5].haloDepth = cpuHaloWidth[5]; remoteHaloSpecs[5].remoteMpiRank = back;


    tausch->setLocalHaloInfoCpu(6, localHaloSpecs);
    tausch->setRemoteHaloInfoCpu(6, remoteHaloSpecs);

}

Sample::~Sample() {

    delete[] localHaloSpecs;
    delete[] remoteHaloSpecs;
    delete tausch;
    delete[] dat1;
    delete[] dat2;

}

void Sample::launchCPU() {

    int sendtags[6] = {0, 1, 2, 3, 4, 5};
    int recvtags[6] = {1, 0, 3, 2, 5, 4};

    for(int iter = 0; iter < loops; ++iter) {

        tausch->postAllReceivesCpu(recvtags);

        // left/right

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

        // top/bottom

        tausch->packNextSendBufferCpu(2, dat1);
        tausch->packNextSendBufferCpu(2, dat2);
        tausch->sendCpu(2, sendtags[2]);

        tausch->packNextSendBufferCpu(3, dat1);
        tausch->packNextSendBufferCpu(3, dat2);
        tausch->sendCpu(3, sendtags[3]);

        tausch->recvCpu(2);
        tausch->unpackNextRecvBufferCpu(2, dat1);
        tausch->unpackNextRecvBufferCpu(2, dat2);

        tausch->recvCpu(3);
        tausch->unpackNextRecvBufferCpu(3, dat1);
        tausch->unpackNextRecvBufferCpu(3, dat2);

        // front/back

        tausch->packNextSendBufferCpu(4, dat1);
        tausch->packNextSendBufferCpu(4, dat2);
        tausch->sendCpu(4, sendtags[4]);

        tausch->packNextSendBufferCpu(5, dat1);
        tausch->packNextSendBufferCpu(5, dat2);
        tausch->sendCpu(5, sendtags[5]);

        tausch->recvCpu(4);
        tausch->unpackNextRecvBufferCpu(4, dat1);
        tausch->unpackNextRecvBufferCpu(4, dat2);

        tausch->recvCpu(5);
        tausch->unpackNextRecvBufferCpu(5, dat1);
        tausch->unpackNextRecvBufferCpu(5, dat2);

    }

}

void Sample::print() {

    for(int z = 0; z < localDim[TAUSCH_Z]+cpuHaloWidth[4]+cpuHaloWidth[5]; ++z) {

        std::cout << std::endl << "z = " << z << std::endl;

        for(int j = localDim[TAUSCH_Y]+cpuHaloWidth[2]+cpuHaloWidth[3]-1; j >= 0; --j) {

            for(int val = 0; val < valuesPerPoint[0]; ++val) {
                for(int i = 0; i < localDim[TAUSCH_X]+cpuHaloWidth[0]+cpuHaloWidth[1]; ++i)
                    std::cout << std::setw(4) << dat1[z*(localDim[TAUSCH_X]+cpuHaloWidth[0]+cpuHaloWidth[1])*
                                                        (localDim[TAUSCH_Y]+cpuHaloWidth[2]+cpuHaloWidth[3]) +
                                                      j*(localDim[TAUSCH_X]+cpuHaloWidth[0]+cpuHaloWidth[1]) + i] << " ";
                if(val != valuesPerPoint[0]-1)
                    std::cout << "   ";
            }
            std::cout << "          ";
            for(int val = 0; val < valuesPerPoint[1]; ++val) {
                for(int i = 0; i < localDim[TAUSCH_X]+cpuHaloWidth[0]+cpuHaloWidth[1]; ++i)
                    std::cout << std::setw(4) << dat2[z*(localDim[TAUSCH_X]+cpuHaloWidth[0]+cpuHaloWidth[1])*
                                                        (localDim[TAUSCH_Y]+cpuHaloWidth[2]+cpuHaloWidth[3]) +
                                                      j*(localDim[TAUSCH_X]+cpuHaloWidth[0]+cpuHaloWidth[1]) + i] << " ";
                if(val != valuesPerPoint[1]-1)
                    std::cout << "   ";
            }
            std::cout << std::endl;
        }

    }

}
