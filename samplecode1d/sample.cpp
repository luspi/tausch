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

    valuesPerPointPerBuffer = new size_t[numBuffers];
    for(int b = 0; b < numBuffers; ++b)
        valuesPerPointPerBuffer[b] = 1;
    dat = new double*[numBuffers];
    for(int b = 0; b < numBuffers; ++b)
        dat[b] = new double[valuesPerPointPerBuffer[b]*(localDim + cpuHaloWidth[0] + cpuHaloWidth[1])]{};

    if(!hybrid) {
        for(int i = 0; i < localDim; ++i) {
            for(int b = 0; b < numBuffers; ++b)
                for(int val = 0; val < valuesPerPointPerBuffer[b]; ++val)
                    dat[b][valuesPerPointPerBuffer[b]*(i+cpuHaloWidth[0])+val] = (b*5 + i+1)*10+val;
        }
    } else {
        for(int i = 0; i < localDim; ++i) {
            if(i >= (localDim-gpuDim)/2 && i < (localDim-gpuDim)/2+gpuDim) continue;
            for(int b = 0; b < numBuffers; ++b)
                for(int val = 0; val < valuesPerPointPerBuffer[b]; ++val)
                    dat[b][valuesPerPointPerBuffer[b]*(i+cpuHaloWidth[0])+val] = (b*5 + i+1)*10+val;
        }
    }

    size_t tauschLocalDim = localDim+cpuHaloWidth[0]+cpuHaloWidth[1];
    tausch = new Tausch1D<double>(MPI_DOUBLE, numBuffers, valuesPerPointPerBuffer);

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


    if(hybrid) {

        size_t tauschGpuDim = gpuDim+gpuHaloWidth[0]+gpuHaloWidth[1];
        tausch->enableOpenCL(true, 64, true, false);

        gpudat = new double*[numBuffers];
        for(int b = 0; b < numBuffers; ++b)
            gpudat[b] = new double[valuesPerPointPerBuffer[b]*(gpuDim + gpuHaloWidth[0] + gpuHaloWidth[1])]{};

        for(int i = 0; i < gpuDim; ++i) {
            for(int b = 0; b < numBuffers; ++b)
                for(int val = 0; val < valuesPerPointPerBuffer[b]; ++val)
                    gpudat[b][valuesPerPointPerBuffer[b]*(i+gpuHaloWidth[0])+val] = (b*5 + i+1)*10+val;
        }

        try {
            cl_gpudat = new cl::Buffer[numBuffers];
            for(int b = 0; b < numBuffers; ++b) {
                int s = valuesPerPointPerBuffer[b]*(gpuDim + gpuHaloWidth[0] + gpuHaloWidth[1]);
                cl_gpudat[b] = cl::Buffer(tausch->getOpenCLContext(), &gpudat[b][0], &gpudat[b][s], false);
            }
        } catch(cl::Error error) {
            std::cerr << "Samplecode2D :: constructor :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
            exit(1);
        }

        remoteHaloSpecsCpuForGpu = new TauschHaloSpec[2];
        localHaloSpecsCpuForGpu = new TauschHaloSpec[2];

        remoteHaloSpecsCpuForGpu[0].bufferWidth = tauschLocalDim;
        remoteHaloSpecsCpuForGpu[0].haloX = (localDim-gpuDim)/2+cpuHaloWidth[0];
        remoteHaloSpecsCpuForGpu[0].haloWidth = cpuForGpuHaloWidth[0];

        remoteHaloSpecsCpuForGpu[1].bufferWidth = tauschLocalDim;
        remoteHaloSpecsCpuForGpu[1].haloX = (localDim-gpuDim)/2+cpuHaloWidth[0]+gpuDim-cpuForGpuHaloWidth[1];
        remoteHaloSpecsCpuForGpu[1].haloWidth = cpuForGpuHaloWidth[1];

        localHaloSpecsCpuForGpu[0].bufferWidth = tauschLocalDim;
        localHaloSpecsCpuForGpu[0].haloX = (localDim-gpuDim)/2+cpuHaloWidth[0]-gpuHaloWidth[0];
        localHaloSpecsCpuForGpu[0].haloWidth = gpuHaloWidth[0];

        localHaloSpecsCpuForGpu[1].bufferWidth = tauschLocalDim;
        localHaloSpecsCpuForGpu[1].haloX = (localDim-gpuDim)/2+cpuHaloWidth[0]+gpuDim;
        localHaloSpecsCpuForGpu[1].haloWidth = gpuHaloWidth[1];

        tausch->setLocalHaloInfoCpuForGpu(2, localHaloSpecsCpuForGpu);
        tausch->setRemoteHaloInfoCpuForGpu(2, remoteHaloSpecsCpuForGpu);

        remoteHaloSpecsGpu = new TauschHaloSpec[2];
        localHaloSpecsGpu = new TauschHaloSpec[2];

        localHaloSpecsGpu[0].bufferWidth = tauschGpuDim;
        localHaloSpecsGpu[0].haloX = gpuHaloWidth[0];
        localHaloSpecsGpu[0].haloWidth = cpuForGpuHaloWidth[0];

        localHaloSpecsGpu[1].bufferWidth = tauschGpuDim;
        localHaloSpecsGpu[1].haloX = gpuDim+gpuHaloWidth[0]-cpuForGpuHaloWidth[1];
        localHaloSpecsGpu[1].haloWidth = cpuForGpuHaloWidth[1];

        remoteHaloSpecsGpu[0].bufferWidth = tauschGpuDim;
        remoteHaloSpecsGpu[0].haloX = 0;
        remoteHaloSpecsGpu[0].haloWidth = gpuHaloWidth[0];

        remoteHaloSpecsGpu[1].bufferWidth = tauschGpuDim;
        remoteHaloSpecsGpu[1].haloX = gpuDim+gpuHaloWidth[0];
        remoteHaloSpecsGpu[1].haloWidth = gpuHaloWidth[1];

        tausch->setLocalHaloInfoGpu(2, localHaloSpecsGpu);
        tausch->setRemoteHaloInfoGpu(2, remoteHaloSpecsGpu);

    }

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

    if(hybrid) {

        int sendtagsCpu[2] = {0, 1};
        int recvtagsCpu[2] = {1, 0};

        int sendtagsGpu[2] = {0, 1};
        int recvtagsGpu[2] = {0, 1};

        for(int iter = 0; iter < loops; ++iter) {

            tausch->postAllReceivesCpu(recvtagsCpu);

            for(int b = 0; b < numBuffers; ++b)
                tausch->packSendBufferCpu(0, b, dat[b]);
            tausch->sendCpu(0, sendtagsCpu[0]);

            for(int b = 0; b < numBuffers; ++b)
                tausch->packSendBufferCpu(1, b, dat[b]);
            tausch->sendCpu(1, sendtagsCpu[1]);

            for(int b = 0; b < numBuffers; ++b)
                tausch->packSendBufferCpuToGpu(0, b, dat[b]);
            tausch->sendCpuToGpu(0, sendtagsGpu[0]);

            for(int b = 0; b < numBuffers; ++b)
                tausch->packSendBufferCpuToGpu(1, b, dat[b]);
            tausch->sendCpuToGpu(1, sendtagsGpu[1]);

            tausch->recvCpu(0);
            for(int b = 0; b < numBuffers; ++b)
                tausch->unpackRecvBufferCpu(0, b, dat[b]);

            tausch->recvCpu(1);
            for(int b = 0; b < numBuffers; ++b)
                tausch->unpackRecvBufferCpu(1, b, dat[b]);

            tausch->recvGpuToCpu(0, recvtagsGpu[0]);
            for(int b = 0; b < numBuffers; ++b)
                tausch->unpackRecvBufferGpuToCpu(0, b, dat[b]);

            tausch->recvGpuToCpu(1, recvtagsGpu[1]);
            for(int b = 0; b < numBuffers; ++b)
                tausch->unpackRecvBufferGpuToCpu(1, b, dat[b]);

        }

    } else {

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

}

void Sample::launchGPU() {

    int sendtags[2] = {0, 1};
    int recvtags[2] = {0, 1};

    for(int iter = 0; iter < loops; ++iter) {

        tausch->recvCpuToGpu(0, recvtags[0  ]);
        for(int b = 0; b < numBuffers; ++b)
            tausch->unpackRecvBufferCpuToGpu(0, b, cl_gpudat[b]);

        tausch->recvCpuToGpu(1, recvtags[1]);
        for(int b = 0; b < numBuffers; ++b)
            tausch->unpackRecvBufferCpuToGpu(1, b, cl_gpudat[b]);

        for(int b = 0; b < numBuffers; ++b)
            tausch->packSendBufferGpuToCpu(0, b, cl_gpudat[b]);
        tausch->sendGpuToCpu(0, sendtags[0]);

        for(int b = 0; b < numBuffers; ++b)
            tausch->packSendBufferGpuToCpu(1, b, cl_gpudat[b]);
        tausch->sendGpuToCpu(1, sendtags[1]);

    }

    for(int b = 0; b < numBuffers; ++b) {
        int s = valuesPerPointPerBuffer[b]*(gpuDim + gpuHaloWidth[0] + gpuHaloWidth[1]);
        cl::copy(tausch->getOpenCLQueue(), cl_gpudat[b], &gpudat[b][0], &gpudat[b][s]);
    }

}

void Sample::printCPU() {

    for(int b = 0; b < numBuffers; ++b) {
        for(int val = 0; val < valuesPerPointPerBuffer[b]; ++val) {
            for(int i = 0; i < localDim+cpuHaloWidth[0]+cpuHaloWidth[1]; ++i)
                std::cout << std::setw(3) << dat[b][valuesPerPointPerBuffer[b]*i + val] << " ";
            if(val != valuesPerPointPerBuffer[b]-1)
                std::cout << "   ";
        }
        if(b != numBuffers-1)
            std::cout << "          ";
    }
    std::cout << std::endl;

}

void Sample::printGPU() {

    for(int b = 0; b < numBuffers; ++b) {
        for(int val = 0; val < valuesPerPointPerBuffer[b]; ++val) {
            for(int i = 0; i < gpuDim+gpuHaloWidth[0]+gpuHaloWidth[1]; ++i)
                std::cout << std::setw(3) << gpudat[b][valuesPerPointPerBuffer[b]*i + val] << " ";
            if(val != valuesPerPointPerBuffer[b]-1)
                std::cout << "   ";
        }
        if(b != numBuffers-1)
            std::cout << "          ";
    }
    std::cout << std::endl;

}
