#include "sample.h"

Sample::Sample(size_t localDim, size_t gpuDim, size_t loops, size_t *cpuHaloWidth, size_t *gpuHaloWidth, size_t *cpuForGpuHaloWidth, bool hybrid) {

    this->hybrid = hybrid;
    this->localDim = localDim;
    this->loops = loops;
    for(int i = 0; i < 2; ++i)
        this->cpuHaloWidth[i] = cpuHaloWidth[i];
#ifdef OPENCL
    this->gpuDim = gpuDim;
    for(int i = 0; i < 2; ++i)
        this->gpuHaloWidth[i] = gpuHaloWidth[i];
    for(int i = 0; i < 2; ++i)
        this->cpuForGpuHaloWidth[i] = cpuForGpuHaloWidth[i];
#endif

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

#ifdef OPENCL
    if(!hybrid) {
#endif
        for(int i = 0; i < localDim; ++i) {
            for(int b = 0; b < numBuffers; ++b)
                for(int val = 0; val < valuesPerPointPerBuffer[b]; ++val)
                    dat[b][valuesPerPointPerBuffer[b]*(i+cpuHaloWidth[0])+val] = (b*5 + i+1)*10+val;
        }
#ifdef OPENCL
    } else {
        for(int i = 0; i < localDim; ++i) {
            if(i >= (localDim-gpuDim)/2 && i < (localDim-gpuDim)/2+gpuDim) continue;
            for(int b = 0; b < numBuffers; ++b)
                for(int val = 0; val < valuesPerPointPerBuffer[b]; ++val)
                    dat[b][valuesPerPointPerBuffer[b]*(i+cpuHaloWidth[0])+val] = (b*5 + i+1)*10+val;
        }
    }
#endif

    size_t tauschLocalDim = localDim+cpuHaloWidth[0]+cpuHaloWidth[1];
    tausch = new Tausch<double>(MPI_DOUBLE, numBuffers, valuesPerPointPerBuffer, MPI_COMM_WORLD);

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

    tausch->addLocalHaloInfo1D_CwC(localHaloSpecs[0]);
    tausch->addRemoteHaloInfo1D_CwC(remoteHaloSpecs[0]);

    localHaloSpecs[1].bufferWidth = tauschLocalDim;
    localHaloSpecs[1].haloX = localDim;
    localHaloSpecs[1].haloWidth = cpuHaloWidth[0];
    localHaloSpecs[1].remoteMpiRank = right;
    remoteHaloSpecs[1].bufferWidth = tauschLocalDim;
    remoteHaloSpecs[1].haloX = cpuHaloWidth[0]+localDim;
    remoteHaloSpecs[1].haloWidth = cpuHaloWidth[1];
    remoteHaloSpecs[1].remoteMpiRank = right;

    tausch->addLocalHaloInfo1D_CwC(localHaloSpecs[1]);
    tausch->addRemoteHaloInfo1D_CwC(remoteHaloSpecs[1]);

#ifdef OPENCL
    if(hybrid) {

        size_t tauschGpuDim = gpuDim+gpuHaloWidth[0]+gpuHaloWidth[1];
        tausch->enableOpenCL1D(true, 64, true, false);

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
                cl_gpudat[b] = cl::Buffer(tausch->getOpenCLContext1D(), &gpudat[b][0], &gpudat[b][s], false);
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

        tausch->setLocalHaloInfo1D_CwG(2, localHaloSpecsCpuForGpu);
        tausch->setRemoteHaloInfo1D_CwG(2, remoteHaloSpecsCpuForGpu);

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

        tausch->setLocalHaloInfo1D_GwC(2, localHaloSpecsGpu);
        tausch->setRemoteHaloInfo1D_GwC(2, remoteHaloSpecsGpu);

    }
#endif

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

#ifdef OPENCL
    if(hybrid) {

        int sendtagsCpu[2] = {0, 1};
        int recvtagsCpu[2] = {1, 0};

        int sendtagsGpu[2] = {0, 1};
        int recvtagsGpu[2] = {0, 1};

        for(int iter = 0; iter < loops; ++iter) {

            tausch->postAllReceives1D_CwC(recvtagsCpu);
            tausch->postAllReceives1D_CwG(recvtagsGpu);

            for(int b = 0; b < numBuffers; ++b)
                tausch->packSendBuffer1D_CwC(0, b, dat[b]);
            tausch->send1D_CwC(0, sendtagsCpu[0]);

            for(int b = 0; b < numBuffers; ++b)
                tausch->packSendBuffer1D_CwC(1, b, dat[b]);
            tausch->send1D_CwC(1, sendtagsCpu[1]);

            for(int b = 0; b < numBuffers; ++b)
                tausch->packSendBuffer1D_CwG(0, b, dat[b]);
            tausch->send1D_CwG(0, sendtagsGpu[0]);

            for(int b = 0; b < numBuffers; ++b)
                tausch->packSendBuffer1D_CwG(1, b, dat[b]);
            tausch->send1D_CwG(1, sendtagsGpu[1]);

            tausch->recv1D_CwC(0);
            for(int b = 0; b < numBuffers; ++b)
                tausch->unpackRecvBuffer1D_CwC(0, b, dat[b]);

            tausch->recv1D_CwC(1);
            for(int b = 0; b < numBuffers; ++b)
                tausch->unpackRecvBuffer1D_CwC(1, b, dat[b]);

            tausch->recv1D_CwG(0);
            for(int b = 0; b < numBuffers; ++b)
                tausch->unpackRecvBuffer1D_CwG(0, b, dat[b]);

            tausch->recv1D_CwG(1);
            for(int b = 0; b < numBuffers; ++b)
                tausch->unpackRecvBuffer1D_CwG(1, b, dat[b]);

        }

    } else {

#endif

        int sendtags[2] = {0, 1};
        int recvtags[2] = {1, 0};

        for(int iter = 0; iter < loops; ++iter) {

            tausch->postAllReceives1D_CwC(recvtags);

            for(int b = 0; b < numBuffers; ++b)
                tausch->packSendBuffer1D_CwC(0, b, dat[b]);
            tausch->send1D_CwC(0, sendtags[0]);

            for(int b = 0; b < numBuffers; ++b)
                tausch->packSendBuffer1D_CwC(1, b, dat[b]);
            tausch->send1D_CwC(1, sendtags[1]);

            tausch->recv1D_CwC(0);
            for(int b = 0; b < numBuffers; ++b)
                tausch->unpackRecvBuffer1D_CwC(0, b, dat[b]);

            tausch->recv1D_CwC(1);
            for(int b = 0; b < numBuffers; ++b)
                tausch->unpackRecvBuffer1D_CwC(1, b, dat[b]);

        }

#ifdef OPENCL
    }
#endif

}

#ifdef OPENCL
void Sample::launchGPU() {

    int sendtags[2] = {0, 1};
    int recvtags[2] = {0, 1};

    tausch->postAllReceives1D_GwC(recvtags);

    for(int iter = 0; iter < loops; ++iter) {

        tausch->recv1D_GwC(0);
        for(int b = 0; b < numBuffers; ++b)
            tausch->unpackRecvBuffer1D_GwC(0, b, cl_gpudat[b]);

        tausch->recv1D_GwC(1);
        for(int b = 0; b < numBuffers; ++b)
            tausch->unpackRecvBuffer1D_GwC(1, b, cl_gpudat[b]);

        for(int b = 0; b < numBuffers; ++b)
            tausch->packSendBuffer1D_GwC(0, b, cl_gpudat[b]);
        tausch->send1D_GwC(0, sendtags[0]);

        for(int b = 0; b < numBuffers; ++b)
            tausch->packSendBuffer1D_GwC(1, b, cl_gpudat[b]);
        tausch->send1D_GwC(1, sendtags[1]);

    }

    for(int b = 0; b < numBuffers; ++b) {
        int s = valuesPerPointPerBuffer[b]*(gpuDim + gpuHaloWidth[0] + gpuHaloWidth[1]);
        cl::copy(tausch->getOpenCLQueue1D(), cl_gpudat[b], &gpudat[b][0], &gpudat[b][s]);
    }

}
#endif

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

#ifdef OPENCL
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
#endif
