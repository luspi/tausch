#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

#include <mpi.h>
#define TAUSCH_OPENCL
#define TAUSCH_CUDA
#include "../tausch.h"

static cl::Device tauschcl_device;
static cl::Context tauschcl_context;
static cl::CommandQueue tauschcl_queue;

void setupOpenCL();

int test_1buf(std::vector<int> sendIndices, std::vector<int> recvIndices, double *expected) {

    Tausch<double> *tausch = new Tausch<double>(tauschcl_device, tauschcl_context, tauschcl_queue, "double",
                                                MPI_DOUBLE, MPI_COMM_WORLD, true);

    double *buf = new double[10]{};
    for(int i = 0; i < 10; ++i)
        buf[i] = i + 1;
    double *cudabuf;
    cudaMalloc(&cudabuf, 10*sizeof(double));
    cudaMemcpy(cudabuf, buf, 10*sizeof(double), cudaMemcpyHostToDevice);
    cl::Buffer clbuf(tauschcl_context, &buf[0], &buf[10], false);

    tausch->addLocalHaloInfoCUDA(sendIndices);
    tausch->addRemoteHaloInfoOCL(recvIndices);

    tausch->packSendBufferCUDA(0, 0, cudabuf);

    tausch->sendCUDA(0, 0, 0);
    tausch->recvOCL(0, 0, 0);

    tausch->unpackRecvBufferOCL(0, 0, clbuf);

    delete tausch;

    double *_clbuf = new double[10]{};
    cl::copy(tauschcl_queue, clbuf, &_clbuf[0], &_clbuf[10]);

    for(int i = 0; i < 10; ++i)
        if(fabs(expected[i]-_clbuf[i]) > 1e-10)
            return 1;

    delete[] buf;

    return 0;

}

int test_2buf(std::vector<int> sendIndices, std::vector<int> recvIndices, double *expected1, double *expected2) {

    Tausch<double> *tausch = new Tausch<double>(tauschcl_device, tauschcl_context, tauschcl_queue, "double",
                                                MPI_DOUBLE, MPI_COMM_WORLD, true);

    double *buf1 = new double[10]{};
    double *buf2 = new double[10]{};
    for(int i = 0; i < 10; ++i) {
        buf1[i] = i + 1;
        buf2[i] = i + 11;
    }
    double *cudabuf1, *cudabuf2;
    cudaMalloc(&cudabuf1, 10*sizeof(double));
    cudaMalloc(&cudabuf2, 10*sizeof(double));
    cudaMemcpy(cudabuf1, buf1, 10*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cudabuf2, buf2, 10*sizeof(double), cudaMemcpyHostToDevice);
    cl::Buffer clbuf1(tauschcl_context, &buf1[0], &buf1[10], false);
    cl::Buffer clbuf2(tauschcl_context, &buf2[0], &buf2[10], false);

    tausch->addLocalHaloInfoCUDA(sendIndices, 2);
    tausch->addRemoteHaloInfoOCL(recvIndices, 2);

    tausch->packSendBufferCUDA(0, 0, cudabuf1);
    tausch->packSendBufferCUDA(0, 1, cudabuf2);

    tausch->sendCUDA(0, 0, 0);
    tausch->recvOCL(0, 0, 0);

    tausch->unpackRecvBufferOCL(0, 0, clbuf2);
    tausch->unpackRecvBufferOCL(0, 1, clbuf1);

    delete tausch;

    double *_clbuf1 = new double[10]{};
    double *_clbuf2 = new double[10]{};
    cl::copy(tauschcl_queue, clbuf1, &_clbuf1[0], &_clbuf1[10]);
    cl::copy(tauschcl_queue, clbuf2, &_clbuf2[0], &_clbuf2[10]);

    for(int i = 0; i < 10; ++i) {
        if(fabs(expected1[i]-_clbuf1[i]) > 1e-10)
            return 1;
        if(fabs(expected2[i]-_clbuf2[i]) > 1e-10)
            return 1;
    }

    delete[] buf1;
    delete[] buf2;

    return 0;

}



int main(int argc, char** argv) {

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);

    setupOpenCL();

    int result = Catch::Session().run(argc, argv);

    MPI_Finalize();

    return result;
}

TEST_CASE("1 buffer data exchange") {

    std::vector<int> sendIndices1 = {0,1,2,3,4};
    std::vector<int> recvIndices1 = {5,6,7,8,9};
    double *expected1 = new double[10]{1,2,3,4,5,1,2,3,4,5};
    REQUIRE(test_1buf(sendIndices1, recvIndices1, expected1) == 0);

    std::vector<int> sendIndices2 = {0,2,4,6,8};
    std::vector<int> recvIndices2 = {1,3,5,7,9};
    double *expected2 = new double[10]{1,1,3,3,5,5,7,7,9,9};
    REQUIRE(test_1buf(sendIndices2, recvIndices2, expected2) == 0);

    std::vector<int> sendIndices3 = {0,2,4,7};
    std::vector<int> recvIndices3 = {0,2,4,7};
    double *expected3 = new double[10]{1,2,3,4,5,6,7,8,9,10};
    REQUIRE(test_1buf(sendIndices3, recvIndices3, expected3) == 0);

    std::vector<int> sendIndices4 = {};
    std::vector<int> recvIndices4 = {};
    double *expected4 = new double[10]{1,2,3,4,5,6,7,8,9,10};
    REQUIRE(test_1buf(sendIndices4, recvIndices4, expected4) == 0);

    delete[] expected1;
    delete[] expected2;
    delete[] expected3;
    delete[] expected4;

}

TEST_CASE("2 buffer data exchange") {

    std::vector<int> sendIndices1 = {0,1,2,3,4};
    std::vector<int> recvIndices1 = {5,6,7,8,9};
    double *expected1_1 = new double[10]{1,2,3,4,5,11,12,13,14,15};
    double *expected1_2 = new double[10]{11,12,13,14,15,1,2,3,4,5};
    REQUIRE(test_2buf(sendIndices1, recvIndices1, expected1_1, expected1_2) == 0);

    std::vector<int> sendIndices2 = {0,2,4,6,8};
    std::vector<int> recvIndices2 = {1,3,5,7,9};
    double *expected2_1 = new double[10]{1,11,3,13,5,15,7,17,9,19};
    double *expected2_2 = new double[10]{11,1,13,3,15,5,17,7,19,9};
    REQUIRE(test_2buf(sendIndices2, recvIndices2, expected2_1, expected2_2) == 0);

    std::vector<int> sendIndices3 = {0,2,4,7};
    std::vector<int> recvIndices3 = {0,2,4,7};
    double *expected3_1 = new double[10]{11,2,13,4,15,6,7,18,9,10};
    double *expected3_2 = new double[10]{1,12,3,14,5,16,17,8,19,20};
    REQUIRE(test_2buf(sendIndices3, recvIndices3, expected3_1, expected3_2) == 0);

    std::vector<int> sendIndices4 = {};
    std::vector<int> recvIndices4 = {};
    double *expected4_1 = new double[10]{1,2,3,4,5,6,7,8,9,10};
    double *expected4_2 = new double[10]{11,12,13,14,15,16,17,18,19,20};
    REQUIRE(test_2buf(sendIndices4, recvIndices4, expected4_1, expected4_2) == 0);

    delete[] expected1_1;
    delete[] expected1_2;
    delete[] expected2_1;
    delete[] expected2_2;
    delete[] expected3_1;
    delete[] expected3_2;
    delete[] expected4_1;
    delete[] expected4_2;

}

void setupOpenCL() {
    try {

        std::vector<cl::Platform> all_platforms;
        cl::Platform::get(&all_platforms);
        cl::Platform tauschcl_platform = all_platforms[0];

        std::vector<cl::Device> all_devices;
        tauschcl_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
        tauschcl_device = all_devices[0];

        std::cout << "Using OpenCL device " << tauschcl_device.getInfo<CL_DEVICE_NAME>() << std::endl;

        // Create context and queue
        tauschcl_context = cl::Context({tauschcl_device});
        tauschcl_queue = cl::CommandQueue(tauschcl_context,tauschcl_device);

    } catch(cl::Error error) {
        std::cout << "[setup] OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}
