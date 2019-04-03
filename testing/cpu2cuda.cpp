#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

#include <mpi.h>
#define TAUSCH_CUDA
#include "../tausch.h"

int test_1buf(std::vector<int> sendIndices, std::vector<int> recvIndices, double *expected) {

    Tausch<double> *tausch = new Tausch<double>(MPI_DOUBLE, MPI_COMM_WORLD, true);

    double *buf1 = new double[10]{};
    for(int i = 0; i < 10; ++i)
        buf1[i] = i + 1;

    double *cudabuf;
    cudaMalloc(&cudabuf, 10*sizeof(double));
    cudaMemcpy(cudabuf, buf1, 10*sizeof(double), cudaMemcpyHostToDevice);

    tausch->addLocalHaloInfo(sendIndices);
    tausch->addRemoteHaloInfoCUDA(recvIndices);

    tausch->packSendBuffer(0, 0, buf1);

    tausch->send(0, 0, 0);
    tausch->recvCUDA(0, 0, 0);

    tausch->unpackRecvBufferCUDA(0, 0, cudabuf);

    delete tausch;

    double *_cudabuf = new double[10]{};
    cudaMemcpy(_cudabuf, cudabuf, 10*sizeof(double), cudaMemcpyDeviceToHost);

    for(int i = 0; i < 10; ++i)
        if(fabs(expected[i]-_cudabuf[i]) > 1e-10)
            return 1;

    delete[] buf1;
    delete[] _cudabuf;

    return 0;

}

int test_2buf(std::vector<int> sendIndices, std::vector<int> recvIndices, double *expected1, double *expected2) {

    Tausch<double> *tausch = new Tausch<double>(MPI_DOUBLE, MPI_COMM_WORLD, true);

    double *buf1 = new double[10]{};
    double *buf2 = new double[10]{};
    for(int i = 0; i < 10; ++i) {
        buf1[i] = i + 1;
        buf2[i] = i + 11;
    }
    double *cudabuf1;
    double *cudabuf2;
    cudaMalloc(&cudabuf1, 10*sizeof(double));
    cudaMalloc(&cudabuf2, 10*sizeof(double));
    cudaMemcpy(cudabuf1, buf1, 10*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cudabuf2, buf2, 10*sizeof(double), cudaMemcpyHostToDevice);

    tausch->addLocalHaloInfo(sendIndices, 2);
    tausch->addRemoteHaloInfoCUDA(recvIndices, 2);

    tausch->packSendBuffer(0, 0, buf1);
    tausch->packSendBuffer(0, 1, buf2);

    tausch->send(0, 0, 0);
    tausch->recvCUDA(0, 0, 0);

    tausch->unpackRecvBufferCUDA(0, 0, cudabuf2);
    tausch->unpackRecvBufferCUDA(0, 1, cudabuf1);

    delete tausch;

    double *_cudabuf1 = new double[10]{};
    double *_cudabuf2 = new double[10]{};
    cudaMemcpy(_cudabuf1, cudabuf1, 10*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(_cudabuf2, cudabuf2, 10*sizeof(double), cudaMemcpyDeviceToHost);

    for(int i = 0; i < 10; ++i) {
        if(fabs(expected1[i]-_cudabuf1[i]) > 1e-10)
            return 1;
        if(fabs(expected2[i]-_cudabuf2[i]) > 1e-10)
            return 1;
    }

    delete[] buf1;
    delete[] buf2;
    delete[] _cudabuf1;
    delete[] _cudabuf2;

    return 0;

}



int main(int argc, char** argv) {

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);

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
