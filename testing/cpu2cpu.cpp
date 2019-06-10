#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

#include <mpi.h>
#include "../tausch.h"

int test_1buf(std::vector<int> sendIndices, std::vector<int> recvIndices, std::vector<int> expected, bool useDerivedMpiDatatype) {

    Tausch<double> *tausch = new Tausch<double>(MPI_DOUBLE, MPI_COMM_WORLD, true);

    double *buf1 = new double[10]{};
    for(int i = 0; i < 10; ++i)
        buf1[i] = i + 1;

    if(useDerivedMpiDatatype) {

        tausch->addLocalHaloInfo(sendIndices, 1, -1, TauschOptimizationHint::UseMpiDerivedDatatype);
        tausch->addRemoteHaloInfo(recvIndices, 1, -1, TauschOptimizationHint::UseMpiDerivedDatatype);

        tausch->send(0, 0, 0, buf1);
        tausch->recv(0, 0, 0, buf1);

    } else {

        tausch->addLocalHaloInfo(sendIndices, 1);
        tausch->addRemoteHaloInfo(recvIndices, 1);

        tausch->packSendBuffer(0, 0, buf1);

        tausch->send(0, 0, 0);
        tausch->recv(0, 0, 0);

        tausch->unpackRecvBuffer(0, 0, buf1);

    }

    delete tausch;

    for(int i = 0; i < 10; ++i) {
        if(fabs(expected[i]-buf1[i]) > 1e-10)
            return 1;
    }

    delete[] buf1;

    return 0;

}

int test_2buf(std::vector<int> sendIndices, std::vector<int> recvIndices, std::vector<int> expected1, std::vector<int> expected2, bool useDerivedMpiDatatype) {

    Tausch<double> *tausch = new Tausch<double>(MPI_DOUBLE, MPI_COMM_WORLD, true);

    double *buf1 = new double[10]{};
    double *buf2 = new double[10]{};
    for(int i = 0; i < 10; ++i) {
        buf1[i] = i + 1;
        buf2[i] = i + 11;
    }

    if(useDerivedMpiDatatype) {

        tausch->addLocalHaloInfo(sendIndices, 2, -1, TauschOptimizationHint::UseMpiDerivedDatatype);
        tausch->addRemoteHaloInfo(recvIndices, 2, -1, TauschOptimizationHint::UseMpiDerivedDatatype);

        tausch->send(0, 0, 0, buf1);
        tausch->send(0, 1, 0, buf2);
        tausch->recv(0, 0, 0, buf2);
        tausch->recv(0, 1, 0, buf1);


    } else {

        tausch->addLocalHaloInfo(sendIndices, 2);
        tausch->addRemoteHaloInfo(recvIndices, 2);

        tausch->packSendBuffer(0, 0, buf1);
        tausch->packSendBuffer(0, 1, buf2);

        tausch->send(0, 0, 0);
        tausch->recv(0, 0, 0);

        tausch->unpackRecvBuffer(0, 0, buf2);
        tausch->unpackRecvBuffer(0, 1, buf1);

    }

    delete tausch;

    for(int i = 0; i < 10; ++i) {
        if(fabs(expected1[i]-buf1[i]) > 1e-10)
            return 1;
        if(fabs(expected2[i]-buf2[i]) > 1e-10)
            return 1;
    }

    delete[] buf1;
    delete[] buf2;

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
    std::vector<int> expected1 = {1,2,3,4,5,1,2,3,4,5};
    REQUIRE(test_1buf(sendIndices1, recvIndices1, expected1, false) == 0);
    sendIndices1 = {0,1,2,3,4};
    recvIndices1 = {5,6,7,8,9};
    REQUIRE(test_1buf(sendIndices1, recvIndices1, expected1, true) == 0);

    std::vector<int> sendIndices2 = {0,2,4,6,8};
    std::vector<int> recvIndices2 = {1,3,5,7,9};
    std::vector<int> expected2 = {1,1,3,3,5,5,7,7,9,9};
    REQUIRE(test_1buf(sendIndices2, recvIndices2, expected2, false) == 0);
    sendIndices2 = {0,2,4,6,8};
    recvIndices2 = {1,3,5,7,9};
    REQUIRE(test_1buf(sendIndices2, recvIndices2, expected2, true) == 0);

    std::vector<int> sendIndices3 = {0,2,4,7};
    std::vector<int> recvIndices3 = {0,2,4,7};
    std::vector<int> expected3 = {1,2,3,4,5,6,7,8,9,10};
    REQUIRE(test_1buf(sendIndices3, recvIndices3, expected3, false) == 0);
    sendIndices3 = {0,2,4,7};
    recvIndices3 = {0,2,4,7};
    REQUIRE(test_1buf(sendIndices3, recvIndices3, expected3, true) == 0);

    std::vector<int> sendIndices4 = {};
    std::vector<int> recvIndices4 = {};
    std::vector<int> expected4 = {1,2,3,4,5,6,7,8,9,10};
    REQUIRE(test_1buf(sendIndices4, recvIndices4, expected4, false) == 0);
    sendIndices4 = {};
    recvIndices4 = {};
    REQUIRE(test_1buf(sendIndices4, recvIndices4, expected4, true) == 0);

    std::vector<int> sendIndices5 = {0,3,4,6,7};
    std::vector<int> recvIndices5 = {1,2,3,4,5};
    std::vector<int> expected5 = {1,1,4,5,7,8,7,8,9,10};
    REQUIRE(test_1buf(sendIndices5, recvIndices5, expected5, false) == 0);
    sendIndices5 = {0,3,4,6,7};
    recvIndices5 = {1,2,3,4,5};
    REQUIRE(test_1buf(sendIndices5, recvIndices5, expected5, true) == 0);

}

TEST_CASE("2 buffer data exchange") {

    std::vector<int> sendIndices1 = {0,1,2,3,4};
    std::vector<int> recvIndices1 = {5,6,7,8,9};
    std::vector<int> expected1_1 = {1,2,3,4,5,11,12,13,14,15};
    std::vector<int> expected1_2 = {11,12,13,14,15,1,2,3,4,5};
    REQUIRE(test_2buf(sendIndices1, recvIndices1, expected1_1, expected1_2, false) == 0);
    sendIndices1 = {0,1,2,3,4};
    recvIndices1 = {5,6,7,8,9};
    REQUIRE(test_2buf(sendIndices1, recvIndices1, expected1_1, expected1_2, true) == 0);

    std::vector<int> sendIndices2 = {0,2,4,6,8};
    std::vector<int> recvIndices2 = {1,3,5,7,9};
    std::vector<int> expected2_1 = {1,11,3,13,5,15,7,17,9,19};
    std::vector<int> expected2_2 = {11,1,13,3,15,5,17,7,19,9};
    REQUIRE(test_2buf(sendIndices2, recvIndices2, expected2_1, expected2_2, false) == 0);
    sendIndices2 = {0,2,4,6,8};
    recvIndices2 = {1,3,5,7,9};
    REQUIRE(test_2buf(sendIndices2, recvIndices2, expected2_1, expected2_2, true) == 0);

    std::vector<int> sendIndices3 = {0,2,4,7};
    std::vector<int> recvIndices3 = {0,2,4,7};
    std::vector<int> expected3_1 = {11,2,13,4,15,6,7,18,9,10};
    std::vector<int> expected3_2 = {1,12,3,14,5,16,17,8,19,20};
    REQUIRE(test_2buf(sendIndices3, recvIndices3, expected3_1, expected3_2, false) == 0);
    sendIndices3 = {0,2,4,7};
    recvIndices3 = {0,2,4,7};
    REQUIRE(test_2buf(sendIndices3, recvIndices3, expected3_1, expected3_2, true) == 0);

    std::vector<int> sendIndices4 = {};
    std::vector<int> recvIndices4 = {};
    std::vector<int> expected4_1 = {1,2,3,4,5,6,7,8,9,10};
    std::vector<int> expected4_2 = {11,12,13,14,15,16,17,18,19,20};
    REQUIRE(test_2buf(sendIndices4, recvIndices4, expected4_1, expected4_2, false) == 0);
    sendIndices4 = {};
    recvIndices4 = {};
    REQUIRE(test_2buf(sendIndices4, recvIndices4, expected4_1, expected4_2, true) == 0);

    std::vector<int> sendIndices5 = {0,3,4,6,7};
    std::vector<int> recvIndices5 = {1,2,3,4,5};
    std::vector<int> expected5_1 = {1,11,14,15,17,18,7,8,9,10};
    std::vector<int> expected5_2 = {11,1,4,5,7,8,17,18,19,20};
    REQUIRE(test_2buf(sendIndices5, recvIndices5, expected5_1, expected5_2, false) == 0);
    sendIndices5 = {0,3,4,6,7};
    recvIndices5 = {1,2,3,4,5};
    REQUIRE(test_2buf(sendIndices5, recvIndices5, expected5_1, expected5_2, true) == 0);

}


