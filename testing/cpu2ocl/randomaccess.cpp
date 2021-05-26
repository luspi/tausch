#include <catch2/catch.hpp>
#include <iomanip>
#define TAUSCH_OPENCL
#include "../../tausch.h"
#include "ocl.h"
/*
TEST_CASE("1 buffer, random indices, derived, same MPI rank") {

    setupOpenCL();

    const std::vector<int> sizes = {10, 100, 377};
    const std::vector<int> halowidths = {1, 2, 3};

    for(auto size : sizes) {

        for(auto halowidth : halowidths) {

            double *in = new double[(size+2*halowidth)*(size+2*halowidth)]{};
            double *out = new double[(size+2*halowidth)*(size+2*halowidth)]{};
            for(int i = 0; i < size; ++i) {
                for(int j = 0; j < size; ++j) {
                    in[(i+halowidth)*(size+2*halowidth) + j+halowidth] = i*size + j + 1;
                    out[(i+halowidth)*(size+2*halowidth) + j+halowidth] = i*size + j + 1;
                }
            }
            cl::Buffer cl_out(tauschcl_queue, out, &out[(size+2*halowidth)*(size+2*halowidth)], false);

            std::vector<int> sendIndices;
            std::vector<int> recvIndices;

            // 5 along bottom left corner
            for(int i = 0; i < 5; ++i) {
                sendIndices.push_back(halowidth*(size+2*halowidth) + i + halowidth);
                recvIndices.push_back(i + halowidth);
            }
            // 1 at bottom right corner
            sendIndices.push_back((halowidth+1)*(size+2*halowidth)-halowidth-1);
            recvIndices.push_back(halowidth*(size+2*halowidth)-halowidth-1);
            // 3 along left edge
            for(int j = 0; j < 3; ++j) {
                sendIndices.push_back((j+halowidth+1)*(size+2*halowidth)+halowidth);
                recvIndices.push_back((j+halowidth+1)*(size+2*halowidth));
            }
            // 3 along right edge
            for(int j = 0; j < 3; ++j) {
                sendIndices.push_back((j+halowidth+1)*(size+2*halowidth)+size+halowidth-1);
                recvIndices.push_back((j+halowidth+2)*(size+2*halowidth)-halowidth);
            }
            // 2 along top edge
            sendIndices.push_back((size+2*halowidth)*(size+2*halowidth)-(halowidth+1)*(size+2*halowidth)+size/2);
            sendIndices.push_back((size+2*halowidth)*(size+2*halowidth)-(halowidth+1)*(size+2*halowidth)+size/2 +1);
            recvIndices.push_back((size+2*halowidth)*(size+2*halowidth)-(size+2*halowidth)+size/2);
            recvIndices.push_back((size+2*halowidth)*(size+2*halowidth)-(size+2*halowidth)+size/2 +1);

            Tausch<double> *tausch = new Tausch<double>(tauschcl_device, tauschcl_context, tauschcl_queue, MPI_DOUBLE, MPI_COMM_WORLD, false);

            int mpiRank;
            MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

            tausch->addLocalHaloInfo(sendIndices, 1, -1, TauschOptimizationHint::UseMpiDerivedDatatype);
            tausch->addRemoteHaloInfo(recvIndices, 1, -1, TauschOptimizationHint::SenderUsesMpiDerivedDatatype);

            tausch->send(0, 0, mpiRank, 0, in, false);
            tausch->recv(0, 0, mpiRank, true);

            tausch->unpackRecvBuffer(0, 0, cl_out);

            cl::copy(tauschcl_queue, cl_out, out, &out[(size+2*halowidth)*(size+2*halowidth)]);

            double *expected = new double[(size+2*halowidth)*(size+2*halowidth)]{};
            for(int i = 0; i < size; ++i) {
                for(int j = 0; j < size; ++j) {
                    expected[(i+halowidth)*(size+2*halowidth) + j+halowidth] = i*size + j + 1;
                }
            }

            // 5 along bottom left corner
            for(int i = 0; i < 5; ++i)
                expected[i+halowidth] = i+1;
            // 1 at bottom right corner
            expected[(halowidth-1)*(size+2*halowidth)+size+halowidth-1] = size;
            // 3 along left/right edge
            for(int j = 0; j < 3; ++j) {
                expected[(j+halowidth+1)*(size+2*halowidth)] = (j+1)*size +1;
                expected[(j+halowidth+2)*(size+2*halowidth) - halowidth] = (j+2)*size;
            }
            // 2 along top edge
            expected[(size+2*halowidth)*(size+2*halowidth)-(size+2*halowidth)+size/2] = size*(size-1)+size/2-halowidth +1;
            expected[(size+2*halowidth)*(size+2*halowidth)-(size+2*halowidth)+size/2 +1] = size*(size-1)+size/2-halowidth +2;

            // check result
            for(int i = 0; i < (size+2*halowidth); ++i)
                for(int j = 0; j < (size+2*halowidth); ++j)
                    REQUIRE(expected[i*(size+2*halowidth)+j] == out[i*(size+2*halowidth)+j]);


        }

    }

}

TEST_CASE("1 buffer, random indices, derived, multiple MPI ranks") {

    setupOpenCL();

    const std::vector<int> sizes = {10, 100, 377};
    const std::vector<int> halowidths = {1, 2, 3};

    for(auto size : sizes) {

        for(auto halowidth : halowidths) {

            double *in = new double[(size+2*halowidth)*(size+2*halowidth)]{};
            double *out = new double[(size+2*halowidth)*(size+2*halowidth)]{};
            for(int i = 0; i < size; ++i) {
                for(int j = 0; j < size; ++j) {
                    in[(i+halowidth)*(size+2*halowidth) + j+halowidth] = i*size + j + 1;
                    out[(i+halowidth)*(size+2*halowidth) + j+halowidth] = i*size + j + 1;
                }
            }
            cl::Buffer cl_out(tauschcl_queue, out, &out[(size+2*halowidth)*(size+2*halowidth)], false);

            std::vector<int> sendIndices;
            std::vector<int> recvIndices;

            // 5 along bottom left corner
            for(int i = 0; i < 5; ++i) {
                sendIndices.push_back(halowidth*(size+2*halowidth) + i + halowidth);
                recvIndices.push_back(i + halowidth);
            }
            // 1 at bottom right corner
            sendIndices.push_back((halowidth+1)*(size+2*halowidth)-halowidth-1);
            recvIndices.push_back(halowidth*(size+2*halowidth)-halowidth-1);
            // 3 along left edge
            for(int j = 0; j < 3; ++j) {
                sendIndices.push_back((j+halowidth+1)*(size+2*halowidth)+halowidth);
                recvIndices.push_back((j+halowidth+1)*(size+2*halowidth));
            }
            // 3 along right edge
            for(int j = 0; j < 3; ++j) {
                sendIndices.push_back((j+halowidth+1)*(size+2*halowidth)+size+halowidth-1);
                recvIndices.push_back((j+halowidth+2)*(size+2*halowidth)-halowidth);
            }
            // 2 along top edge
            sendIndices.push_back((size+2*halowidth)*(size+2*halowidth)-(halowidth+1)*(size+2*halowidth)+size/2);
            sendIndices.push_back((size+2*halowidth)*(size+2*halowidth)-(halowidth+1)*(size+2*halowidth)+size/2 +1);
            recvIndices.push_back((size+2*halowidth)*(size+2*halowidth)-(size+2*halowidth)+size/2);
            recvIndices.push_back((size+2*halowidth)*(size+2*halowidth)-(size+2*halowidth)+size/2 +1);

            Tausch<double> *tausch = new Tausch<double>(tauschcl_device, tauschcl_context, tauschcl_queue, MPI_DOUBLE, MPI_COMM_WORLD, false);

            int mpiRank, mpiSize;
            MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
            MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

            tausch->addLocalHaloInfo(sendIndices, 1, -1, TauschOptimizationHint::UseMpiDerivedDatatype);
            tausch->addRemoteHaloInfo(recvIndices, 1, -1, TauschOptimizationHint::SenderUsesMpiDerivedDatatype);

            tausch->send(0, 0, (mpiRank+1)%mpiSize, 0, in, false);
            tausch->recv(0, 0, (mpiRank+mpiSize-1)%mpiSize, true);

            tausch->unpackRecvBuffer(0, 0, cl_out);

            cl::copy(tauschcl_queue, cl_out, out, &out[(size+2*halowidth)*(size+2*halowidth)]);

            double *expected = new double[(size+2*halowidth)*(size+2*halowidth)]{};
            for(int i = 0; i < size; ++i) {
                for(int j = 0; j < size; ++j) {
                    expected[(i+halowidth)*(size+2*halowidth) + j+halowidth] = i*size + j + 1;
                }
            }

            // 5 along bottom left corner
            for(int i = 0; i < 5; ++i)
                expected[i+halowidth] = i+1;
            // 1 at bottom right corner
            expected[(halowidth-1)*(size+2*halowidth)+size+halowidth-1] = size;
            // 3 along left/right edge
            for(int j = 0; j < 3; ++j) {
                expected[(j+halowidth+1)*(size+2*halowidth)] = (j+1)*size +1;
                expected[(j+halowidth+2)*(size+2*halowidth) - halowidth] = (j+2)*size;
            }
            // 2 along top edge
            expected[(size+2*halowidth)*(size+2*halowidth)-(size+2*halowidth)+size/2] = size*(size-1)+size/2-halowidth +1;
            expected[(size+2*halowidth)*(size+2*halowidth)-(size+2*halowidth)+size/2 +1] = size*(size-1)+size/2-halowidth +2;

            // check result
            for(int i = 0; i < (size+2*halowidth); ++i)
                for(int j = 0; j < (size+2*halowidth); ++j)
                    REQUIRE(expected[i*(size+2*halowidth)+j] == out[i*(size+2*halowidth)+j]);


        }

    }

}
*/
TEST_CASE("1 buffer, random indices, with pack/unpack, same MPI rank") {

    setupOpenCL();

    const std::vector<int> sizes = {10, 100, 377};
    const std::vector<int> halowidths = {1, 2, 3};

    for(auto size : sizes) {

        for(auto halowidth : halowidths) {

            double *in = new double[(size+2*halowidth)*(size+2*halowidth)]{};
            double *out = new double[(size+2*halowidth)*(size+2*halowidth)]{};
            for(int i = 0; i < size; ++i) {
                for(int j = 0; j < size; ++j) {
                    in[(i+halowidth)*(size+2*halowidth) + j+halowidth] = i*size + j + 1;
                    out[(i+halowidth)*(size+2*halowidth) + j+halowidth] = i*size + j + 1;
                }
            }
            cl::Buffer cl_out(tauschcl_queue, out, &out[(size+2*halowidth)*(size+2*halowidth)], false);

            std::vector<int> sendIndices;
            std::vector<int> recvIndices;

            // 5 along bottom left corner
            for(int i = 0; i < 5; ++i) {
                sendIndices.push_back(halowidth*(size+2*halowidth) + i + halowidth);
                recvIndices.push_back(i + halowidth);
            }
            // 1 at bottom right corner
            sendIndices.push_back((halowidth+1)*(size+2*halowidth)-halowidth-1);
            recvIndices.push_back(halowidth*(size+2*halowidth)-halowidth-1);
            // 3 along left edge
            for(int j = 0; j < 3; ++j) {
                sendIndices.push_back((j+halowidth+1)*(size+2*halowidth)+halowidth);
                recvIndices.push_back((j+halowidth+1)*(size+2*halowidth));
            }
            // 3 along right edge
            for(int j = 0; j < 3; ++j) {
                sendIndices.push_back((j+halowidth+1)*(size+2*halowidth)+size+halowidth-1);
                recvIndices.push_back((j+halowidth+2)*(size+2*halowidth)-halowidth);
            }
            // 2 along top edge
            sendIndices.push_back((size+2*halowidth)*(size+2*halowidth)-(halowidth+1)*(size+2*halowidth)+size/2);
            sendIndices.push_back((size+2*halowidth)*(size+2*halowidth)-(halowidth+1)*(size+2*halowidth)+size/2 +1);
            recvIndices.push_back((size+2*halowidth)*(size+2*halowidth)-(size+2*halowidth)+size/2);
            recvIndices.push_back((size+2*halowidth)*(size+2*halowidth)-(size+2*halowidth)+size/2 +1);

            Tausch *tausch = new Tausch(MPI_COMM_WORLD, false);
            tausch->setOpenCL(tauschcl_device, tauschcl_context, tauschcl_queue);

            int mpiRank;
            MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

            tausch->addSendHaloInfo(sendIndices, sizeof(double));
            tausch->addRecvHaloInfo(recvIndices, sizeof(double));

            tausch->packSendBuffer(0, 0, in);
            tausch->send(0, 0, mpiRank, false);
            tausch->recv(0, 0, mpiRank, true);
            tausch->unpackRecvBufferOCL(0, 0, cl_out);

            cl::copy(tauschcl_queue, cl_out, out, &out[(size+2*halowidth)*(size+2*halowidth)]);

            double *expected = new double[(size+2*halowidth)*(size+2*halowidth)]{};
            for(int i = 0; i < size; ++i) {
                for(int j = 0; j < size; ++j) {
                    expected[(i+halowidth)*(size+2*halowidth) + j+halowidth] = i*size + j + 1;
                }
            }

            // 5 along bottom left corner
            for(int i = 0; i < 5; ++i)
                expected[i+halowidth] = i+1;
            // 1 at bottom right corner
            expected[(halowidth-1)*(size+2*halowidth)+size+halowidth-1] = size;
            // 3 along left/right edge
            for(int j = 0; j < 3; ++j) {
                expected[(j+halowidth+1)*(size+2*halowidth)] = (j+1)*size +1;
                expected[(j+halowidth+2)*(size+2*halowidth) - halowidth] = (j+2)*size;
            }
            // 2 along top edge
            expected[(size+2*halowidth)*(size+2*halowidth)-(size+2*halowidth)+size/2] = size*(size-1)+size/2-halowidth +1;
            expected[(size+2*halowidth)*(size+2*halowidth)-(size+2*halowidth)+size/2 +1] = size*(size-1)+size/2-halowidth +2;

            // check result
            for(int i = 0; i < (size+2*halowidth); ++i)
                for(int j = 0; j < (size+2*halowidth); ++j)
                    REQUIRE(expected[i*(size+2*halowidth)+j] == out[i*(size+2*halowidth)+j]);


        }

    }

}

TEST_CASE("1 buffer, random indices, with pack/unpack, multiple MPI ranks") {

    setupOpenCL();

    const std::vector<int> sizes = {10, 100, 377};
    const std::vector<int> halowidths = {1, 2, 3};

    for(auto size : sizes) {

        for(auto halowidth : halowidths) {

            double *in = new double[(size+2*halowidth)*(size+2*halowidth)]{};
            double *out = new double[(size+2*halowidth)*(size+2*halowidth)]{};
            for(int i = 0; i < size; ++i) {
                for(int j = 0; j < size; ++j) {
                    in[(i+halowidth)*(size+2*halowidth) + j+halowidth] = i*size + j + 1;
                    out[(i+halowidth)*(size+2*halowidth) + j+halowidth] = i*size + j + 1;
                }
            }
            cl::Buffer cl_out(tauschcl_queue, out, &out[(size+2*halowidth)*(size+2*halowidth)], false);

            std::vector<int> sendIndices;
            std::vector<int> recvIndices;

            // 5 along bottom left corner
            for(int i = 0; i < 5; ++i) {
                sendIndices.push_back(halowidth*(size+2*halowidth) + i + halowidth);
                recvIndices.push_back(i + halowidth);
            }
            // 1 at bottom right corner
            sendIndices.push_back((halowidth+1)*(size+2*halowidth)-halowidth-1);
            recvIndices.push_back(halowidth*(size+2*halowidth)-halowidth-1);
            // 3 along left edge
            for(int j = 0; j < 3; ++j) {
                sendIndices.push_back((j+halowidth+1)*(size+2*halowidth)+halowidth);
                recvIndices.push_back((j+halowidth+1)*(size+2*halowidth));
            }
            // 3 along right edge
            for(int j = 0; j < 3; ++j) {
                sendIndices.push_back((j+halowidth+1)*(size+2*halowidth)+size+halowidth-1);
                recvIndices.push_back((j+halowidth+2)*(size+2*halowidth)-halowidth);
            }
            // 2 along top edge
            sendIndices.push_back((size+2*halowidth)*(size+2*halowidth)-(halowidth+1)*(size+2*halowidth)+size/2);
            sendIndices.push_back((size+2*halowidth)*(size+2*halowidth)-(halowidth+1)*(size+2*halowidth)+size/2 +1);
            recvIndices.push_back((size+2*halowidth)*(size+2*halowidth)-(size+2*halowidth)+size/2);
            recvIndices.push_back((size+2*halowidth)*(size+2*halowidth)-(size+2*halowidth)+size/2 +1);

            Tausch *tausch = new Tausch(MPI_COMM_WORLD, false);
            tausch->setOpenCL(tauschcl_device, tauschcl_context, tauschcl_queue);

            int mpiRank, mpiSize;
            MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
            MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

            tausch->addSendHaloInfo(sendIndices, sizeof(double));
            tausch->addRecvHaloInfo(recvIndices, sizeof(double));

            tausch->packSendBuffer(0, 0, in);
            tausch->send(0, 0, (mpiRank+1)%mpiSize, false);
            tausch->recv(0, 0, (mpiRank+mpiSize-1)%mpiSize, true);
            tausch->unpackRecvBufferOCL(0, 0, cl_out);

            cl::copy(tauschcl_queue, cl_out, out, &out[(size+2*halowidth)*(size+2*halowidth)]);

            double *expected = new double[(size+2*halowidth)*(size+2*halowidth)]{};
            for(int i = 0; i < size; ++i) {
                for(int j = 0; j < size; ++j) {
                    expected[(i+halowidth)*(size+2*halowidth) + j+halowidth] = i*size + j + 1;
                }
            }

            // 5 along bottom left corner
            for(int i = 0; i < 5; ++i)
                expected[i+halowidth] = i+1;
            // 1 at bottom right corner
            expected[(halowidth-1)*(size+2*halowidth)+size+halowidth-1] = size;
            // 3 along left/right edge
            for(int j = 0; j < 3; ++j) {
                expected[(j+halowidth+1)*(size+2*halowidth)] = (j+1)*size +1;
                expected[(j+halowidth+2)*(size+2*halowidth) - halowidth] = (j+2)*size;
            }
            // 2 along top edge
            expected[(size+2*halowidth)*(size+2*halowidth)-(size+2*halowidth)+size/2] = size*(size-1)+size/2-halowidth +1;
            expected[(size+2*halowidth)*(size+2*halowidth)-(size+2*halowidth)+size/2 +1] = size*(size-1)+size/2-halowidth +2;

            // check result
            for(int i = 0; i < (size+2*halowidth); ++i)
                for(int j = 0; j < (size+2*halowidth); ++j)
                    REQUIRE(expected[i*(size+2*halowidth)+j] == out[i*(size+2*halowidth)+j]);


        }

    }

}
