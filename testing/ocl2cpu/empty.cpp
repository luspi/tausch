#include <catch2/catch.hpp>
#define TAUSCH_OPENCL
#include "../../tausch.h"
#include "ocl.h"

TEST_CASE("1 buffer, empty indices, with pack/unpack, same MPI rank") {

    setupOpenCL();

    const std::vector<int> sizes = {3, 10, 100, 377};
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
            cl::Buffer cl_in(tauschcl_queue, in, &in[(size+2*halowidth)*(size+2*halowidth)], false);

            std::vector<int> sendIndices;
            std::vector<int> recvIndices;

            Tausch *tausch = new Tausch(MPI_COMM_WORLD, false);
            tausch->setOpenCL(tauschcl_device, tauschcl_context, tauschcl_queue);

            int mpiRank;
            MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

            tausch->addSendHaloInfo(sendIndices, sizeof(double));
            tausch->addRecvHaloInfo(recvIndices, sizeof(double));

            tausch->packSendBufferOCL(0, 0, cl_in);
            tausch->send(0, 0, mpiRank, false);
            tausch->recv(0, 0, mpiRank, true);
            tausch->unpackRecvBuffer(0, 0, out);

            double *expected = new double[(size+2*halowidth)*(size+2*halowidth)]{};
            for(int i = 0; i < size; ++i) {
                for(int j = 0; j < size; ++j) {
                    expected[(i+halowidth)*(size+2*halowidth) + j+halowidth] = i*size + j + 1;
                }
            }

            // check result
            for(int i = 0; i < (size+2*halowidth); ++i)
                for(int j = 0; j < (size+2*halowidth); ++j)
                    REQUIRE(expected[i*(size+2*halowidth)+j] == out[i*(size+2*halowidth)+j]);

        }

    }

}

TEST_CASE("1 buffer, empty indices, with pack/unpack, multiple MPI ranks") {

    setupOpenCL();

    const std::vector<int> sizes = {3, 10, 100, 377};
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
            cl::Buffer cl_in(tauschcl_queue, in, &in[(size+2*halowidth)*(size+2*halowidth)], false);

            std::vector<int> sendIndices;
            std::vector<int> recvIndices;

            Tausch *tausch = new Tausch(MPI_COMM_WORLD, false);
            tausch->setOpenCL(tauschcl_device, tauschcl_context, tauschcl_queue);

            int mpiRank, mpiSize;
            MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
            MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

            tausch->addSendHaloInfo(sendIndices, sizeof(double));
            tausch->addRecvHaloInfo(recvIndices, sizeof(double));

            tausch->packSendBufferOCL(0, 0, cl_in);
            tausch->send(0, 0, (mpiRank+1)%mpiSize, false);
            tausch->recv(0, 0, (mpiRank+mpiSize-1)%mpiSize, true);
            tausch->unpackRecvBuffer(0, 0, out);

            double *expected = new double[(size+2*halowidth)*(size+2*halowidth)]{};
            for(int i = 0; i < size; ++i) {
                for(int j = 0; j < size; ++j) {
                    expected[(i+halowidth)*(size+2*halowidth) + j+halowidth] = i*size + j + 1;
                }
            }

            // check result
            for(int i = 0; i < (size+2*halowidth); ++i)
                for(int j = 0; j < (size+2*halowidth); ++j)
                    REQUIRE(expected[i*(size+2*halowidth)+j] == out[i*(size+2*halowidth)+j]);

        }

    }

}
/*
TEST_CASE("1 buffer, empty indices, derived, same MPI rank") {

    setupOpenCL();

    const std::vector<int> sizes = {3, 10, 100, 377};
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
            cl::Buffer cl_in(tauschcl_queue, in, &in[(size+2*halowidth)*(size+2*halowidth)], false);

            std::vector<int> sendIndices;
            std::vector<int> recvIndices;

            Tausch<double> *tausch = new Tausch<double>(tauschcl_device, tauschcl_context, tauschcl_queue, MPI_DOUBLE, MPI_COMM_WORLD, false);

            int mpiRank;
            MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

            tausch->addLocalHaloInfo(sendIndices, 1, -1, TauschOptimizationHint::ReceiverUsesMpiDerivedDatatype);
            tausch->addRemoteHaloInfo(recvIndices, 1, -1, TauschOptimizationHint::UseMpiDerivedDatatype);

            tausch->packSendBuffer(0, 0, cl_in);
            tausch->send(0, 0, mpiRank, false);
            tausch->recv(0, 0, mpiRank, 0, out, true);

            double *expected = new double[(size+2*halowidth)*(size+2*halowidth)]{};
            for(int i = 0; i < size; ++i) {
                for(int j = 0; j < size; ++j) {
                    expected[(i+halowidth)*(size+2*halowidth) + j+halowidth] = i*size + j + 1;
                }
            }

            // check result
            for(int i = 0; i < (size+2*halowidth); ++i)
                for(int j = 0; j < (size+2*halowidth); ++j)
                    REQUIRE(expected[i*(size+2*halowidth)+j] == out[i*(size+2*halowidth)+j]);

        }

    }

}

TEST_CASE("1 buffer, empty indices, derived, multiple MPI ranks") {

    setupOpenCL();

    const std::vector<int> sizes = {3, 10, 100, 377};
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
            cl::Buffer cl_in(tauschcl_queue, in, &in[(size+2*halowidth)*(size+2*halowidth)], false);

            std::vector<int> sendIndices;
            std::vector<int> recvIndices;

            Tausch<double> *tausch = new Tausch<double>(tauschcl_device, tauschcl_context, tauschcl_queue, MPI_DOUBLE, MPI_COMM_WORLD, false);

            int mpiRank, mpiSize;
            MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
            MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

            tausch->addLocalHaloInfo(sendIndices, 1, -1, TauschOptimizationHint::ReceiverUsesMpiDerivedDatatype);
            tausch->addRemoteHaloInfo(recvIndices, 1, -1, TauschOptimizationHint::UseMpiDerivedDatatype);

            tausch->packSendBuffer(0, 0, cl_in);
            tausch->send(0, 0, (mpiRank+1)%mpiSize, false);
            tausch->recv(0, 0, (mpiRank+mpiSize-1)%mpiSize, 0, out, true);

            double *expected = new double[(size+2*halowidth)*(size+2*halowidth)]{};
            for(int i = 0; i < size; ++i) {
                for(int j = 0; j < size; ++j) {
                    expected[(i+halowidth)*(size+2*halowidth) + j+halowidth] = i*size + j + 1;
                }
            }

            // check result
            for(int i = 0; i < (size+2*halowidth); ++i)
                for(int j = 0; j < (size+2*halowidth); ++j)
                    REQUIRE(expected[i*(size+2*halowidth)+j] == out[i*(size+2*halowidth)+j]);

        }

    }

}
*/
