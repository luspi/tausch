#include <catch2/catch.hpp>
#define TAUSCH_HIP
#include "../../tausch.h"

TEST_CASE("1 buffer, with pack/unpack, same MPI rank") {

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
            double *hip_in, *hip_out;
            hipMalloc(&hip_in, (size+2*halowidth)*(size+2*halowidth)*sizeof(double));
            hipMalloc(&hip_out, (size+2*halowidth)*(size+2*halowidth)*sizeof(double));
            hipMemcpy(hip_in, in, (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyHostToDevice);
            hipMemcpy(hip_out, out, (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyHostToDevice);

            std::vector<int> sendIndices;
            std::vector<int> recvIndices;
            // bottom edge
            for(int i = 0; i < size; ++i)
                for(int j = 0; j < halowidth; ++j) {
                    sendIndices.push_back((j+halowidth)*(size+2*halowidth) + i+halowidth);
                    recvIndices.push_back(j*(size+2*halowidth) + i+halowidth);
                }
            // left edge
            for(int i = 0; i < halowidth; ++i)
                for(int j = 0; j < size; ++j) {
                    sendIndices.push_back((j+halowidth)*(size+2*halowidth) + i+halowidth);
                    recvIndices.push_back((j+halowidth)*(size+2*halowidth) + i);
                }
            // right edge
            for(int i = 0; i < halowidth; ++i)
                for(int j = 0; j < size; ++j) {
                    sendIndices.push_back((j+halowidth)*(size+2*halowidth) + i+size);
                    recvIndices.push_back((j+halowidth)*(size+2*halowidth) + i+(size+halowidth));
                }
            // top edge
            for(int i = 0; i < size; ++i)
                for(int j = 0; j < halowidth; ++j) {
                    sendIndices.push_back((j+size)*(size+2*halowidth) + i+halowidth);
                    recvIndices.push_back((j+(size+halowidth))*(size+2*halowidth) + i+halowidth);
                }

            Tausch *tausch = new Tausch(MPI_COMM_WORLD, false);

            int mpiRank;
            MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

            tausch->addSendHaloInfo(sendIndices, sizeof(double));
            tausch->addRecvHaloInfo(recvIndices, sizeof(double));

            tausch->packSendBufferHIP(0, 0, hip_in);
            tausch->send(0, 0, mpiRank);
            tausch->recv(0, 0, mpiRank);
            tausch->unpackRecvBufferHIP(0, 0, hip_out);

            hipMemcpy(out, hip_out, (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyDeviceToHost);

            double *expected = new double[(size+2*halowidth)*(size+2*halowidth)]{};
            for(int i = 0; i < size; ++i) {
                for(int j = 0; j < size; ++j) {
                    expected[(i+halowidth)*(size+2*halowidth) + j+halowidth] = i*size + j + 1;
                }
            }

            for(int i = 0; i < size; ++i)
                for(int j = 0; j < halowidth; ++j) {
                    expected[j*(size+2*halowidth) + i+halowidth] = j*size+i+1;  // bottom
                    expected[(j+size+halowidth)*(size+2*halowidth) + i+halowidth] = (j+(size-halowidth))*size + i+1; // top
                    expected[(i+halowidth)*(size+2*halowidth) + j] = i*size+j+1;    // left
                    expected[(i+halowidth)*(size+2*halowidth) + j+(size+halowidth)] = i*size + (size-halowidth)+j+1;    // left
                }

            // check result
            for(int i = 0; i < (size+2*halowidth); ++i)
                for(int j = 0; j < (size+2*halowidth); ++j)
                    REQUIRE(expected[i*(size+2*halowidth)+j] == out[i*(size+2*halowidth)+j]);


        }

    }

}

TEST_CASE("1 buffer, with pack/unpack, same MPI rank, GPUMultiCopy") {

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
            double *hip_in, *hip_out;
            hipMalloc(&hip_in, (size+2*halowidth)*(size+2*halowidth)*sizeof(double));
            hipMalloc(&hip_out, (size+2*halowidth)*(size+2*halowidth)*sizeof(double));
            hipMemcpy(hip_in, in, (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyHostToDevice);
            hipMemcpy(hip_out, out, (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyHostToDevice);

            std::vector<int> sendIndices;
            std::vector<int> recvIndices;
            // bottom edge
            for(int i = 0; i < size; ++i)
                for(int j = 0; j < halowidth; ++j) {
                    sendIndices.push_back((j+halowidth)*(size+2*halowidth) + i+halowidth);
                    recvIndices.push_back(j*(size+2*halowidth) + i+halowidth);
                }
            // left edge
            for(int i = 0; i < halowidth; ++i)
                for(int j = 0; j < size; ++j) {
                    sendIndices.push_back((j+halowidth)*(size+2*halowidth) + i+halowidth);
                    recvIndices.push_back((j+halowidth)*(size+2*halowidth) + i);
                }
            // right edge
            for(int i = 0; i < halowidth; ++i)
                for(int j = 0; j < size; ++j) {
                    sendIndices.push_back((j+halowidth)*(size+2*halowidth) + i+size);
                    recvIndices.push_back((j+halowidth)*(size+2*halowidth) + i+(size+halowidth));
                }
            // top edge
            for(int i = 0; i < size; ++i)
                for(int j = 0; j < halowidth; ++j) {
                    sendIndices.push_back((j+size)*(size+2*halowidth) + i+halowidth);
                    recvIndices.push_back((j+(size+halowidth))*(size+2*halowidth) + i+halowidth);
                }

            Tausch *tausch = new Tausch(MPI_COMM_WORLD, false);

            int mpiRank;
            MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

            tausch->addSendHaloInfo(sendIndices, sizeof(double));
            tausch->addRecvHaloInfo(recvIndices, sizeof(double));

            tausch->setSendCommunicationStrategy(0, Tausch::Communication::GPUMultiCopy);
            tausch->setRecvCommunicationStrategy(0, Tausch::Communication::GPUMultiCopy);

            tausch->packSendBufferHIP(0, 0, hip_in);
            tausch->send(0, 0, mpiRank);
            tausch->recv(0, 0, mpiRank);
            tausch->unpackRecvBufferHIP(0, 0, hip_out);

            hipMemcpy(out, hip_out, (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyDeviceToHost);

            double *expected = new double[(size+2*halowidth)*(size+2*halowidth)]{};
            for(int i = 0; i < size; ++i) {
                for(int j = 0; j < size; ++j) {
                    expected[(i+halowidth)*(size+2*halowidth) + j+halowidth] = i*size + j + 1;
                }
            }

            for(int i = 0; i < size; ++i)
                for(int j = 0; j < halowidth; ++j) {
                    expected[j*(size+2*halowidth) + i+halowidth] = j*size+i+1;  // bottom
                    expected[(j+size+halowidth)*(size+2*halowidth) + i+halowidth] = (j+(size-halowidth))*size + i+1; // top
                    expected[(i+halowidth)*(size+2*halowidth) + j] = i*size+j+1;    // left
                    expected[(i+halowidth)*(size+2*halowidth) + j+(size+halowidth)] = i*size + (size-halowidth)+j+1;    // left
                }

            // check result
            for(int i = 0; i < (size+2*halowidth); ++i)
                for(int j = 0; j < (size+2*halowidth); ++j)
                    REQUIRE(expected[i*(size+2*halowidth)+j] == out[i*(size+2*halowidth)+j]);


        }

    }

}

TEST_CASE("1 buffer, with pack/unpack, multiple MPI ranks") {

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
            double *hip_in, *hip_out;
            hipMalloc(&hip_in, (size+2*halowidth)*(size+2*halowidth)*sizeof(double));
            hipMalloc(&hip_out, (size+2*halowidth)*(size+2*halowidth)*sizeof(double));
            hipMemcpy(hip_in, in, (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyHostToDevice);
            hipMemcpy(hip_out, out, (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyHostToDevice);

            std::vector<int> sendIndices;
            std::vector<int> recvIndices;
            // bottom edge
            for(int i = 0; i < size; ++i)
                for(int j = 0; j < halowidth; ++j) {
                    sendIndices.push_back((j+halowidth)*(size+2*halowidth) + i+halowidth);
                    recvIndices.push_back(j*(size+2*halowidth) + i+halowidth);
                }
            // left edge
            for(int i = 0; i < halowidth; ++i)
                for(int j = 0; j < size; ++j) {
                    sendIndices.push_back((j+halowidth)*(size+2*halowidth) + i+halowidth);
                    recvIndices.push_back((j+halowidth)*(size+2*halowidth) + i);
                }
            // right edge
            for(int i = 0; i < halowidth; ++i)
                for(int j = 0; j < size; ++j) {
                    sendIndices.push_back((j+halowidth)*(size+2*halowidth) + i+size);
                    recvIndices.push_back((j+halowidth)*(size+2*halowidth) + i+(size+halowidth));
                }
            // top edge
            for(int i = 0; i < size; ++i)
                for(int j = 0; j < halowidth; ++j) {
                    sendIndices.push_back((j+size)*(size+2*halowidth) + i+halowidth);
                    recvIndices.push_back((j+(size+halowidth))*(size+2*halowidth) + i+halowidth);
                }

            Tausch *tausch = new Tausch(MPI_COMM_WORLD, false);

            int mpiRank, mpiSize;
            MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
            MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

            tausch->addSendHaloInfo(sendIndices, sizeof(double));
            tausch->addRecvHaloInfo(recvIndices, sizeof(double));

            tausch->packSendBufferHIP(0, 0, hip_in);
            tausch->send(0, 0, (mpiRank+1)%mpiSize, false);
            tausch->recv(0, 0, (mpiRank+mpiSize-1)%mpiSize, true);
            tausch->unpackRecvBufferHIP(0, 0, hip_out);

            hipMemcpy(out, hip_out, (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyDeviceToHost);

            double *expected = new double[(size+2*halowidth)*(size+2*halowidth)]{};
            for(int i = 0; i < size; ++i) {
                for(int j = 0; j < size; ++j) {
                    expected[(i+halowidth)*(size+2*halowidth) + j+halowidth] = i*size + j + 1;
                }
            }

            for(int i = 0; i < size; ++i)
                for(int j = 0; j < halowidth; ++j) {
                    expected[j*(size+2*halowidth) + i+halowidth] = j*size+i+1;  // bottom
                    expected[(j+size+halowidth)*(size+2*halowidth) + i+halowidth] = (j+(size-halowidth))*size + i+1; // top
                    expected[(i+halowidth)*(size+2*halowidth) + j] = i*size+j+1;    // left
                    expected[(i+halowidth)*(size+2*halowidth) + j+(size+halowidth)] = i*size + (size-halowidth)+j+1;    // left
                }

            // check result
            for(int i = 0; i < (size+2*halowidth); ++i)
                for(int j = 0; j < (size+2*halowidth); ++j)
                    REQUIRE(expected[i*(size+2*halowidth)+j] == out[i*(size+2*halowidth)+j]);


        }

    }

}

TEST_CASE("2 buffers, with pack/unpack, same MPI rank") {

    const std::vector<int> sizes = {3, 10, 100, 377};
    const std::vector<int> halowidths = {1, 2, 3};

    for(auto size : sizes) {

        for(auto halowidth : halowidths) {

            double *in1 = new double[(size+2*halowidth)*(size+2*halowidth)]{};
            double *in2 = new double[(size+2*halowidth)*(size+2*halowidth)]{};
            double *out1 = new double[(size+2*halowidth)*(size+2*halowidth)]{};
            double *out2 = new double[(size+2*halowidth)*(size+2*halowidth)]{};
            for(int i = 0; i < size; ++i) {
                for(int j = 0; j < size; ++j) {
                    in1[(i+halowidth)*(size+2*halowidth) + j+halowidth] = i*size + j + 1;
                    in2[(i+halowidth)*(size+2*halowidth) + j+halowidth] = size*size + i*size + j + 1;
                    out1[(i+halowidth)*(size+2*halowidth) + j+halowidth] = i*size + j + 1;
                    out2[(i+halowidth)*(size+2*halowidth) + j+halowidth] = size*size + i*size + j + 1;
                }
            }
            double *hip_in1, *hip_in2, *hip_out1, *hip_out2;
            hipMalloc(&hip_in1, (size+2*halowidth)*(size+2*halowidth)*sizeof(double));
            hipMalloc(&hip_in2, (size+2*halowidth)*(size+2*halowidth)*sizeof(double));
            hipMalloc(&hip_out1, (size+2*halowidth)*(size+2*halowidth)*sizeof(double));
            hipMalloc(&hip_out2, (size+2*halowidth)*(size+2*halowidth)*sizeof(double));
            hipMemcpy(hip_in1, in1, (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyHostToDevice);
            hipMemcpy(hip_in2, in2, (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyHostToDevice);
            hipMemcpy(hip_out1, out1, (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyHostToDevice);
            hipMemcpy(hip_out2, out2, (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyHostToDevice);

            std::vector<int> sendIndices;
            std::vector<int> recvIndices;
            // bottom edge
            for(int i = 0; i < size; ++i)
                for(int j = 0; j < halowidth; ++j) {
                    sendIndices.push_back((j+halowidth)*(size+2*halowidth) + i+halowidth);
                    recvIndices.push_back(j*(size+2*halowidth) + i+halowidth);
                }
            // left edge
            for(int i = 0; i < halowidth; ++i)
                for(int j = 0; j < size; ++j) {
                    sendIndices.push_back((j+halowidth)*(size+2*halowidth) + i+halowidth);
                    recvIndices.push_back((j+halowidth)*(size+2*halowidth) + i);
                }
            // right edge
            for(int i = 0; i < halowidth; ++i)
                for(int j = 0; j < size; ++j) {
                    sendIndices.push_back((j+halowidth)*(size+2*halowidth) + i+size);
                    recvIndices.push_back((j+halowidth)*(size+2*halowidth) + i+(size+halowidth));
                }
            // top edge
            for(int i = 0; i < size; ++i)
                for(int j = 0; j < halowidth; ++j) {
                    sendIndices.push_back((j+size)*(size+2*halowidth) + i+halowidth);
                    recvIndices.push_back((j+(size+halowidth))*(size+2*halowidth) + i+halowidth);
                }

            Tausch *tausch = new Tausch(MPI_COMM_WORLD, false);

            int mpiRank;
            MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

            tausch->addSendHaloInfos(sendIndices, sizeof(double), 2);
            tausch->addRecvHaloInfos(recvIndices, sizeof(double), 2);

            tausch->packSendBufferHIP(0, 0, hip_in1);
            tausch->packSendBufferHIP(0, 1, hip_in2);

            tausch->send(0, 0, mpiRank);
            tausch->recv(0, 0, mpiRank);

            tausch->unpackRecvBufferHIP(0, 0, hip_out2);
            tausch->unpackRecvBufferHIP(0, 1, hip_out1);

            hipMemcpy(out1, hip_out1, (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyDeviceToHost);
            hipMemcpy(out2, hip_out2, (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyDeviceToHost);

            double *expected1 = new double[(size+2*halowidth)*(size+2*halowidth)]{};
            double *expected2 = new double[(size+2*halowidth)*(size+2*halowidth)]{};
            for(int i = 0; i < size; ++i) {
                for(int j = 0; j < size; ++j) {
                    expected1[(i+halowidth)*(size+2*halowidth) + j+halowidth] = i*size + j + 1;
                    expected2[(i+halowidth)*(size+2*halowidth) + j+halowidth] = size*size + i*size + j + 1;
                }
            }

            for(int i = 0; i < size; ++i)
                for(int j = 0; j < halowidth; ++j) {
                    expected1[j*(size+2*halowidth) + i+halowidth] = size*size + j*size+i+1;  // bottom
                    expected1[(j+size+halowidth)*(size+2*halowidth) + i+halowidth] = size*size + (j+(size-halowidth))*size + i+1; // top
                    expected1[(i+halowidth)*(size+2*halowidth) + j] = size*size + i*size+j+1;    // left
                    expected1[(i+halowidth)*(size+2*halowidth) + j+(size+halowidth)] = size*size + i*size + (size-halowidth)+j+1;    // left

                    expected2[j*(size+2*halowidth) + i+halowidth] = j*size+i+1;  // bottom
                    expected2[(j+size+halowidth)*(size+2*halowidth) + i+halowidth] = (j+(size-halowidth))*size + i+1; // top
                    expected2[(i+halowidth)*(size+2*halowidth) + j] = i*size+j+1;    // left
                    expected2[(i+halowidth)*(size+2*halowidth) + j+(size+halowidth)] = i*size + (size-halowidth)+j+1;    // left
                }

            // check result
            for(int i = 0; i < (size+2*halowidth); ++i)
                for(int j = 0; j < (size+2*halowidth); ++j) {
                    REQUIRE(expected1[i*(size+2*halowidth)+j] == out1[i*(size+2*halowidth)+j]);
                    REQUIRE(expected2[i*(size+2*halowidth)+j] == out2[i*(size+2*halowidth)+j]);
                }


        }

    }

}

TEST_CASE("2 buffers, with pack/unpack, same MPI rank, GPUMultiCopy") {

    const std::vector<int> sizes = {3, 10, 100, 377};
    const std::vector<int> halowidths = {1, 2, 3};

    for(auto size : sizes) {

        for(auto halowidth : halowidths) {

            double *in1 = new double[(size+2*halowidth)*(size+2*halowidth)]{};
            double *in2 = new double[(size+2*halowidth)*(size+2*halowidth)]{};
            double *out1 = new double[(size+2*halowidth)*(size+2*halowidth)]{};
            double *out2 = new double[(size+2*halowidth)*(size+2*halowidth)]{};
            for(int i = 0; i < size; ++i) {
                for(int j = 0; j < size; ++j) {
                    in1[(i+halowidth)*(size+2*halowidth) + j+halowidth] = i*size + j + 1;
                    in2[(i+halowidth)*(size+2*halowidth) + j+halowidth] = size*size + i*size + j + 1;
                    out1[(i+halowidth)*(size+2*halowidth) + j+halowidth] = i*size + j + 1;
                    out2[(i+halowidth)*(size+2*halowidth) + j+halowidth] = size*size + i*size + j + 1;
                }
            }
            double *hip_in1, *hip_in2, *hip_out1, *hip_out2;
            hipMalloc(&hip_in1, (size+2*halowidth)*(size+2*halowidth)*sizeof(double));
            hipMalloc(&hip_in2, (size+2*halowidth)*(size+2*halowidth)*sizeof(double));
            hipMalloc(&hip_out1, (size+2*halowidth)*(size+2*halowidth)*sizeof(double));
            hipMalloc(&hip_out2, (size+2*halowidth)*(size+2*halowidth)*sizeof(double));
            hipMemcpy(hip_in1, in1, (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyHostToDevice);
            hipMemcpy(hip_in2, in2, (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyHostToDevice);
            hipMemcpy(hip_out1, out1, (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyHostToDevice);
            hipMemcpy(hip_out2, out2, (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyHostToDevice);

            std::vector<int> sendIndices;
            std::vector<int> recvIndices;
            // bottom edge
            for(int i = 0; i < size; ++i)
                for(int j = 0; j < halowidth; ++j) {
                    sendIndices.push_back((j+halowidth)*(size+2*halowidth) + i+halowidth);
                    recvIndices.push_back(j*(size+2*halowidth) + i+halowidth);
                }
            // left edge
            for(int i = 0; i < halowidth; ++i)
                for(int j = 0; j < size; ++j) {
                    sendIndices.push_back((j+halowidth)*(size+2*halowidth) + i+halowidth);
                    recvIndices.push_back((j+halowidth)*(size+2*halowidth) + i);
                }
            // right edge
            for(int i = 0; i < halowidth; ++i)
                for(int j = 0; j < size; ++j) {
                    sendIndices.push_back((j+halowidth)*(size+2*halowidth) + i+size);
                    recvIndices.push_back((j+halowidth)*(size+2*halowidth) + i+(size+halowidth));
                }
            // top edge
            for(int i = 0; i < size; ++i)
                for(int j = 0; j < halowidth; ++j) {
                    sendIndices.push_back((j+size)*(size+2*halowidth) + i+halowidth);
                    recvIndices.push_back((j+(size+halowidth))*(size+2*halowidth) + i+halowidth);
                }

            Tausch *tausch = new Tausch(MPI_COMM_WORLD, false);

            int mpiRank;
            MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

            tausch->addSendHaloInfos(sendIndices, sizeof(double), 2);
            tausch->addRecvHaloInfos(recvIndices, sizeof(double), 2);

            tausch->setSendCommunicationStrategy(0, Tausch::Communication::GPUMultiCopy);
            tausch->setRecvCommunicationStrategy(0, Tausch::Communication::GPUMultiCopy);

            tausch->packSendBufferHIP(0, 0, hip_in1);
            tausch->packSendBufferHIP(0, 1, hip_in2);

            tausch->send(0, 0, mpiRank);
            tausch->recv(0, 0, mpiRank);

            tausch->unpackRecvBufferHIP(0, 0, hip_out2);
            tausch->unpackRecvBufferHIP(0, 1, hip_out1);

            hipMemcpy(out1, hip_out1, (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyDeviceToHost);
            hipMemcpy(out2, hip_out2, (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyDeviceToHost);

            double *expected1 = new double[(size+2*halowidth)*(size+2*halowidth)]{};
            double *expected2 = new double[(size+2*halowidth)*(size+2*halowidth)]{};
            for(int i = 0; i < size; ++i) {
                for(int j = 0; j < size; ++j) {
                    expected1[(i+halowidth)*(size+2*halowidth) + j+halowidth] = i*size + j + 1;
                    expected2[(i+halowidth)*(size+2*halowidth) + j+halowidth] = size*size + i*size + j + 1;
                }
            }

            for(int i = 0; i < size; ++i)
                for(int j = 0; j < halowidth; ++j) {
                    expected1[j*(size+2*halowidth) + i+halowidth] = size*size + j*size+i+1;  // bottom
                    expected1[(j+size+halowidth)*(size+2*halowidth) + i+halowidth] = size*size + (j+(size-halowidth))*size + i+1; // top
                    expected1[(i+halowidth)*(size+2*halowidth) + j] = size*size + i*size+j+1;    // left
                    expected1[(i+halowidth)*(size+2*halowidth) + j+(size+halowidth)] = size*size + i*size + (size-halowidth)+j+1;    // left

                    expected2[j*(size+2*halowidth) + i+halowidth] = j*size+i+1;  // bottom
                    expected2[(j+size+halowidth)*(size+2*halowidth) + i+halowidth] = (j+(size-halowidth))*size + i+1; // top
                    expected2[(i+halowidth)*(size+2*halowidth) + j] = i*size+j+1;    // left
                    expected2[(i+halowidth)*(size+2*halowidth) + j+(size+halowidth)] = i*size + (size-halowidth)+j+1;    // left
                }

            // check result
            for(int i = 0; i < (size+2*halowidth); ++i)
                for(int j = 0; j < (size+2*halowidth); ++j) {
                    REQUIRE(expected1[i*(size+2*halowidth)+j] == out1[i*(size+2*halowidth)+j]);
                    REQUIRE(expected2[i*(size+2*halowidth)+j] == out2[i*(size+2*halowidth)+j]);
                }


        }

    }

}

TEST_CASE("2 buffers, with pack/unpack, multiple MPI ranks") {

    const std::vector<int> sizes = {3, 10, 100, 377};
    const std::vector<int> halowidths = {1, 2, 3};

    for(auto size : sizes) {

        for(auto halowidth : halowidths) {

            double *in1 = new double[(size+2*halowidth)*(size+2*halowidth)]{};
            double *in2 = new double[(size+2*halowidth)*(size+2*halowidth)]{};
            double *out1 = new double[(size+2*halowidth)*(size+2*halowidth)]{};
            double *out2 = new double[(size+2*halowidth)*(size+2*halowidth)]{};
            for(int i = 0; i < size; ++i) {
                for(int j = 0; j < size; ++j) {
                    in1[(i+halowidth)*(size+2*halowidth) + j+halowidth] = i*size + j + 1;
                    in2[(i+halowidth)*(size+2*halowidth) + j+halowidth] = size*size + i*size + j + 1;
                    out1[(i+halowidth)*(size+2*halowidth) + j+halowidth] = i*size + j + 1;
                    out2[(i+halowidth)*(size+2*halowidth) + j+halowidth] = size*size + i*size + j + 1;
                }
            }
            double *hip_in1, *hip_in2, *hip_out1, *hip_out2;
            hipMalloc(&hip_in1, (size+2*halowidth)*(size+2*halowidth)*sizeof(double));
            hipMalloc(&hip_in2, (size+2*halowidth)*(size+2*halowidth)*sizeof(double));
            hipMalloc(&hip_out1, (size+2*halowidth)*(size+2*halowidth)*sizeof(double));
            hipMalloc(&hip_out2, (size+2*halowidth)*(size+2*halowidth)*sizeof(double));
            hipMemcpy(hip_in1, in1, (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyHostToDevice);
            hipMemcpy(hip_in2, in2, (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyHostToDevice);
            hipMemcpy(hip_out1, out1, (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyHostToDevice);
            hipMemcpy(hip_out2, out2, (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyHostToDevice);

            std::vector<int> sendIndices;
            std::vector<int> recvIndices;
            // bottom edge
            for(int i = 0; i < size; ++i)
                for(int j = 0; j < halowidth; ++j) {
                    sendIndices.push_back((j+halowidth)*(size+2*halowidth) + i+halowidth);
                    recvIndices.push_back(j*(size+2*halowidth) + i+halowidth);
                }
            // left edge
            for(int i = 0; i < halowidth; ++i)
                for(int j = 0; j < size; ++j) {
                    sendIndices.push_back((j+halowidth)*(size+2*halowidth) + i+halowidth);
                    recvIndices.push_back((j+halowidth)*(size+2*halowidth) + i);
                }
            // right edge
            for(int i = 0; i < halowidth; ++i)
                for(int j = 0; j < size; ++j) {
                    sendIndices.push_back((j+halowidth)*(size+2*halowidth) + i+size);
                    recvIndices.push_back((j+halowidth)*(size+2*halowidth) + i+(size+halowidth));
                }
            // top edge
            for(int i = 0; i < size; ++i)
                for(int j = 0; j < halowidth; ++j) {
                    sendIndices.push_back((j+size)*(size+2*halowidth) + i+halowidth);
                    recvIndices.push_back((j+(size+halowidth))*(size+2*halowidth) + i+halowidth);
                }

            Tausch *tausch = new Tausch(MPI_COMM_WORLD, false);

            int mpiRank, mpiSize;
            MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
            MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

            tausch->addSendHaloInfos(sendIndices, sizeof(double), 2);
            tausch->addRecvHaloInfos(recvIndices, sizeof(double), 2);

            tausch->packSendBufferHIP(0, 0, hip_in1);
            tausch->packSendBufferHIP(0, 1, hip_in2);

            tausch->send(0, 0, (mpiRank+1)%mpiSize);
            tausch->recv(0, 0, (mpiRank+mpiSize-1)%mpiSize);

            tausch->unpackRecvBufferHIP(0, 0, hip_out2);
            tausch->unpackRecvBufferHIP(0, 1, hip_out1);

            hipMemcpy(out1, hip_out1, (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyDeviceToHost);
            hipMemcpy(out2, hip_out2, (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyDeviceToHost);

            double *expected1 = new double[(size+2*halowidth)*(size+2*halowidth)]{};
            double *expected2 = new double[(size+2*halowidth)*(size+2*halowidth)]{};
            for(int i = 0; i < size; ++i) {
                for(int j = 0; j < size; ++j) {
                    expected1[(i+halowidth)*(size+2*halowidth) + j+halowidth] = i*size + j + 1;
                    expected2[(i+halowidth)*(size+2*halowidth) + j+halowidth] = size*size + i*size + j + 1;
                }
            }

            for(int i = 0; i < size; ++i)
                for(int j = 0; j < halowidth; ++j) {
                    expected1[j*(size+2*halowidth) + i+halowidth] = size*size + j*size+i+1;  // bottom
                    expected1[(j+size+halowidth)*(size+2*halowidth) + i+halowidth] = size*size + (j+(size-halowidth))*size + i+1; // top
                    expected1[(i+halowidth)*(size+2*halowidth) + j] = size*size + i*size+j+1;    // left
                    expected1[(i+halowidth)*(size+2*halowidth) + j+(size+halowidth)] = size*size + i*size + (size-halowidth)+j+1;    // left

                    expected2[j*(size+2*halowidth) + i+halowidth] = j*size+i+1;  // bottom
                    expected2[(j+size+halowidth)*(size+2*halowidth) + i+halowidth] = (j+(size-halowidth))*size + i+1; // top
                    expected2[(i+halowidth)*(size+2*halowidth) + j] = i*size+j+1;    // left
                    expected2[(i+halowidth)*(size+2*halowidth) + j+(size+halowidth)] = i*size + (size-halowidth)+j+1;    // left
                }

            // check result
            for(int i = 0; i < (size+2*halowidth); ++i)
                for(int j = 0; j < (size+2*halowidth); ++j) {
                    REQUIRE(expected1[i*(size+2*halowidth)+j] == out1[i*(size+2*halowidth)+j]);
                    REQUIRE(expected2[i*(size+2*halowidth)+j] == out2[i*(size+2*halowidth)+j]);
                }


        }

    }

}

