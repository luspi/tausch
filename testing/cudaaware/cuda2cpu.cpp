#include <catch2/catch.hpp>
#define TAUSCH_CUDA
#include "../../tausch.h"

TEST_CASE("CUDA-to-CPU, 1 buffer, with pack/unpack, same MPI rank") {

    const std::vector<int> sizes = {10, 10, 100, 377};
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
            double *cuda_in;
            cudaMalloc(&cuda_in, (size+2*halowidth)*(size+2*halowidth)*sizeof(double));
            cudaMemcpy(cuda_in, in, (size+2*halowidth)*(size+2*halowidth)*sizeof(double), cudaMemcpyHostToDevice);

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

            tausch->setSendCommunicationStrategy(0, Tausch::Communication::CUDAAwareMPI);

            tausch->packSendBufferCUDA(0, 0, cuda_in);
            tausch->send(0, 0, mpiRank);
            tausch->recv(0, 0, mpiRank);
            tausch->unpackRecvBuffer(0, 0, out);

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

            delete[] in;
            delete[] out;
            cudaFree(&cuda_in);
            delete tausch;


        }

    }

}

TEST_CASE("CUDA-to-CPU, 1 buffer, with pack/unpack, multiple MPI ranks") {

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
            double *cuda_in;
            cudaMalloc(&cuda_in, (size+2*halowidth)*(size+2*halowidth)*sizeof(double));
            cudaMemcpy(cuda_in, in, (size+2*halowidth)*(size+2*halowidth)*sizeof(double), cudaMemcpyHostToDevice);

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

            tausch->setSendCommunicationStrategy(0, Tausch::Communication::CUDAAwareMPI);

            tausch->packSendBufferCUDA(0, 0, cuda_in);
            tausch->send(0, 0, (mpiRank+1)%mpiSize);
            tausch->recv(0, 0, (mpiRank+mpiSize-1)%mpiSize);
            tausch->unpackRecvBuffer(0, 0, out);

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

            delete[] in;
            delete[] out;
            cudaFree(&cuda_in);
            delete tausch;


        }

    }

}

TEST_CASE("CUDA-to-CPU, 2 buffers, with pack/unpack, same MPI rank") {

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
            double *cuda_in1, *cuda_in2;
            cudaMalloc(&cuda_in1, (size+2*halowidth)*(size+2*halowidth)*sizeof(double));
            cudaMalloc(&cuda_in2, (size+2*halowidth)*(size+2*halowidth)*sizeof(double));
            cudaMemcpy(cuda_in1, in1, (size+2*halowidth)*(size+2*halowidth)*sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(cuda_in2, in2, (size+2*halowidth)*(size+2*halowidth)*sizeof(double), cudaMemcpyHostToDevice);

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

            tausch->addSendHaloInfo(sendIndices, sizeof(double), 2);
            tausch->addRecvHaloInfo(recvIndices, sizeof(double), 2);

            tausch->setSendCommunicationStrategy(0, Tausch::Communication::CUDAAwareMPI);

            tausch->packSendBufferCUDA(0, 0, cuda_in1);
            tausch->packSendBufferCUDA(0, 1, cuda_in2);

            tausch->send(0, 0, mpiRank);
            tausch->recv(0, 0, mpiRank);

            tausch->unpackRecvBuffer(0, 0, out2);
            tausch->unpackRecvBuffer(0, 1, out1);

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

            delete[] in1;
            delete[] in2;
            delete[] out1;
            delete[] out2;
            cudaFree(&cuda_in1);
            cudaFree(&cuda_in2);
            delete tausch;


        }

    }

}

TEST_CASE("CUDA-to-CPU, 2 buffers, with pack/unpack, multiple MPI ranks") {

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
            double *cuda_in1, *cuda_in2;
            cudaMalloc(&cuda_in1, (size+2*halowidth)*(size+2*halowidth)*sizeof(double));
            cudaMalloc(&cuda_in2, (size+2*halowidth)*(size+2*halowidth)*sizeof(double));
            cudaMemcpy(cuda_in1, in1, (size+2*halowidth)*(size+2*halowidth)*sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(cuda_in2, in2, (size+2*halowidth)*(size+2*halowidth)*sizeof(double), cudaMemcpyHostToDevice);

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

            tausch->addSendHaloInfo(sendIndices, sizeof(double), 2);
            tausch->addRecvHaloInfo(recvIndices, sizeof(double), 2);

            tausch->setSendCommunicationStrategy(0, Tausch::Communication::CUDAAwareMPI);

            tausch->packSendBufferCUDA(0, 0, cuda_in1);
            tausch->packSendBufferCUDA(0, 1, cuda_in2);

            tausch->send(0, 0, (mpiRank+1)%mpiSize);
            tausch->recv(0, 0, (mpiRank+mpiSize-1)%mpiSize);

            tausch->unpackRecvBuffer(0, 0, out2);
            tausch->unpackRecvBuffer(0, 1, out1);

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

            delete[] in1;
            delete[] in2;
            delete[] out1;
            delete[] out2;
            cudaFree(&cuda_in1);
            cudaFree(&cuda_in2);
            delete tausch;


        }

    }

}

