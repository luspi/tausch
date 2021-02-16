#include <catch2/catch.hpp>
#define TAUSCH_OPENCL
#include "../../tausch.h"
#include <iomanip>
#include "ocl.h"

TEST_CASE("1 buffer, derived MPI datatype, same MPI rank") {

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

            Tausch<double> *tausch = new Tausch<double>(tauschcl_device, tauschcl_context, tauschcl_queue, MPI_DOUBLE, MPI_COMM_WORLD, false);

            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);

            tausch->addLocalHaloInfo(sendIndices, 1, -1, TauschOptimizationHint::ReceiverUsesMpiDerivedDatatype);
            tausch->addRemoteHaloInfo(recvIndices, 1, -1, TauschOptimizationHint::UseMpiDerivedDatatype);

            tausch->packSendBuffer(0, 0, cl_in);

            tausch->send(0, 0, nullptr, rank, false);
            tausch->recv(0, 0, nullptr, rank, 0, out, true);

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

TEST_CASE("1 buffer, derived MPI datatype, multiple MPI ranks") {

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

            Tausch<double> *tausch = new Tausch<double>(tauschcl_device, tauschcl_context, tauschcl_queue, MPI_DOUBLE, MPI_COMM_WORLD, false);

            int mpiRank, mpiSize;
            MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
            MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

            tausch->addLocalHaloInfo(sendIndices, 1, -1, TauschOptimizationHint::ReceiverUsesMpiDerivedDatatype);
            tausch->addRemoteHaloInfo(recvIndices, 1, -1, TauschOptimizationHint::UseMpiDerivedDatatype);

            tausch->packSendBuffer(0, 0, cl_in);

            tausch->send(0, 0, nullptr, (mpiRank+1)%mpiSize, false);
            tausch->recv(0, 0, nullptr, (mpiRank+mpiSize-1)%mpiSize, 0, out, true);

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


/**************************************************************************/
/**************************************************************************/

// THE TWO TESTS BELOW DO CURRENTLY NOT WORK:
// WITH DERIVED DATATYPES THE BUFFERS ARE SENT AS SEPERATE MESSAGE YET THE RECV EXPECTS THEM AS ONE LARGE MESSAGE



//TEST_CASE("2 buffers, derived MPI datatype, same MPI rank") {

//    setupOpenCL();

//    const std::vector<int> sizes = {3};//, 10, 100, 377};
//    const std::vector<int> halowidths = {1};//, 2, 3};

//    for(auto size : sizes) {

//        for(auto halowidth : halowidths) {

//            double *in1 = new double[(size+2*halowidth)*(size+2*halowidth)]{};
//            double *in2 = new double[(size+2*halowidth)*(size+2*halowidth)]{};
//            double *out1 = new double[(size+2*halowidth)*(size+2*halowidth)]{};
//            double *out2 = new double[(size+2*halowidth)*(size+2*halowidth)]{};
//            for(int i = 0; i < size; ++i) {
//                for(int j = 0; j < size; ++j) {
//                    in1[(i+halowidth)*(size+2*halowidth) + j+halowidth] = i*size + j + 1;
//                    in2[(i+halowidth)*(size+2*halowidth) + j+halowidth] = size*size + i*size + j + 1;
//                    out1[(i+halowidth)*(size+2*halowidth) + j+halowidth] = i*size + j + 1;
//                    out2[(i+halowidth)*(size+2*halowidth) + j+halowidth] = size*size + i*size + j + 1;
//                }
//            }
//            cl::Buffer cl_out1(tauschcl_queue, out1, &out1[(size+2*halowidth)*(size+2*halowidth)], false);
//            cl::Buffer cl_out2(tauschcl_queue, out2, &out2[(size+2*halowidth)*(size+2*halowidth)], false);

//            std::vector<int> sendIndices;
//            std::vector<int> recvIndices;
//            // bottom edge
//            for(int i = 0; i < size; ++i)
//                for(int j = 0; j < halowidth; ++j) {
//                    sendIndices.push_back((j+halowidth)*(size+2*halowidth) + i+halowidth);
//                    recvIndices.push_back(j*(size+2*halowidth) + i+halowidth);
//                }
//            // left edge
//            for(int i = 0; i < halowidth; ++i)
//                for(int j = 0; j < size; ++j) {
//                    sendIndices.push_back((j+halowidth)*(size+2*halowidth) + i+halowidth);
//                    recvIndices.push_back((j+halowidth)*(size+2*halowidth) + i);
//                }
//            // right edge
//            for(int i = 0; i < halowidth; ++i)
//                for(int j = 0; j < size; ++j) {
//                    sendIndices.push_back((j+halowidth)*(size+2*halowidth) + i+size);
//                    recvIndices.push_back((j+halowidth)*(size+2*halowidth) + i+(size+halowidth));
//                }
//            // top edge
//            for(int i = 0; i < size; ++i)
//                for(int j = 0; j < halowidth; ++j) {
//                    sendIndices.push_back((j+size)*(size+2*halowidth) + i+halowidth);
//                    recvIndices.push_back((j+(size+halowidth))*(size+2*halowidth) + i+halowidth);
//                }

//            Tausch<double> *tausch = new Tausch<double>(tauschcl_device, tauschcl_context, tauschcl_queue, "double", MPI_DOUBLE, MPI_COMM_WORLD, false);

//            int mpiRank;
//            MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

//            tausch->addLocalHaloInfo(sendIndices, 2, -1, TauschOptimizationHint::UseMpiDerivedDatatype);
//            tausch->addRemoteHaloInfo(recvIndices, 2, -1, TauschOptimizationHint::SenderUsesMpiDerivedDatatype);

//            std::cout << "a" << std::endl;

//            tausch->send(0, 0, nullptr, mpiRank, in1, false);
//            tausch->send(0, 1, nullptr, mpiRank, in2, false);

//            std::cout << "b" << std::endl;

//            tausch->recv(0, 0, nullptr, mpiRank);
//            tausch->recv(0, 1, nullptr, mpiRank);

//            std::cout << "c" << std::endl;

//            tausch->unpackRecvBuffer(0, 0, cl_out1);
//            tausch->unpackRecvBuffer(0, 1, cl_out2);

//            std::cout << "d" << std::endl;

//            cl::copy(tauschcl_queue, cl_out1, out1, &out1[(size+2*halowidth)*(size+2*halowidth)]);
//            cl::copy(tauschcl_queue, cl_out2, out2, &out2[(size+2*halowidth)*(size+2*halowidth)]);

//            double *expected1 = new double[(size+2*halowidth)*(size+2*halowidth)]{};
//            double *expected2 = new double[(size+2*halowidth)*(size+2*halowidth)]{};
//            for(int i = 0; i < size; ++i) {
//                for(int j = 0; j < size; ++j) {
//                    expected1[(i+halowidth)*(size+2*halowidth) + j+halowidth] = i*size + j + 1;
//                    expected2[(i+halowidth)*(size+2*halowidth) + j+halowidth] = size*size + i*size + j + 1;
//                }
//            }

//            for(int i = 0; i < size; ++i)
//                for(int j = 0; j < halowidth; ++j) {
//                    expected1[j*(size+2*halowidth) + i+halowidth] = size*size + j*size+i+1;  // bottom
//                    expected1[(j+size+halowidth)*(size+2*halowidth) + i+halowidth] = size*size + (j+(size-halowidth))*size + i+1; // top
//                    expected1[(i+halowidth)*(size+2*halowidth) + j] = size*size + i*size+j+1;    // left
//                    expected1[(i+halowidth)*(size+2*halowidth) + j+(size+halowidth)] = size*size + i*size + (size-halowidth)+j+1;    // left

//                    expected2[j*(size+2*halowidth) + i+halowidth] = j*size+i+1;  // bottom
//                    expected2[(j+size+halowidth)*(size+2*halowidth) + i+halowidth] = (j+(size-halowidth))*size + i+1; // top
//                    expected2[(i+halowidth)*(size+2*halowidth) + j] = i*size+j+1;    // left
//                    expected2[(i+halowidth)*(size+2*halowidth) + j+(size+halowidth)] = i*size + (size-halowidth)+j+1;    // left
//                }

//            // check result
//            for(int i = 0; i < (size+2*halowidth); ++i)
//                for(int j = 0; j < (size+2*halowidth); ++j) {
//                    if(i%5) std::cout << "checking " << i << " / " << j << std::endl;
//                    REQUIRE(expected1[i*(size+2*halowidth)+j] == out1[i*(size+2*halowidth)+j]);
//                    REQUIRE(expected2[i*(size+2*halowidth)+j] == out2[i*(size+2*halowidth)+j]);
//                }


//        }

//    }

//}

//TEST_CASE("2 buffers, derived MPI datatype, multiple MPI ranks") {

//    const std::vector<int> sizes = {3, 10, 100, 377};
//    const std::vector<int> halowidths = {1, 2, 3};

//    for(auto size : sizes) {

//        for(auto halowidth : halowidths) {

//            double *in1 = new double[(size+2*halowidth)*(size+2*halowidth)]{};
//            double *in2 = new double[(size+2*halowidth)*(size+2*halowidth)]{};
//            double *out1 = new double[(size+2*halowidth)*(size+2*halowidth)]{};
//            double *out2 = new double[(size+2*halowidth)*(size+2*halowidth)]{};
//            for(int i = 0; i < size; ++i) {
//                for(int j = 0; j < size; ++j) {
//                    in1[(i+halowidth)*(size+2*halowidth) + j+halowidth] = i*size + j + 1;
//                    in2[(i+halowidth)*(size+2*halowidth) + j+halowidth] = size*size + i*size + j + 1;
//                    out1[(i+halowidth)*(size+2*halowidth) + j+halowidth] = i*size + j + 1;
//                    out2[(i+halowidth)*(size+2*halowidth) + j+halowidth] = size*size + i*size + j + 1;
//                }
//            }

//            std::vector<int> sendIndices;
//            std::vector<int> recvIndices;
//            // bottom edge
//            for(int i = 0; i < size; ++i)
//                for(int j = 0; j < halowidth; ++j) {
//                    sendIndices.push_back((j+halowidth)*(size+2*halowidth) + i+halowidth);
//                    recvIndices.push_back(j*(size+2*halowidth) + i+halowidth);
//                }
//            // left edge
//            for(int i = 0; i < halowidth; ++i)
//                for(int j = 0; j < size; ++j) {
//                    sendIndices.push_back((j+halowidth)*(size+2*halowidth) + i+halowidth);
//                    recvIndices.push_back((j+halowidth)*(size+2*halowidth) + i);
//                }
//            // right edge
//            for(int i = 0; i < halowidth; ++i)
//                for(int j = 0; j < size; ++j) {
//                    sendIndices.push_back((j+halowidth)*(size+2*halowidth) + i+size);
//                    recvIndices.push_back((j+halowidth)*(size+2*halowidth) + i+(size+halowidth));
//                }
//            // top edge
//            for(int i = 0; i < size; ++i)
//                for(int j = 0; j < halowidth; ++j) {
//                    sendIndices.push_back((j+size)*(size+2*halowidth) + i+halowidth);
//                    recvIndices.push_back((j+(size+halowidth))*(size+2*halowidth) + i+halowidth);
//                }

//            Tausch<double> *tausch = new Tausch<double>(MPI_DOUBLE, MPI_COMM_WORLD, false);

//            int mpiRank, mpiSize;
//            MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
//            MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

//            tausch->addLocalHaloInfo(sendIndices, 2, -1, TauschOptimizationHint::UseMpiDerivedDatatype);
//            tausch->addRemoteHaloInfo(recvIndices, 2, -1, TauschOptimizationHint::UseMpiDerivedDatatype);

//            tausch->send(0, 0, nullptr, (mpiRank+1)%mpiSize, in1, false);
//            tausch->send(0, 1, nullptr, (mpiRank+1)%mpiSize, in2, false);
//            tausch->recv(0, 1, nullptr, (mpiRank+mpiSize-1)%mpiSize, out1, true);
//            tausch->recv(0, 0, nullptr, (mpiRank+mpiSize-1)%mpiSize, out2, true);

//            double *expected1 = new double[(size+2*halowidth)*(size+2*halowidth)]{};
//            double *expected2 = new double[(size+2*halowidth)*(size+2*halowidth)]{};
//            for(int i = 0; i < size; ++i) {
//                for(int j = 0; j < size; ++j) {
//                    expected1[(i+halowidth)*(size+2*halowidth) + j+halowidth] = i*size + j + 1;
//                    expected2[(i+halowidth)*(size+2*halowidth) + j+halowidth] = size*size + i*size + j + 1;
//                }
//            }

//            for(int i = 0; i < size; ++i)
//                for(int j = 0; j < halowidth; ++j) {
//                    expected1[j*(size+2*halowidth) + i+halowidth] = size*size + j*size+i+1;  // bottom
//                    expected1[(j+size+halowidth)*(size+2*halowidth) + i+halowidth] = size*size + (j+(size-halowidth))*size + i+1; // top
//                    expected1[(i+halowidth)*(size+2*halowidth) + j] = size*size + i*size+j+1;    // left
//                    expected1[(i+halowidth)*(size+2*halowidth) + j+(size+halowidth)] = size*size + i*size + (size-halowidth)+j+1;    // left

//                    expected2[j*(size+2*halowidth) + i+halowidth] = j*size+i+1;  // bottom
//                    expected2[(j+size+halowidth)*(size+2*halowidth) + i+halowidth] = (j+(size-halowidth))*size + i+1; // top
//                    expected2[(i+halowidth)*(size+2*halowidth) + j] = i*size+j+1;    // left
//                    expected2[(i+halowidth)*(size+2*halowidth) + j+(size+halowidth)] = i*size + (size-halowidth)+j+1;    // left
//                }

//            // check result
//            for(int i = 0; i < (size+2*halowidth); ++i)
//                for(int j = 0; j < (size+2*halowidth); ++j) {
//                    REQUIRE(expected1[i*(size+2*halowidth)+j] == out1[i*(size+2*halowidth)+j]);
//                    REQUIRE(expected2[i*(size+2*halowidth)+j] == out2[i*(size+2*halowidth)+j]);
//                }


//        }

//    }

//}
