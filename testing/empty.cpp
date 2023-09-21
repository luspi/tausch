#include <catch2/catch.hpp>
#if defined(TEST_SEND_TAUSCH_CUDA) || defined(TEST_RECV_TAUSCH_CUDA)
    #define TAUSCH_CUDA
#endif
#if defined(TEST_SEND_TAUSCH_HIP) || defined(TEST_RECV_TAUSCH_HIP)
    #define TAUSCH_HIP
#endif
#if defined(TEST_SEND_TAUSCH_OPENCL) || defined(TEST_RECV_TAUSCH_OPENCL)
    #define TAUSCH_OPENCL
#endif
#include "../tausch.h"
#if defined(TEST_SEND_TAUSCH_OPENCL) || defined(TEST_RECV_TAUSCH_OPENCL)
    #include "ocl.h"
#endif

TEST_CASE("1 buffer, empty indices, with pack/unpack, same MPI rank") {

    std::cout << " * Test: " << "1 buffer, empty indices, with pack/unpack, same MPI rank" << std::endl;

    const std::vector<int> sizes = {3, 10, 100, 377};
    const std::vector<int> halowidths = {1, 2, 3};

#if defined(TEST_SEND_TAUSCH_OPENCL) || defined(TEST_RECV_TAUSCH_OPENCL)
    setupOpenCL();
#endif

    for(auto size : sizes) {

        for(auto halowidth : halowidths) {

            Tausch tausch(MPI_COMM_WORLD, false);
#if defined(TEST_SEND_TAUSCH_OPENCL) || defined(TEST_RECV_TAUSCH_OPENCL)
            tausch.setOpenCL(tauschcl_device, tauschcl_context, tauschcl_queue);
#endif

            std::vector<double> in((size+2*halowidth)*(size+2*halowidth));
            std::vector<double> out((size+2*halowidth)*(size+2*halowidth));
            for(int i = 0; i < size; ++i) {
                for(int j = 0; j < size; ++j) {
                    in[(i+halowidth)*(size+2*halowidth) + j+halowidth] = i*size + j + 1;
                    out[(i+halowidth)*(size+2*halowidth) + j+halowidth] = i*size + j + 1;
                }
            }

#ifdef TEST_SEND_TAUSCH_CUDA
            double *cuda_in;
            cudaMalloc(&cuda_in, (size+2*halowidth)*(size+2*halowidth)*sizeof(double));
            cudaMemcpy(cuda_in, &in[0], (size+2*halowidth)*(size+2*halowidth)*sizeof(double), cudaMemcpyHostToDevice);
#elifdef TEST_SEND_TAUSCH_HIP
            double *hip_in;
            hipMalloc(&hip_in, (size+2*halowidth)*(size+2*halowidth)*sizeof(double));
            hipMemcpy(hip_in, &in[0], (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyHostToDevice);
#elifdef TEST_SEND_TAUSCH_OPENCL
            cl::Buffer cl_in(tausch.getOclQueue(), &in[0], &in[(size+2*halowidth)*(size+2*halowidth)], false);
#endif

#ifdef TEST_RECV_TAUSCH_CUDA
            double *cuda_out;
            cudaMalloc(&cuda_out, (size+2*halowidth)*(size+2*halowidth)*sizeof(double));
            cudaMemcpy(cuda_out, &out[0], (size+2*halowidth)*(size+2*halowidth)*sizeof(double), cudaMemcpyHostToDevice);
#elifdef TEST_RECV_TAUSCH_HIP
            double *hip_out;
            hipMalloc(&hip_out, (size+2*halowidth)*(size+2*halowidth)*sizeof(double));
            hipMemcpy(hip_out, &out[0], (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyHostToDevice);
#elifdef TEST_RECV_TAUSCH_OPENCL
            cl::Buffer cl_out(tausch.getOclQueue(), &out[0], &out[(size+2*halowidth)*(size+2*halowidth)], false);
#endif

            std::vector<int> sendIndices;
            std::vector<int> recvIndices;

            int mpiRank;
            MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

            tausch.addSendHaloInfo(sendIndices, sizeof(double));
            tausch.addRecvHaloInfo(recvIndices, sizeof(double));

#ifdef TEST_SEND_TAUSCH_CUDA
            tausch.packSendBufferCUDA(0, 0, cuda_in);
#elifdef TEST_SEND_TAUSCH_HIP
            tausch.packSendBufferHIP(0, 0, hip_in);
#elifdef TEST_SEND_TAUSCH_OPENCL
            tausch.packSendBufferOCL(0, 0, cl_in);
#else
            tausch.packSendBuffer(0, 0, &in[0]);
#endif

            Status status = tausch.send(0, 0, mpiRank, false);
            tausch.recv(0, 0, mpiRank, true);

            status.wait();

#ifdef TEST_RECV_TAUSCH_CUDA
            tausch.unpackRecvBufferCUDA(0, 0, cuda_out);
            cudaMemcpy(&out[0], cuda_out, (size+2*halowidth)*(size+2*halowidth)*sizeof(double), cudaMemcpyDeviceToHost);
#elifdef TEST_RECV_TAUSCH_HIP
            tausch.unpackRecvBufferHIP(0, 0, hip_out);
            hipMemcpy(&out[0], hip_out, (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyDeviceToHost);
#elifdef TEST_RECV_TAUSCH_OPENCL
            tausch.unpackRecvBufferOCL(0, 0, cl_out);
            cl::copy(tausch.getOclQueue(), cl_out, &out[0], &out[(size+2*halowidth)*(size+2*halowidth)]);
#else
            tausch.unpackRecvBuffer(0, 0, &out[0]);
#endif

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

#ifdef TEST_SEND_TAUSCH_CUDA
            cudaFree(cuda_in);
#elifdef TEST_SEND_TAUSCH_HIP
            hipFree(hip_in);
#endif

#ifdef TEST_RECV_TAUSCH_CUDA
            cudaFree(cuda_out);
#elifdef TEST_RECV_TAUSCH_HIP
            hipFree(hip_out);
#endif

        }

    }

}

TEST_CASE("1 buffer, empty indices, with pack/unpack, multiple MPI ranks") {

    std::cout << " * Test: " << "1 buffer, empty indices, with pack/unpack, multiple MPI ranks" << std::endl;

    const std::vector<int> sizes = {3, 10, 100, 377};
    const std::vector<int> halowidths = {1, 2, 3};

#if defined(TEST_SEND_TAUSCH_OPENCL) || defined(TEST_RECV_TAUSCH_OPENCL)
    setupOpenCL();
#endif

    for(auto size : sizes) {

        for(auto halowidth : halowidths) {

            Tausch tausch(MPI_COMM_WORLD, false);
#if defined(TEST_SEND_TAUSCH_OPENCL) || defined(TEST_RECV_TAUSCH_OPENCL)
            tausch.setOpenCL(tauschcl_device, tauschcl_context, tauschcl_queue);
#endif

            std::vector<double> in((size+2*halowidth)*(size+2*halowidth));
            std::vector<double> out((size+2*halowidth)*(size+2*halowidth));
            for(int i = 0; i < size; ++i) {
                for(int j = 0; j < size; ++j) {
                    in[(i+halowidth)*(size+2*halowidth) + j+halowidth] = i*size + j + 1;
                    out[(i+halowidth)*(size+2*halowidth) + j+halowidth] = i*size + j + 1;
                }
            }

#ifdef TEST_SEND_TAUSCH_CUDA
            double *cuda_in;
            cudaMalloc(&cuda_in, (size+2*halowidth)*(size+2*halowidth)*sizeof(double));
            cudaMemcpy(cuda_in, &in[0], (size+2*halowidth)*(size+2*halowidth)*sizeof(double), cudaMemcpyHostToDevice);
#elifdef TEST_SEND_TAUSCH_HIP
            double *hip_in;
            hipMalloc(&hip_in, (size+2*halowidth)*(size+2*halowidth)*sizeof(double));
            hipMemcpy(hip_in, &in[0], (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyHostToDevice);
#elifdef TEST_SEND_TAUSCH_OPENCL
            cl::Buffer cl_in(tausch.getOclQueue(), &in[0], &in[(size+2*halowidth)*(size+2*halowidth)], false);
#endif

#ifdef TEST_RECV_TAUSCH_CUDA
            double *cuda_out;
            cudaMalloc(&cuda_out, (size+2*halowidth)*(size+2*halowidth)*sizeof(double));
            cudaMemcpy(cuda_out, &out[0], (size+2*halowidth)*(size+2*halowidth)*sizeof(double), cudaMemcpyHostToDevice);
#elifdef TEST_RECV_TAUSCH_HIP
            double *hip_out;
            hipMalloc(&hip_out, (size+2*halowidth)*(size+2*halowidth)*sizeof(double));
            hipMemcpy(hip_out, &out[0], (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyHostToDevice);
#elifdef TEST_RECV_TAUSCH_OPENCL
            cl::Buffer cl_out(tausch.getOclQueue(), &out[0], &out[(size+2*halowidth)*(size+2*halowidth)], false);
#endif

            std::vector<int> sendIndices;
            std::vector<int> recvIndices;

            int mpiRank, mpiSize;
            MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
            MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

            tausch.addSendHaloInfo(sendIndices, sizeof(double));
            tausch.addRecvHaloInfo(recvIndices, sizeof(double));

            #ifdef TEST_SEND_TAUSCH_CUDA
            tausch.packSendBufferCUDA(0, 0, cuda_in);
#elifdef TEST_SEND_TAUSCH_HIP
            tausch.packSendBufferHIP(0, 0, hip_in);
#elifdef TEST_SEND_TAUSCH_OPENCL
            tausch.packSendBufferOCL(0, 0, cl_in);
#else
            tausch.packSendBuffer(0, 0, &in[0]);
#endif

            Status status = tausch.send(0, 0, (mpiRank+1)%mpiSize, false);
            tausch.recv(0, 0, (mpiRank+mpiSize-1)%mpiSize, true);

            status.wait();

#ifdef TEST_RECV_TAUSCH_CUDA
            tausch.unpackRecvBufferCUDA(0, 0, cuda_out);
            cudaMemcpy(&out[0], cuda_out, (size+2*halowidth)*(size+2*halowidth)*sizeof(double), cudaMemcpyDeviceToHost);
#elifdef TEST_RECV_TAUSCH_HIP
            tausch.unpackRecvBufferHIP(0, 0, hip_out);
            hipMemcpy(&out[0], hip_out, (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyDeviceToHost);
#elifdef TEST_RECV_TAUSCH_OPENCL
            tausch.unpackRecvBufferOCL(0, 0, cl_out);
            cl::copy(tausch.getOclQueue(), cl_out, &out[0], &out[(size+2*halowidth)*(size+2*halowidth)]);
#else
            tausch.unpackRecvBuffer(0, 0, &out[0]);
#endif

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

#ifdef TEST_SEND_TAUSCH_CUDA
            cudaFree(cuda_in);
#elifdef TEST_SEND_TAUSCH_HIP
            hipFree(hip_in);
#endif

#ifdef TEST_RECV_TAUSCH_CUDA
            cudaFree(cuda_out);
#elifdef TEST_RECV_TAUSCH_HIP
            hipFree(hip_out);
#endif

        }

    }

}
