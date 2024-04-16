#include <catch2/catch_all.hpp>
#if defined(TEST_SEND_TAUSCH_CUDA) || defined(TEST_RECV_TAUSCH_CUDA)
    #define TAUSCH_CUDA
    #include <thrust/device_vector.h>
    #include <thrust/copy.h>
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

TEST_CASE("1 buffer, random indices, with pack/unpack, same MPI rank") {

    std::cout << " * Test: " << "1 buffer, random indices, with pack/unpack, same MPI rank" << std::endl;

#if defined(TEST_SEND_TAUSCH_OPENCL) || defined(TEST_RECV_TAUSCH_OPENCL)
    setupOpenCL();
#endif

    const std::vector<int> sizes = {10, 100, 377};
    const std::vector<int> halowidths = {1, 2, 3};

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
            thrust::device_vector<double> cuda_in(in);
#elif defined(TEST_SEND_TAUSCH_HIP)
            double *hip_in;
            hipMalloc(&hip_in, (size+2*halowidth)*(size+2*halowidth)*sizeof(double));
            hipMemcpy(hip_in, &in[0], (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyHostToDevice);
#elif defined(TEST_SEND_TAUSCH_OPENCL)
            cl::Buffer cl_in(tausch.getOclQueue(), &in[0], &in[(size+2*halowidth)*(size+2*halowidth)], false);
#endif

#ifdef TEST_RECV_TAUSCH_CUDA
            thrust::device_vector<double> cuda_out(out);
#elif defined(TEST_RECV_TAUSCH_HIP)
            double *hip_out;
            hipMalloc(&hip_out, (size+2*halowidth)*(size+2*halowidth)*sizeof(double));
            hipMemcpy(hip_out, &out[0], (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyHostToDevice);
#elif defined(TEST_RECV_TAUSCH_OPENCL)
            cl::Buffer cl_out(tausch.getOclQueue(), &out[0], &out[(size+2*halowidth)*(size+2*halowidth)], false);
#endif

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

            int mpiRank;
            MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

            tausch.addSendHaloInfo(sendIndices, sizeof(double));
            tausch.addRecvHaloInfo(recvIndices, sizeof(double));

#ifdef TEST_SEND_TAUSCH_CUDA
            tausch.packSendBufferCUDA(0, 0, thrust::raw_pointer_cast(cuda_in.data()));
#elif defined(TEST_SEND_TAUSCH_HIP)
            tausch.packSendBufferHIP(0, 0, hip_in);
#elif defined(TEST_SEND_TAUSCH_OPENCL)
            tausch.packSendBufferOCL(0, 0, cl_in);
#else
            tausch.packSendBuffer(0, 0, &in[0]);
#endif

            Status status = tausch.send(0, 0, mpiRank, false);
            tausch.recv(0, 0, mpiRank, true);

            status.wait();

#ifdef TEST_RECV_TAUSCH_CUDA
            tausch.unpackRecvBufferCUDA(0, 0, thrust::raw_pointer_cast(cuda_out.data()));
            thrust::copy(cuda_out.begin(), cuda_out.end(), out.begin());
#elif defined(TEST_RECV_TAUSCH_HIP)
            tausch.unpackRecvBufferHIP(0, 0, hip_out);
            hipMemcpy(&out[0], hip_out, (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyDeviceToHost);
#elif defined(TEST_RECV_TAUSCH_OPENCL)
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

#ifdef TEST_SEND_TAUSCH_HIP
            hipFree(hip_in);
#endif

#ifdef TEST_RECV_TAUSCH_HIP
            hipFree(hip_out);
#endif

        }

    }

}

TEST_CASE("1 buffer, random indices, with pack/unpack, multiple MPI ranks") {

    std::cout << " * Test: " << "1 buffer, random indices, with pack/unpack, multiple MPI ranks" << std::endl;

#if defined(TEST_SEND_TAUSCH_OPENCL) || defined(TEST_RECV_TAUSCH_OPENCL)
    setupOpenCL();
#endif

    const std::vector<int> sizes = {10, 100, 377};
    const std::vector<int> halowidths = {1, 2, 3};

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
            thrust::device_vector<double> cuda_in(in);
#elif defined(TEST_SEND_TAUSCH_HIP)
            double *hip_in;
            hipMalloc(&hip_in, (size+2*halowidth)*(size+2*halowidth)*sizeof(double));
            hipMemcpy(hip_in, &in[0], (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyHostToDevice);
#elif defined(TEST_SEND_TAUSCH_OPENCL)
            cl::Buffer cl_in(tausch.getOclQueue(), &in[0], &in[(size+2*halowidth)*(size+2*halowidth)], false);
#endif

#ifdef TEST_RECV_TAUSCH_CUDA
            thrust::device_vector<double> cuda_out(out);
#elif defined(TEST_RECV_TAUSCH_HIP)
            double *hip_out;
            hipMalloc(&hip_out, (size+2*halowidth)*(size+2*halowidth)*sizeof(double));
            hipMemcpy(hip_out, &out[0], (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyHostToDevice);
#elif defined(TEST_RECV_TAUSCH_OPENCL)
            cl::Buffer cl_out(tausch.getOclQueue(), &out[0], &out[(size+2*halowidth)*(size+2*halowidth)], false);
#endif

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

            int mpiRank, mpiSize;
            MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
            MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

            tausch.addSendHaloInfo(sendIndices, sizeof(double));
            tausch.addRecvHaloInfo(recvIndices, sizeof(double));

#ifdef TEST_SEND_TAUSCH_CUDA
            tausch.packSendBufferCUDA(0, 0, thrust::raw_pointer_cast(cuda_in.data()));
#elif defined(TEST_SEND_TAUSCH_HIP)
            tausch.packSendBufferHIP(0, 0, hip_in);
#elif defined(TEST_SEND_TAUSCH_OPENCL)
            tausch.packSendBufferOCL(0, 0, cl_in);
#else
            tausch.packSendBuffer(0, 0, &in[0]);
#endif

            Status status = tausch.send(0, 0, (mpiRank+1)%mpiSize, false);
            tausch.recv(0, 0, (mpiRank+mpiSize-1)%mpiSize, true);

            status.wait();

#ifdef TEST_RECV_TAUSCH_CUDA
            tausch.unpackRecvBufferCUDA(0, 0, thrust::raw_pointer_cast(cuda_out.data()));
            thrust::copy(cuda_out.begin(), cuda_out.end(), out.begin());
#elif defined(TEST_RECV_TAUSCH_HIP)
            tausch.unpackRecvBufferHIP(0, 0, hip_out);
            hipMemcpy(&out[0], hip_out, (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyDeviceToHost);
#elif defined(TEST_RECV_TAUSCH_OPENCL)
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

#ifdef TEST_SEND_TAUSCH_HIP
            hipFree(hip_in);
#endif

#ifdef TEST_RECV_TAUSCH_HIP
            hipFree(hip_out);
#endif

        }

    }

}
