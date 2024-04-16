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

TEST_CASE("1 buffer, with pack/unpack, same MPI rank") {

    std::cout << " * Test: " << "1 buffer, with pack/unpack, same MPI rank" << std::endl;

#if defined(TEST_SEND_TAUSCH_OPENCL) || defined(TEST_RECV_TAUSCH_OPENCL)
    setupOpenCL();
#endif

    const std::vector<int> sizes = {3, 10, 100, 377};
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
#elif defined(TEST_SEND_TAUSCH_OPENCLUSCH_HIP)
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

            int mpiRank, mpiSize;
            MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
            MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

            tausch.addSendHaloInfo(sendIndices, sizeof(double));
            tausch.addRecvHaloInfo(recvIndices, sizeof(double));

#ifdef TEST_SEND_TAUSCH_CUDA
            tausch.packSendBufferCUDA(0, 0, thrust::raw_pointer_cast(cuda_in.data()));
#elif defined(TEST_SEND_TAUSCH_HIP)
            tausch.packSendBufferHIP(0, 0, hip_in);
#elif defined(TEST_SEND_TAUSCH_OPENCLUSCH_HIP)
            tausch.packSendBufferOCL(0, 0, cl_in);
#else
            tausch.packSendBuffer(0, 0, &in[0]);
#endif

            Status status = tausch.send(0, 0, mpiRank);
            tausch.recv(0, 0, mpiRank);

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

            std::vector<double> expected((size+2*halowidth)*(size+2*halowidth));
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

#ifdef TEST_SEND_TAUSCH_HIP
            hipFree(hip_in);
#endif

#ifdef TEST_RECV_TAUSCH_HIP
            hipFree(hip_out);
#endif

        }

    }

}

#if !defined(TEST_SEND_TAUSCH_CPU) && !defined(TEST_RECV_TAUSCH_CPU)
TEST_CASE("1 buffer, with pack/unpack, same MPI rank, GPUMultiCopy") {

    std::cout << " * Test: " << "1 buffer, with pack/unpack, same MPI rank, GPUMultiCopy" << std::endl;

#if defined(TEST_SEND_TAUSCH_OPENCL) || defined(TEST_RECV_TAUSCH_OPENCL)
    setupOpenCL();
#endif

    const std::vector<int> sizes = {3, 10, 100, 377};
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
#elif defined(TEST_SEND_TAUSCH_OPENCLUSCH_HIP)
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

            int mpiRank;
            MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

            tausch.addSendHaloInfo(sendIndices, sizeof(double));
            tausch.addRecvHaloInfo(recvIndices, sizeof(double));

            tausch.setSendCommunicationStrategy(0, Tausch::Communication::GPUMultiCopy);
            tausch.setRecvCommunicationStrategy(0, Tausch::Communication::GPUMultiCopy);

#ifdef TEST_SEND_TAUSCH_CUDA
            tausch.packSendBufferCUDA(0, 0, thrust::raw_pointer_cast(cuda_in.data()));
#elif defined(TEST_SEND_TAUSCH_HIP)
            tausch.packSendBufferHIP(0, 0, hip_in);
#elif defined(TEST_SEND_TAUSCH_OPENCLUSCH_HIP)
            tausch.packSendBufferOCL(0, 0, cl_in);
#else
            tausch.packSendBuffer(0, 0, &in[0]);
#endif

            Status status = tausch.send(0, 0, mpiRank);
            tausch.recv(0, 0, mpiRank);

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

#ifdef TEST_SEND_TAUSCH_HIP
            hipFree(hip_in);
#endif

#ifdef TEST_RECV_TAUSCH_HIP
            hipFree(hip_out);
#endif

        }

    }

}
#endif

TEST_CASE("1 buffer, with pack/unpack, multiple MPI ranks") {

    std::cout << " * Test: " << "1 buffer, with pack/unpack, multiple MPI rank" << std::endl;

#if defined(TEST_SEND_TAUSCH_OPENCL) || defined(TEST_RECV_TAUSCH_OPENCL)
    setupOpenCL();
#endif

    const std::vector<int> sizes = {3, 10, 100, 377};
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
#elif defined(TEST_SEND_TAUSCH_OPENCLUSCH_HIP)
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

            int mpiRank, mpiSize;
            MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
            MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

            tausch.addSendHaloInfo(sendIndices, sizeof(double));
            tausch.addRecvHaloInfo(recvIndices, sizeof(double));

#ifdef TEST_SEND_TAUSCH_CUDA
            tausch.packSendBufferCUDA(0, 0, thrust::raw_pointer_cast(cuda_in.data()));
#elif defined(TEST_SEND_TAUSCH_HIP)
            tausch.packSendBufferHIP(0, 0, hip_in);
#elif defined(TEST_SEND_TAUSCH_OPENCLUSCH_HIP)
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

#ifdef TEST_SEND_TAUSCH_HIP
            hipFree(hip_in);
#endif

#ifdef TEST_RECV_TAUSCH_HIP
            hipFree(hip_out);
#endif

        }

    }

}

TEST_CASE("2 buffers, with pack/unpack, same MPI rank") {

    std::cout << " * Test: " << "2 buffers, with pack/unpack, same MPI rank" << std::endl;

#if defined(TEST_SEND_TAUSCH_OPENCL) || defined(TEST_RECV_TAUSCH_OPENCL)
    setupOpenCL();
#endif

    const std::vector<int> sizes = {3, 10, 100, 377};
    const std::vector<int> halowidths = {1, 2, 3};

    for(auto size : sizes) {

        for(auto halowidth : halowidths) {

            Tausch tausch(MPI_COMM_WORLD, false);
#if defined(TEST_SEND_TAUSCH_OPENCL) || defined(TEST_RECV_TAUSCH_OPENCL)
            tausch.setOpenCL(tauschcl_device, tauschcl_context, tauschcl_queue);
#endif

            std::vector<double> in1((size+2*halowidth)*(size+2*halowidth));
            std::vector<double> in2((size+2*halowidth)*(size+2*halowidth));
            std::vector<double> out1((size+2*halowidth)*(size+2*halowidth));
            std::vector<double> out2((size+2*halowidth)*(size+2*halowidth));
            for(int i = 0; i < size; ++i) {
                for(int j = 0; j < size; ++j) {
                    in1[(i+halowidth)*(size+2*halowidth) + j+halowidth] = i*size + j + 1;
                    in2[(i+halowidth)*(size+2*halowidth) + j+halowidth] = size*size + i*size + j + 1;
                }
            }

#ifdef TEST_SEND_TAUSCH_CUDA
            thrust::device_vector<double> cuda_in1(in1);
            thrust::device_vector<double> cuda_in2(in2);
#elif defined(TEST_SEND_TAUSCH_HIP)
            double *hip_in1, *hip_in2;
            hipMalloc(&hip_in1, (size+2*halowidth)*(size+2*halowidth)*sizeof(double));
            hipMalloc(&hip_in2, (size+2*halowidth)*(size+2*halowidth)*sizeof(double));
            hipMemcpy(hip_in1, &in1[0], (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyHostToDevice);
            hipMemcpy(hip_in2, &in2[0], (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyHostToDevice);
#elif defined(TEST_SEND_TAUSCH_OPENCLUSCH_HIP)
            cl::Buffer cl_in1(tausch.getOclQueue(), &in1[0], &in1[(size+2*halowidth)*(size+2*halowidth)], false);
            cl::Buffer cl_in2(tausch.getOclQueue(), &in2[0], &in2[(size+2*halowidth)*(size+2*halowidth)], false);
#endif

#ifdef TEST_RECV_TAUSCH_CUDA
            thrust::device_vector<double> cuda_out1(out1);
            thrust::device_vector<double> cuda_out2(out2);
#elif defined(TEST_RECV_TAUSCH_HIP)
            double *hip_out1, *hip_out2;
            hipMalloc(&hip_out1, (size+2*halowidth)*(size+2*halowidth)*sizeof(double));
            hipMalloc(&hip_out2, (size+2*halowidth)*(size+2*halowidth)*sizeof(double));
            hipMemcpy(hip_out1, &out1[0], (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyHostToDevice);
            hipMemcpy(hip_out2, &out2[0], (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyHostToDevice);
#elif defined(TEST_RECV_TAUSCH_OPENCL)
            cl::Buffer cl_out1(tausch.getOclQueue(), &out1[0], &out1[(size+2*halowidth)*(size+2*halowidth)], false);
            cl::Buffer cl_out2(tausch.getOclQueue(), &out2[0], &out2[(size+2*halowidth)*(size+2*halowidth)], false);
#endif

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

            int mpiRank;
            MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

            tausch.addSendHaloInfos(sendIndices, sizeof(double), 2);
            tausch.addRecvHaloInfos(recvIndices, sizeof(double), 2);

#ifdef TEST_SEND_TAUSCH_CUDA
            tausch.packSendBufferCUDA(0, 0, thrust::raw_pointer_cast(cuda_in1.data()));
            tausch.packSendBufferCUDA(0, 1, thrust::raw_pointer_cast(cuda_in2.data()));
#elif defined(TEST_SEND_TAUSCH_HIP)
            tausch.packSendBufferHIP(0, 0, hip_in1);
            tausch.packSendBufferHIP(0, 1, hip_in2);
#elif defined(TEST_SEND_TAUSCH_OPENCLUSCH_HIP)
            tausch.packSendBufferOCL(0, 0, cl_in1);
            tausch.packSendBufferOCL(0, 1, cl_in2);
#else
            tausch.packSendBuffer(0, 0, &in1[0]);
            tausch.packSendBuffer(0, 1, &in2[0]);
#endif

            Status status = tausch.send(0, 0, mpiRank);
            tausch.recv(0, 0, mpiRank);

            status.wait();

#ifdef TEST_RECV_TAUSCH_CUDA
            tausch.unpackRecvBufferCUDA(0, 0, thrust::raw_pointer_cast(cuda_out2.data()));
            tausch.unpackRecvBufferCUDA(0, 1, thrust::raw_pointer_cast(cuda_out1.data()));
            thrust::copy(cuda_out1.begin(), cuda_out1.end(), out1.begin());
            thrust::copy(cuda_out2.begin(), cuda_out2.end(), out2.begin());
#elif defined(TEST_RECV_TAUSCH_HIP)
            tausch.unpackRecvBufferHIP(0, 0, hip_out2);
            tausch.unpackRecvBufferHIP(0, 1, hip_out1);
            hipMemcpy(&out1[0], hip_out1, (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyDeviceToHost);
            hipMemcpy(&out2[0], hip_out2, (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyDeviceToHost);
#elif defined(TEST_RECV_TAUSCH_OPENCL)
            tausch.unpackRecvBufferOCL(0, 0, cl_out2);
            tausch.unpackRecvBufferOCL(0, 1, cl_out1);
            cl::copy(tausch.getOclQueue(), cl_out1, &out1[0], &out1[(size+2*halowidth)*(size+2*halowidth)]);
            cl::copy(tausch.getOclQueue(), cl_out2, &out2[0], &out2[(size+2*halowidth)*(size+2*halowidth)]);
#else
            tausch.unpackRecvBuffer(0, 0, &out2[0]);
            tausch.unpackRecvBuffer(0, 1, &out1[0]);
#endif

            double *expected1 = new double[(size+2*halowidth)*(size+2*halowidth)]{};
            double *expected2 = new double[(size+2*halowidth)*(size+2*halowidth)]{};

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

#ifdef TEST_SEND_TAUSCH_HIP
            hipFree(hip_in1);
            hipFree(hip_in2);
#endif

#ifdef TEST_RECV_TAUSCH_HIP
            hipFree(hip_out1);
            hipFree(hip_out2);
#endif

        }

    }

}

#if !defined(TEST_SEND_TAUSCH_CPU) && !defined(TEST_RECV_TAUSCH_CPU)
TEST_CASE("2 buffers, with pack/unpack, same MPI rank, GPUMultiCopy") {

    std::cout << " * Test: " << "2 buffers, with pack/unpack, same MPI rank, GPUMultiCopy" << std::endl;

#if defined(TEST_SEND_TAUSCH_OPENCL) || defined(TEST_RECV_TAUSCH_OPENCL)
    setupOpenCL();
#endif

    const std::vector<int> sizes = {3, 10, 100, 377};
    const std::vector<int> halowidths = {1, 2, 3};

    for(auto size : sizes) {

        for(auto halowidth : halowidths) {

            Tausch tausch(MPI_COMM_WORLD, false);
#if defined(TEST_SEND_TAUSCH_OPENCL) || defined(TEST_RECV_TAUSCH_OPENCL)
            tausch.setOpenCL(tauschcl_device, tauschcl_context, tauschcl_queue);
#endif

            std::vector<double> in1((size+2*halowidth)*(size+2*halowidth));
            std::vector<double> in2((size+2*halowidth)*(size+2*halowidth));
            std::vector<double> out1((size+2*halowidth)*(size+2*halowidth));
            std::vector<double> out2((size+2*halowidth)*(size+2*halowidth));
            for(int i = 0; i < size; ++i) {
                for(int j = 0; j < size; ++j) {
                    in1[(i+halowidth)*(size+2*halowidth) + j+halowidth] = i*size + j + 1;
                    in2[(i+halowidth)*(size+2*halowidth) + j+halowidth] = size*size + i*size + j + 1;
                }
            }

#ifdef TEST_SEND_TAUSCH_CUDA
            thrust::device_vector<double> cuda_in1(in1);
            thrust::device_vector<double> cuda_in2(in2);
#elif defined(TEST_SEND_TAUSCH_HIP)
            double *hip_in1, *hip_in2;
            hipMalloc(&hip_in1, (size+2*halowidth)*(size+2*halowidth)*sizeof(double));
            hipMalloc(&hip_in2, (size+2*halowidth)*(size+2*halowidth)*sizeof(double));
            hipMemcpy(hip_in1, &in1[0], (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyHostToDevice);
            hipMemcpy(hip_in2, &in2[0], (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyHostToDevice);
#elif defined(TEST_SEND_TAUSCH_OPENCL)
            cl::Buffer cl_in1(tausch.getOclQueue(), &in1[0], &in1[(size+2*halowidth)*(size+2*halowidth)], false);
            cl::Buffer cl_in2(tausch.getOclQueue(), &in2[0], &in2[(size+2*halowidth)*(size+2*halowidth)], false);
#endif

#ifdef TEST_RECV_TAUSCH_CUDA
            thrust::device_vector<double> cuda_out1(out1);
            thrust::device_vector<double> cuda_out2(out2);
#elif defined(TEST_RECV_TAUSCH_HIP)
            double *hip_out1, *hip_out2;
            hipMalloc(&hip_out1, (size+2*halowidth)*(size+2*halowidth)*sizeof(double));
            hipMalloc(&hip_out2, (size+2*halowidth)*(size+2*halowidth)*sizeof(double));
            hipMemcpy(hip_out1, &out1[0], (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyHostToDevice);
            hipMemcpy(hip_out2, &out2[0], (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyHostToDevice);
#elif defined(TEST_RECV_TAUSCH_OPENCL)
            cl::Buffer cl_out1(tausch.getOclQueue(), &out1[0], &out1[(size+2*halowidth)*(size+2*halowidth)], false);
            cl::Buffer cl_out2(tausch.getOclQueue(), &out2[0], &out2[(size+2*halowidth)*(size+2*halowidth)], false);
#endif

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

            int mpiRank;
            MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

            tausch.addSendHaloInfos(sendIndices, sizeof(double), 2);
            tausch.addRecvHaloInfos(recvIndices, sizeof(double), 2);

            tausch.setSendCommunicationStrategy(0, Tausch::Communication::GPUMultiCopy);
            tausch.setRecvCommunicationStrategy(0, Tausch::Communication::GPUMultiCopy);

#ifdef TEST_SEND_TAUSCH_CUDA
            tausch.packSendBufferCUDA(0, 0, thrust::raw_pointer_cast(cuda_in1.data()));
            tausch.packSendBufferCUDA(0, 1, thrust::raw_pointer_cast(cuda_in2.data()));
#elif defined(TEST_SEND_TAUSCH_HIP)
            tausch.packSendBufferHIP(0, 0, hip_in1);
            tausch.packSendBufferHIP(0, 1, hip_in2);
#elif defined(TEST_SEND_TAUSCH_OPENCL)
            tausch.packSendBufferOCL(0, 0, cl_in1);
            tausch.packSendBufferOCL(0, 1, cl_in2);
#else
            tausch.packSendBuffer(0, 0, &in1[0]);
            tausch.packSendBuffer(0, 1, &in2[0]);
#endif

            Status status = tausch.send(0, 0, mpiRank);
            tausch.recv(0, 0, mpiRank);

            status.wait();

#ifdef TEST_RECV_TAUSCH_CUDA
            tausch.unpackRecvBufferCUDA(0, 0, thrust::raw_pointer_cast(cuda_out2.data()));
            tausch.unpackRecvBufferCUDA(0, 1, thrust::raw_pointer_cast(cuda_out1.data()));
            thrust::copy(cuda_out1.begin(), cuda_out1.end(), out1.begin());
            thrust::copy(cuda_out2.begin(), cuda_out2.end(), out2.begin());
#elif defined(TEST_RECV_TAUSCH_HIP)
            tausch.unpackRecvBufferHIP(0, 0, hip_out2);
            tausch.unpackRecvBufferHIP(0, 1, hip_out1);
            hipMemcpy(&out1[0], hip_out1, (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyDeviceToHost);
            hipMemcpy(&out2[0], hip_out2, (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyDeviceToHost);
#elif defined(TEST_RECV_TAUSCH_OPENCL)
            tausch.unpackRecvBufferOCL(0, 0, cl_out2);
            tausch.unpackRecvBufferOCL(0, 1, cl_out1);
            cl::copy(tausch.getOclQueue(), cl_out1, &out1[0], &out1[(size+2*halowidth)*(size+2*halowidth)]);
            cl::copy(tausch.getOclQueue(), cl_out2, &out2[0], &out2[(size+2*halowidth)*(size+2*halowidth)]);
#else
            tausch.unpackRecvBuffer(0, 0, &out2[0]);
            tausch.unpackRecvBuffer(0, 1, &out1[0]);
#endif

            double *expected1 = new double[(size+2*halowidth)*(size+2*halowidth)]{};
            double *expected2 = new double[(size+2*halowidth)*(size+2*halowidth)]{};

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

#ifdef TEST_SEND_TAUSCH_HIP
            hipFree(hip_in1);
            hipFree(hip_in2);
#endif

#ifdef TEST_RECV_TAUSCH_HIP
            hipFree(hip_out1);
            hipFree(hip_out2);
#endif

        }

    }

}
#endif

TEST_CASE("2 buffers, with pack/unpack, multiple MPI ranks") {

    std::cout << " * Test: " << "2 buffers, with pack/unpack, multiple MPI rank" << std::endl;

#if defined(TEST_SEND_TAUSCH_OPENCL) || defined(TEST_RECV_TAUSCH_OPENCL)
    setupOpenCL();
#endif

    const std::vector<int> sizes = {3, 10, 100, 377};
    const std::vector<int> halowidths = {1, 2, 3};

    for(auto size : sizes) {

        for(auto halowidth : halowidths) {

            Tausch tausch(MPI_COMM_WORLD, false);
#if defined(TEST_SEND_TAUSCH_OPENCL) || defined(TEST_RECV_TAUSCH_OPENCL)
            tausch.setOpenCL(tauschcl_device, tauschcl_context, tauschcl_queue);
#endif

            std::vector<double> in1((size+2*halowidth)*(size+2*halowidth));
            std::vector<double> in2((size+2*halowidth)*(size+2*halowidth));
            std::vector<double> out1((size+2*halowidth)*(size+2*halowidth));
            std::vector<double> out2((size+2*halowidth)*(size+2*halowidth));
            for(int i = 0; i < size; ++i) {
                for(int j = 0; j < size; ++j) {
                    in1[(i+halowidth)*(size+2*halowidth) + j+halowidth] = i*size + j + 1;
                    in2[(i+halowidth)*(size+2*halowidth) + j+halowidth] = size*size + i*size + j + 1;
                }
            }

#ifdef TEST_SEND_TAUSCH_CUDA
            thrust::device_vector<double> cuda_in1(in1);
            thrust::device_vector<double> cuda_in2(in2);
#elif defined(TEST_SEND_TAUSCH_HIP)
            double *hip_in1, *hip_in2;
            hipMalloc(&hip_in1, (size+2*halowidth)*(size+2*halowidth)*sizeof(double));
            hipMalloc(&hip_in2, (size+2*halowidth)*(size+2*halowidth)*sizeof(double));
            hipMemcpy(hip_in1, &in1[0], (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyHostToDevice);
            hipMemcpy(hip_in2, &in2[0], (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyHostToDevice);
#elif defined(TEST_SEND_TAUSCH_OPENCL)
            cl::Buffer cl_in1(tausch.getOclQueue(), &in1[0], &in1[(size+2*halowidth)*(size+2*halowidth)], false);
            cl::Buffer cl_in2(tausch.getOclQueue(), &in2[0], &in2[(size+2*halowidth)*(size+2*halowidth)], false);
#endif

#ifdef TEST_RECV_TAUSCH_CUDA
            thrust::device_vector<double> cuda_out1(out1);
            thrust::device_vector<double> cuda_out2(out2);
#elif defined(TEST_RECV_TAUSCH_HIP)
            double *hip_out1, *hip_out2;
            hipMalloc(&hip_out1, (size+2*halowidth)*(size+2*halowidth)*sizeof(double));
            hipMalloc(&hip_out2, (size+2*halowidth)*(size+2*halowidth)*sizeof(double));
            hipMemcpy(hip_out1, &out1[0], (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyHostToDevice);
            hipMemcpy(hip_out2, &out2[0], (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyHostToDevice);
#elif defined(TEST_RECV_TAUSCH_OPENCL)
            cl::Buffer cl_out1(tausch.getOclQueue(), &out1[0], &out1[(size+2*halowidth)*(size+2*halowidth)], false);
            cl::Buffer cl_out2(tausch.getOclQueue(), &out2[0], &out2[(size+2*halowidth)*(size+2*halowidth)], false);
#endif

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

            int mpiRank, mpiSize;
            MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
            MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

            tausch.addSendHaloInfos(sendIndices, sizeof(double), 2);
            tausch.addRecvHaloInfos(recvIndices, sizeof(double), 2);

#ifdef TEST_SEND_TAUSCH_CUDA
            tausch.packSendBufferCUDA(0, 0, thrust::raw_pointer_cast(cuda_in1.data()));
            tausch.packSendBufferCUDA(0, 1, thrust::raw_pointer_cast(cuda_in2.data()));
#elif defined(TEST_SEND_TAUSCH_HIP)
            tausch.packSendBufferHIP(0, 0, hip_in1);
            tausch.packSendBufferHIP(0, 1, hip_in2);
#elif defined(TEST_SEND_TAUSCH_OPENCL)
            tausch.packSendBufferOCL(0, 0, cl_in1);
            tausch.packSendBufferOCL(0, 1, cl_in2);
#else
            tausch.packSendBuffer(0, 0, &in1[0]);
            tausch.packSendBuffer(0, 1, &in2[0]);
#endif

            Status status = tausch.send(0, 0, (mpiRank+1)%mpiSize);
            tausch.recv(0, 0, (mpiRank+mpiSize-1)%mpiSize);

            status.wait();

#ifdef TEST_RECV_TAUSCH_CUDA
            tausch.unpackRecvBufferCUDA(0, 0, thrust::raw_pointer_cast(cuda_out2.data()));
            tausch.unpackRecvBufferCUDA(0, 1, thrust::raw_pointer_cast(cuda_out1.data()));
            thrust::copy(cuda_out1.begin(), cuda_out1.end(), out1.begin());
            thrust::copy(cuda_out2.begin(), cuda_out2.end(), out2.begin());
#elif defined(TEST_RECV_TAUSCH_HIP)
            tausch.unpackRecvBufferHIP(0, 0, hip_out2);
            tausch.unpackRecvBufferHIP(0, 1, hip_out1);
            hipMemcpy(&out1[0], hip_out1, (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyDeviceToHost);
            hipMemcpy(&out2[0], hip_out2, (size+2*halowidth)*(size+2*halowidth)*sizeof(double), hipMemcpyDeviceToHost);
#elif defined(TEST_RECV_TAUSCH_OPENCL)
            tausch.unpackRecvBufferOCL(0, 0, cl_out2);
            tausch.unpackRecvBufferOCL(0, 1, cl_out1);
            cl::copy(tausch.getOclQueue(), cl_out1, &out1[0], &out1[(size+2*halowidth)*(size+2*halowidth)]);
            cl::copy(tausch.getOclQueue(), cl_out2, &out2[0], &out2[(size+2*halowidth)*(size+2*halowidth)]);
#else
            tausch.unpackRecvBuffer(0, 0, &out2[0]);
            tausch.unpackRecvBuffer(0, 1, &out1[0]);
#endif

            double *expected1 = new double[(size+2*halowidth)*(size+2*halowidth)]{};
            double *expected2 = new double[(size+2*halowidth)*(size+2*halowidth)]{};

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

#ifdef TEST_SEND_TAUSCH_HIP
            hipFree(hip_in1);
            hipFree(hip_in2);
#endif

#ifdef TEST_RECV_TAUSCH_HIP
            hipFree(hip_out1);
            hipFree(hip_out2);
#endif

        }

    }

}

