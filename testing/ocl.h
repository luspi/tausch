#ifndef TAUSCH_SETUP_OPENCL
#define TAUSCH_SETUP_OPENCL


static cl::Device tauschcl_device;
static cl::Context tauschcl_context;
static cl::CommandQueue tauschcl_queue;

static void setupOpenCL() {
    try {

        // Get platform
        std::vector<cl::Platform> all_platforms;
        cl::Platform::get(&all_platforms);
        cl::Platform tauschcl_platform;
        if(all_platforms.size() == 0) {
            std::cerr << "ERROR: no OpenCL capable platform found... Exiting!" << std::endl;
            std::exit(1);
        } else {
            tauschcl_platform = all_platforms[0];
            std::cout << "Using OpenCL platform: " << tauschcl_platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
        }

        for(size_t s = 0; s < all_platforms.size(); ++s) {

            tauschcl_platform = all_platforms[s];

            std::string name = tauschcl_platform.getInfo<CL_PLATFORM_NAME>();
            if(name == "Intel(R) OpenCL Graphics")
                break;
        }

        // Get device
        std::vector<cl::Device> all_devices;
        tauschcl_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
        tauschcl_device = all_devices[0];
        if(all_platforms.size() == 0) {
            std::cerr << "ERROR: no OpenCL capable device on platform found... Exiting!" << std::endl;
            std::exit(1);
        } else {
            tauschcl_device = all_devices[0];
            std::cout << "Using OpenCL device: " << tauschcl_device.getInfo<CL_DEVICE_NAME>() << std::endl;
        }

        // Create context and queue
        tauschcl_context = cl::Context({tauschcl_device});
        tauschcl_queue = cl::CommandQueue(tauschcl_context,tauschcl_device);

    } catch(cl::Error &error) {
        std::cout << "[setup] OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}


#endif
