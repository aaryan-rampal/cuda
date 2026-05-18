#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>

// Wrapper macro for CUDA error handling
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

int main(){
    int count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&count));
    std::cout << count << " CUDA devices\n";

    for (int i = 0; i < count; i++) {
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, i));
        std::cout << "\nDevice " << i << ": " << prop.name << "\n";
        std::cout << "  UUID: ";
        for (int b = 0; b < 16; b++) printf("%02x", (unsigned char)prop.uuid.bytes[b]);
        std::cout << "\n";
        std::cout << "  LUID: " << prop.luid << "\n";
        std::cout << "  LUID Device Node Mask: " << prop.luidDeviceNodeMask << "\n";
        std::cout << "  Total Global Memory: " << prop.totalGlobalMem << " bytes\n";
        std::cout << "  Shared Memory per Block: " << prop.sharedMemPerBlock << " bytes\n";
        std::cout << "  Registers per Block: " << prop.regsPerBlock << "\n";
        std::cout << "  Warp Size: " << prop.warpSize << "\n";
        std::cout << "  Mem Pitch: " << prop.memPitch << "\n";
        std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << "\n";
        std::cout << "  Max Threads Dim: " << prop.maxThreadsDim[0] << " x " << prop.maxThreadsDim[1] << " x " << prop.maxThreadsDim[2] << "\n";
        std::cout << "  Max Grid Size: " << prop.maxGridSize[0] << " x " << prop.maxGridSize[1] << " x " << prop.maxGridSize[2] << "\n";
        std::cout << "  Clock Rate: " << prop.clockRate << " kHz\n";
        std::cout << "  Total Constant Memory: " << prop.totalConstMem << " bytes\n";
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << "\n";
        std::cout << "  Texture Alignment: " << prop.textureAlignment << "\n";
        std::cout << "  Texture Pitch Alignment: " << prop.texturePitchAlignment << "\n";
        std::cout << "  Device Overlap: " << prop.deviceOverlap << "\n";
        std::cout << "  MultiProcessor Count: " << prop.multiProcessorCount << "\n";
        std::cout << "  Kernel Exec Timeout Enabled: " << prop.kernelExecTimeoutEnabled << "\n";
        std::cout << "  Integrated: " << prop.integrated << "\n";
        std::cout << "  Can Map Host Memory: " << prop.canMapHostMemory << "\n";
        std::cout << "  Compute Mode: " << prop.computeMode << "\n";
        std::cout << "  Max Texture 1D: " << prop.maxTexture1D << "\n";
        std::cout << "  Max Texture 1D Mipmap: " << prop.maxTexture1DMipmap << "\n";
        std::cout << "  Max Texture 1D Linear: " << prop.maxTexture1DLinear << "\n";
        std::cout << "  Max Texture 2D: " << prop.maxTexture2D[0] << " x " << prop.maxTexture2D[1] << "\n";
        std::cout << "  Max Texture 2D Mipmap: " << prop.maxTexture2DMipmap[0] << " x " << prop.maxTexture2DMipmap[1] << "\n";
        std::cout << "  Max Texture 2D Linear: " << prop.maxTexture2DLinear[0] << " x " << prop.maxTexture2DLinear[1] << " x " << prop.maxTexture2DLinear[2] << "\n";
        std::cout << "  Max Texture 2D Gather: " << prop.maxTexture2DGather[0] << " x " << prop.maxTexture2DGather[1] << "\n";
        std::cout << "  Max Texture 3D: " << prop.maxTexture3D[0] << " x " << prop.maxTexture3D[1] << " x " << prop.maxTexture3D[2] << "\n";
        std::cout << "  Max Texture 3D Alt: " << prop.maxTexture3DAlt[0] << " x " << prop.maxTexture3DAlt[1] << " x " << prop.maxTexture3DAlt[2] << "\n";
        std::cout << "  Max Texture Cubemap: " << prop.maxTextureCubemap << "\n";
        std::cout << "  Max Texture 1D Layered: " << prop.maxTexture1DLayered[0] << " x " << prop.maxTexture1DLayered[1] << "\n";
        std::cout << "  Max Texture 2D Layered: " << prop.maxTexture2DLayered[0] << " x " << prop.maxTexture2DLayered[1] << " x " << prop.maxTexture2DLayered[2] << "\n";
        std::cout << "  Max Texture Cubemap Layered: " << prop.maxTextureCubemapLayered[0] << " x " << prop.maxTextureCubemapLayered[1] << "\n";
        std::cout << "  Max Surface 1D: " << prop.maxSurface1D << "\n";
        std::cout << "  Max Surface 2D: " << prop.maxSurface2D[0] << " x " << prop.maxSurface2D[1] << "\n";
        std::cout << "  Max Surface 3D: " << prop.maxSurface3D[0] << " x " << prop.maxSurface3D[1] << " x " << prop.maxSurface3D[2] << "\n";
        std::cout << "  Max Surface 1D Layered: " << prop.maxSurface1DLayered[0] << " x " << prop.maxSurface1DLayered[1] << "\n";
        std::cout << "  Max Surface 2D Layered: " << prop.maxSurface2DLayered[0] << " x " << prop.maxSurface2DLayered[1] << " x " << prop.maxSurface2DLayered[2] << "\n";
        std::cout << "  Max Surface Cubemap: " << prop.maxSurfaceCubemap << "\n";
        std::cout << "  Max Surface Cubemap Layered: " << prop.maxSurfaceCubemapLayered[0] << " x " << prop.maxSurfaceCubemapLayered[1] << "\n";
        std::cout << "  Surface Alignment: " << prop.surfaceAlignment << "\n";
        std::cout << "  Concurrent Kernels: " << prop.concurrentKernels << "\n";
        std::cout << "  ECC Enabled: " << prop.ECCEnabled << "\n";
        std::cout << "  PCI Bus ID: " << prop.pciBusID << "\n";
        std::cout << "  PCI Device ID: " << prop.pciDeviceID << "\n";
        std::cout << "  PCI Domain ID: " << prop.pciDomainID << "\n";
        std::cout << "  TCC Driver: " << prop.tccDriver << "\n";
        std::cout << "  Async Engine Count: " << prop.asyncEngineCount << "\n";
        std::cout << "  Unified Addressing: " << prop.unifiedAddressing << "\n";
        std::cout << "  Memory Clock Rate: " << prop.memoryClockRate << " kHz\n";
        std::cout << "  Memory Bus Width: " << prop.memoryBusWidth << " bits\n";
        std::cout << "  L2 Cache Size: " << prop.l2CacheSize << " bytes\n";
        std::cout << "  Persisting L2 Cache Max Size: " << prop.persistingL2CacheMaxSize << " bytes\n";
        std::cout << "  Max Threads per MultiProcessor: " << prop.maxThreadsPerMultiProcessor << "\n";
        std::cout << "  Stream Priorities Supported: " << prop.streamPrioritiesSupported << "\n";
        std::cout << "  Global L1 Cache Supported: " << prop.globalL1CacheSupported << "\n";
        std::cout << "  Local L1 Cache Supported: " << prop.localL1CacheSupported << "\n";
        std::cout << "  Shared Memory per MultiProcessor: " << prop.sharedMemPerMultiprocessor << " bytes\n";
        std::cout << "  Registers per MultiProcessor: " << prop.regsPerMultiprocessor << "\n";
        std::cout << "  Managed Memory: " << prop.managedMemory << "\n";
        std::cout << "  Is Multi-GPU Board: " << prop.isMultiGpuBoard << "\n";
        std::cout << "  Multi-GPU Board Group ID: " << prop.multiGpuBoardGroupID << "\n";
        std::cout << "  Host Native Atomic Supported: " << prop.hostNativeAtomicSupported << "\n";
        std::cout << "  Single to Double Precision Perf Ratio: " << prop.singleToDoublePrecisionPerfRatio << "\n";
        std::cout << "  Pageable Memory Access: " << prop.pageableMemoryAccess << "\n";
        std::cout << "  Concurrent Managed Access: " << prop.concurrentManagedAccess << "\n";
        std::cout << "  Compute Preemption Supported: " << prop.computePreemptionSupported << "\n";
        std::cout << "  Can Use Host Pointer For Registered Mem: " << prop.canUseHostPointerForRegisteredMem << "\n";
        std::cout << "  Cooperative Launch: " << prop.cooperativeLaunch << "\n";
        std::cout << "  Cooperative Multi-Device Launch: " << prop.cooperativeMultiDeviceLaunch << "\n";
        std::cout << "  Shared Memory per Block Opt-in: " << prop.sharedMemPerBlockOptin << " bytes\n";
        std::cout << "  Pageable Memory Access Uses Host Page Tables: " << prop.pageableMemoryAccessUsesHostPageTables << "\n";
        std::cout << "  Direct Managed Mem Access From Host: " << prop.directManagedMemAccessFromHost << "\n";
        std::cout << "  Max Blocks per MultiProcessor: " << prop.maxBlocksPerMultiProcessor << "\n";
        std::cout << "  Access Policy Max Window Size: " << prop.accessPolicyMaxWindowSize << "\n";
        std::cout << "  Reserved Shared Memory per Block: " << prop.reservedSharedMemPerBlock << " bytes\n";
        std::cout << "  Host Register Supported: " << prop.hostRegisterSupported << "\n";
        std::cout << "  Sparse CUDA Array Supported: " << prop.sparseCudaArraySupported << "\n";
        std::cout << "  Host Register Read-Only Supported: " << prop.hostRegisterReadOnlySupported << "\n";
        std::cout << "  Timeline Semaphore Interop Supported: " << prop.timelineSemaphoreInteropSupported << "\n";
        std::cout << "  Memory Pools Supported: " << prop.memoryPoolsSupported << "\n";
        std::cout << "  GPU Direct RDMA Supported: " << prop.gpuDirectRDMASupported << "\n";
        std::cout << "  GPU Direct RDMA Flush Writes Options: " << prop.gpuDirectRDMAFlushWritesOptions << "\n";
        std::cout << "  GPU Direct RDMA Writes Ordering: " << prop.gpuDirectRDMAWritesOrdering << "\n";
        std::cout << "  Memory Pool Supported Handle Types: " << prop.memoryPoolSupportedHandleTypes << "\n";
        std::cout << "  Deferred Mapping CUDA Array Supported: " << prop.deferredMappingCudaArraySupported << "\n";
        std::cout << "  IPC Event Supported: " << prop.ipcEventSupported << "\n";
        std::cout << "  Cluster Launch: " << prop.clusterLaunch << "\n";
        std::cout << "  Unified Function Pointers: " << prop.unifiedFunctionPointers << "\n";
    }

    return 0;
}
