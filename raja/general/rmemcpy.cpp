// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
#include "../raja.hpp"

namespace mfem {

  // *************************************************************************
  void* rmemcpy::rHtoH(void *dest, const void *src, std::size_t count){
    dbg(">\033[m");
    if (count==0) return dest;
    assert(src); assert(dest);
    std::memcpy(dest,src,count);
    return dest;
  }

  // *************************************************************************
  void* rmemcpy::rHtoD(void *dest, const void *src, std::size_t count){
    dbg(">\033[m");
    if (count==0) return dest;
    assert(src); assert(dest);
    if (!rconfig::Get().Cuda()) return std::memcpy(dest,src,count);
#ifdef __NVCC__
    if (!rconfig::Get().Uvm())
      checkCudaErrors(cuMemcpyHtoD((CUdeviceptr)dest,src,count));
    else checkCudaErrors(cuMemcpy((CUdeviceptr)dest,(CUdeviceptr)src,count));
#endif
    return dest;
  }

  // ***************************************************************************
  void* rmemcpy::rDtoH(void *dest, const void *src, std::size_t count){
    dbg("<\033[m");
    if (count==0) return dest;
    assert(src); assert(dest);
    if (!rconfig::Get().Cuda()) return std::memcpy(dest,src,count);
#ifdef __NVCC__
    if (!rconfig::Get().Uvm())
      checkCudaErrors(cuMemcpyDtoH(dest,(CUdeviceptr)src,count));
    else checkCudaErrors(cuMemcpy((CUdeviceptr)dest,(CUdeviceptr)src,count));
#endif
    return dest;
  }
  
  // ***************************************************************************
  void* rmemcpy::rDtoD(void *dest, const void *src, std::size_t count){
    dbg("<\033[m");
    if (count==0) return dest;
    assert(src); assert(dest);
    if (!rconfig::Get().Cuda()) return std::memcpy(dest,src,count);
#ifdef __NVCC__
    if (!rconfig::Get().Uvm())
      checkCudaErrors(cuMemcpyDtoD((CUdeviceptr)dest,(CUdeviceptr)src,count));
    else checkCudaErrors(cuMemcpy((CUdeviceptr)dest,(CUdeviceptr)src,count));
#endif
    return dest;
  }

  // *************************************************************************
  void* rmemcpy::rHtoDAsync(void *dest, const void *src, std::size_t count){
    dbg(">\033[m");
    if (count==0) return dest;
    assert(src); assert(dest);
    if (!rconfig::Get().Cuda()) return std::memcpy(dest,src,count);
#ifdef __NVCC__
    if (!rconfig::Get().Uvm())
      checkCudaErrors(cuMemcpyHtoDAsync((CUdeviceptr)dest,src,count,0));
    else checkCudaErrors(cuMemcpyAsync((CUdeviceptr)dest,(CUdeviceptr)src,count,0));
#endif
    return dest;
  }

  // ***************************************************************************
  void* rmemcpy::rDtoHAsync(void *dest, const void *src, std::size_t count){
    dbg("<\033[m");
    if (count==0) return dest;
    assert(src); assert(dest);
    if (!rconfig::Get().Cuda()) return std::memcpy(dest,src,count);
#ifdef __NVCC__
    if (!rconfig::Get().Uvm())
      checkCudaErrors(cuMemcpyDtoHAsync(dest,(CUdeviceptr)src,count,0));
    else checkCudaErrors(cuMemcpyAsync((CUdeviceptr)dest,(CUdeviceptr)src,count,0));
#endif
    return dest;
  }
  
  // ***************************************************************************
  void* rmemcpy::rDtoDAsync(void *dest, const void *src, std::size_t count){
    dbg("<\033[m");
    if (count==0) return dest;
    assert(src); assert(dest);
    if (!rconfig::Get().Cuda()) return std::memcpy(dest,src,count);
#ifdef __NVCC__
    if (!rconfig::Get().Uvm())
      checkCudaErrors(cuMemcpyDtoDAsync((CUdeviceptr)dest,(CUdeviceptr)src,count,0));
    else checkCudaErrors(cuMemcpyAsync((CUdeviceptr)dest,(CUdeviceptr)src,count,0));
#endif
    return dest;
  }

} // mfem
