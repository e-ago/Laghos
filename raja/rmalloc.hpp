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
#ifndef LAGHOS_RAJA_MALLOC
#define LAGHOS_RAJA_MALLOC

namespace mfem {

// *****************************************************************************
static const bool mem_manager_uvm  = true;
static const bool mem_manager_std  = false;
  
#ifdef __NVCC__
static const bool mng = mem_manager_uvm;
#else
static const bool mng = mem_manager_std;
#endif

// *****************************************************************************
template <typename T, bool = mem_manager_std> class rmalloc;

// CPU *************************************************************************
template<typename T> class rmalloc<T,false> {
public:
  void* operator new(size_t n) {
    dbg("+]\033[m");
    return new T[n];
  }
  void operator delete(void *ptr) {
    dbg("-]\033[m");
    delete[] static_cast<T*>(ptr);
    ptr = nullptr;
  }
};

// GPU *************************************************************************
#ifdef __NVCC__
template<typename T> class rmalloc<T,mem_manager_uvm> {
public:
  void* operator new(size_t n) {
    void *ptr;
    dbg("+]\033[m");
    cudaMallocManaged(&ptr, n*sizeof(T), cudaMemAttachGlobal);
    cudaDeviceSynchronize();
    return ptr;
  }

  void operator delete(void *ptr) {
    dbg("-]\033[m");
    cudaDeviceSynchronize();
    cudaFree(ptr);
    ptr = nullptr;
  }
};
#endif // __NVCC__
  
} // mfem

#endif // LAGHOS_RAJA_MALLOC