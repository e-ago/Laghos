// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.
#include "../raja.hpp"
#include "../../gdacomm/gdacomm.hpp"

namespace mfem {
  
  // ***************************************************************************
  // * RajaConformingProlongationOperator
  // ***************************************************************************
  RajaConformingProlongationOperator::RajaConformingProlongationOperator
  (ParFiniteElementSpace &pfes): RajaOperator(pfes.GetVSize(), pfes.GetTrueVSize()),
                                 external_ldofs(),
                                 d_external_ldofs(Height()-Width()), // size can be 0 here
                                 gc(new RajaCommD(pfes)),
                                 kMaxTh(0){
    Array<int> ldofs;
    Table &group_ldof = gc->GroupLDofTable();
    external_ldofs.Reserve(Height()-Width());
    for (int gr = 1; gr < group_ldof.Size(); gr++)
    {
      if (!gc->GetGroupTopology().IAmMaster(gr)) 
      {
        ldofs.MakeRef(group_ldof.GetRow(gr), group_ldof.RowSize(gr));
        external_ldofs.Append(ldofs);
      }
    }
    external_ldofs.Sort();
#ifdef __NVCC__
    const int HmW=Height()-Width();
    if (HmW>0) d_external_ldofs=external_ldofs;
#endif
    assert(external_ldofs.Size() == Height()-Width());
    const int m = external_ldofs.Size();
    for (int i = 1; i < m; i++){
      const int diff =(external_ldofs[i]-external_ldofs[i-1]);
      if (diff>kMaxTh) kMaxTh=diff;
    }
    //gc->PrintInfo(); 
    //pfes.Dof_TrueDof_Matrix()->PrintCommPkg();
  }

  // ***************************************************************************
  // * ~RajaConformingProlongationOperator
  // ***************************************************************************
  RajaConformingProlongationOperator::~RajaConformingProlongationOperator(){
    delete  gc;
  }

  // ***************************************************************************
  // * k_Mult
  // ***************************************************************************
#ifdef __NVCC__
  static __global__
  void k_Mult2(double *y,const double *x,const int *external_ldofs,const int N){
    const int i = blockIdx.x;
    if (i>=N)return;
    const int j=(i>0)?external_ldofs[i-1]+1:0;
    const int end = external_ldofs[i];
    const int k = threadIdx.x;
    if (k>=(end-j)) return;
    y[j+k]=x[j-i+k];
  }
#endif

  // ***************************************************************************
  // * Device Mult
  // ***************************************************************************
  void RajaConformingProlongationOperator::d_Mult(const RajaVector &x,
                                                  RajaVector &y) const{
    push(Coral);
    const double *d_xdata = x.GetData();
    const int in_layout = 2; // 2 - input is ltdofs array
    
    push(d_BcastBegin,Coral);
    gc->d_BcastBegin(const_cast<double*>(d_xdata), in_layout);
    pop();
    
    push(d_Mult_Work,Coral);
    int j = 0;
    double *d_ydata = y.GetData(); 
    const int m = external_ldofs.Size();
#ifndef __NVCC__
    for (int i = 0; i < m; i++){
      const int end = external_ldofs[i];
      std::copy(d_xdata+j-i, d_xdata+end-i, d_ydata+j);
      j = end+1;
    }
#else
    if (m>0){
      k_Mult2<<<m,kMaxTh>>>(d_ydata,d_xdata,d_external_ldofs,m);
      j = external_ldofs[m-1]+1;
    }
#endif
#ifndef __NVCC__
    std::copy(d_xdata+j-m, d_xdata+Width(), d_ydata+j);
#else
    if(gdacomm::Get().isAsync())
        rmemcpy::rDtoDAsync(d_ydata+j,d_xdata+j-m,(Width()+m-j)*sizeof(double));
    else
        rmemcpy::rDtoD(d_ydata+j,d_xdata+j-m,(Width()+m-j)*sizeof(double));
#endif
    pop();
    
    push(d_BcastEnd,Coral);
    const int out_layout = 0; // 0 - output is ldofs array
    gc->d_BcastEnd(d_ydata, out_layout);
    pop();
    pop();
  }

  
  // ***************************************************************************
  // * k_Mult
  // ***************************************************************************
#ifdef __NVCC__
  static __global__
  void k_MultTranspose2(double *y,const double *x,const int *external_ldofs,const int N){
    const int i = blockIdx.x;
    if (i>=N)return;
    const int j=(i>0)?external_ldofs[i-1]+1:0;
    const int end = external_ldofs[i];
    const int k = threadIdx.x;
    if (k>=(end-j)) return;
    y[j-i+k]=x[j+k];
  }
#endif
  
  // ***************************************************************************
  // * Device MultTranspose
  // ***************************************************************************
  void RajaConformingProlongationOperator::d_MultTranspose(const RajaVector &x,
                                                           RajaVector &y) const{
    push(Coral);
    const double *d_xdata = x.GetData();
    
    push(d_ReduceBegin,Coral);
    gc->d_ReduceBegin(d_xdata);
    pop();
    
    push(d_MultTranspose_Work,Coral);
    int j = 0;
    double *d_ydata = y.GetData();
    const int m = external_ldofs.Size();
#ifndef __NVCC__
    for (int i = 0; i < m; i++){
      const int end = external_ldofs[i];
      std::copy(d_xdata+j, d_xdata+end, d_ydata+j-i);
      j = end+1;
    }
#else
    if (m>0){
      k_MultTranspose2<<<m,kMaxTh>>>(d_ydata,d_xdata,d_external_ldofs,m);
      j = external_ldofs[m-1]+1;
    }
#endif
#ifndef __NVCC__
    std::copy(d_xdata+j, d_xdata+Height(), d_ydata+j-m);
#else
    if(gdacomm::Get().isAsync())
        rmemcpy::rDtoDAsync(d_ydata+j-m,d_xdata+j,(Height()-j)*sizeof(double));
    else
        rmemcpy::rDtoD(d_ydata+j-m,d_xdata+j,(Height()-j)*sizeof(double));

#endif
    pop();
    push(d_ReduceEnd,Coral);
    const int out_layout = 2; // 2 - output is an array on all ltdofs
    gc->d_ReduceEnd<double>(d_ydata, out_layout, GroupCommunicator::Sum);
    pop();
    pop();

    //ele: cudasync?
  }

  // ***************************************************************************
  // * Host Mult
  // ***************************************************************************
  void RajaConformingProlongationOperator::h_Mult(const Vector &x,
                                                  Vector &y) const{
    push(Coral);
    const double *xdata = x.GetData();
    double *ydata = y.GetData(); 
    const int m = external_ldofs.Size();
    const int in_layout = 2; // 2 - input is ltdofs array
    push(BcastBegin,Moccasin);
    gc->BcastBegin(const_cast<double*>(xdata), in_layout);
    pop();
    int j = 0;
    for (int i = 0; i < m; i++){
      const int end = external_ldofs[i];
      std::copy(xdata+j-i, xdata+end-i, ydata+j);
      j = end+1;
    }
    std::copy(xdata+j-m, xdata+Width(), ydata+j);
    const int out_layout = 0; // 0 - output is ldofs array
    push(BcastEnd,PeachPuff);
    gc->BcastEnd(ydata, out_layout);
    pop();
    pop();
  }

  // ***************************************************************************
  // * Host MultTranspose
  // ***************************************************************************
  void RajaConformingProlongationOperator::h_MultTranspose(const Vector &x,
                                                           Vector &y) const{
    push(Coral);
    const double *xdata = x.GetData();
    double *ydata = y.GetData();
    const int m = external_ldofs.Size();
    push(ReduceBegin,PapayaWhip);
    gc->ReduceBegin(xdata);
    pop();
    int j = 0;
    for (int i = 0; i < m; i++)   {
      const int end = external_ldofs[i];
      std::copy(xdata+j, xdata+end, ydata+j-i);
      j = end+1;
    }
    std::copy(xdata+j, xdata+Height(), ydata+j-m);
    const int out_layout = 2; // 2 - output is an array on all ltdofs
    push(ReduceEnd,LavenderBlush);
    gc->ReduceEnd<double>(ydata, out_layout, GroupCommunicator::Sum);
    pop();
    pop();
  }

} // namespace mfem
