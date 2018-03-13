/* Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef LAGHOS_ASYNC
#define LAGHOS_ASYNC

#include <stdio.h>
#include <stdarg.h>
#include <assert.h>
#include <mpi-ext.h>
#include <unistd.h>
#include <iostream>
#include <algorithm>    // std::find
#include <vector>       // std::vector

#include "comm.hpp"

#define GDASYNC_NO 0
#define GDASYNC_SY 1
#define GDASYNC_SA 2
#define GDASYNC_KI 3

#define MAX_CREQS 1024
#define MAX_FLUSH_CREQS 192
typedef std::map<uintptr_t, int>::iterator comm_memreg_it;
typedef std::map<uintptr_t, int> comm_memreg_map;
typedef std::vector<uintptr_t>vect_unistd;

// ***************************************************************************
// * GPUDirect Async communicator
// ***************************************************************************
class gdacomm{
    private:
        int mpi_rank=0;
        int mpi_size=0;
        int gpu_id=0;
        int gdasync=GDASYNC_NO;
        bool comm_initiated=false;
        CUstream *hStream=NULL;
        comm_memreg_map comm_regs;
        comm_request_t * ready_reqs;
        comm_request_t * recv_reqs;
        comm_request_t * send_reqs;
        comm_reg_t * mem_regs;
        int creqSendStart=0, creqSendCurr=0;
        int creqRecvStart=0, creqRecvCurr=0;
        int creqReadyStart=0, creqReadyCurr=0;
        int mregCurr=0;
        int dbg_log=0;
        vect_unistd pin_mem;

    private:
        gdacomm(){}
        gdacomm(gdacomm const&);
        void operator=(gdacomm const&);
    // *************************************************************************
    public:
        static gdacomm& Get(){
            static gdacomm gdacomm_singleton;
            return gdacomm_singleton;
        }
    
    // *************************************************************************
        void Init(const int _mpi_rank,
            const int _mpi_size,
            const int _device,
            const int _gdasync,
            CUstream *hStream);

        void Finalize();
        int isAsync();
        int isMPI();

        void newMemRegion(uintptr_t buffer, int size);
        comm_memreg_it getMemRegion(uintptr_t buffer, int size);
        int countSendReqs();
        int countRecvReqs();

        int iSend(
            void *send_buf, int size, 
            MPI_Datatype mpi_type, int dest_rank, 
            MPI_Comm comm, 
            MPI_Request * mpi_req,
            int mpi_tag,
            comm_memreg_it comm_reg_it
          );

        int iRecv(
            void *recv_buf, int size, 
            MPI_Datatype mpi_type, int source_rank, 
            MPI_Comm comm, 
            MPI_Request * mpi_req,
            int mpi_tag,
            comm_memreg_it comm_reg_it
          );

        int WaitAll(
            MPI_Request * mpi_req,
            int num_requests
        );

        int FlushAll(bool force_flush);

        int AsyncProgress();
        int AsyncWaitAllRecv();
        int AsyncWaitAllReady();
        int AsyncWaitAllSend();
    
        int pinMemory(void * ptr, int size);

        void Log(const char *fmt, ...);
};

#endif // LAGHOS_RAJA_CONFIG
