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

#include "gdacomm.hpp"

void gdacomm::Log(const char *fmt, ...)
{
    if(dbg_log)
    {
        char buffer[4096];
        va_list args;
        va_start(args, fmt);
        int rc = vsnprintf(buffer, sizeof(buffer), fmt, args);
        va_end(args);
        fprintf(stderr, "\033[32m[gdacomm][%d - %d] %s\033[m\n", mpi_rank, getpid(), buffer);
    }
}

// ***************************************************************************
// *   Setup
// ***************************************************************************
void gdacomm::Init(const int _mpi_rank,
    const int _mpi_size,
    const int _device,
    const int _gdasync,
    CUstream *_hStream
    )
{
    mpi_rank = _mpi_rank;
    mpi_size = _mpi_size;
    gpu_id   = _device;
    gdasync  = GDASYNC_NO;
    hStream  = _hStream;

    if(_gdasync == GDASYNC_NO) gdasync = GDASYNC_NO;
    if(_gdasync == GDASYNC_SY) gdasync = GDASYNC_SY;
    if(_gdasync == GDASYNC_SA) gdasync = GDASYNC_SA;
    if(_gdasync == GDASYNC_KI) gdasync = GDASYNC_KI;

    printf("\033[32m[laghos] GPUDirect Async=%d\033[m\n", gdasync);
    //printf("\033[32m[laghos] \033[31;1mGPUDirect Async=%d\033[m\n", gdasync);
    //ToDo: Should check if GDR, then check system requirements
    //ToDo: should set the communicator as Setup input
    COMM_CHECK(comm_init(MPI_COMM_WORLD, gpu_id));

    comm_initiated=true;

    ready_reqs = (comm_request_t*) calloc(MAX_CREQS, sizeof(comm_request_t));
    assert(ready_reqs);
    recv_reqs = (comm_request_t*) calloc(MAX_CREQS, sizeof(comm_request_t));
    assert(recv_reqs);
    send_reqs = (comm_request_t*) calloc(MAX_CREQS, sizeof(comm_request_t));
    assert(send_reqs);
    mem_regs = (comm_reg_t*)calloc(MAX_CREQS, sizeof(comm_reg_t));
    assert(mem_regs);

    creqReadyStart=0;
    creqReadyCurr=0;
    creqSendStart=0;
    creqSendCurr=0;
    creqRecvStart=0;
    creqRecvCurr=0;
    mregCurr=0;
}

void gdacomm::Finalize()
{
    if(comm_initiated == true)
    {
        //ToDo: deregister all memory regions
        //Unpin all memory
        //cleanup all memory structures
        mp_finalize();
        comm_initiated=false;

        free(ready_reqs);
        free(recv_reqs);
        free(send_reqs);
    }
}

/* ========== COMMUNICATIONS ========== */
int gdacomm::iSend(
    void *send_buf, int size, 
    MPI_Datatype mpi_type, int dest_rank, 
    MPI_Comm comm, 
    MPI_Request * mpi_req,
    int mpi_tag,
    comm_memreg_it comm_reg_it
    )
{
    assert(send_buf);
    assert(size);
    int ret=0;

    if(gdasync == GDASYNC_NO)
    {
        assert(mpi_req!=NULL);
        //ToDo: add MPI_CHECK
        ret = MPI_Isend(send_buf,
            size,
            mpi_type,
            dest_rank,
            mpi_tag,
            comm,
            mpi_req);
    }
    else
    {
        assert(comm_initiated==true);
        int indexRegion=0;
        //if(comm_reg_it == comm_reg_it.end()) Alloc new region?
        indexRegion = comm_reg_it->second;
        Log("iSend - gdasync=%d, creqSendCurr=%d, #region=%d, dstrank=%d, bytes=%d", 
            gdasync, creqSendCurr, indexRegion, dest_rank, size);
        if(gdasync == GDASYNC_SY)
        {
            COMM_CHECK(comm_wait_ready(dest_rank));
            COMM_CHECK(comm_isend(send_buf, 
                size, 
                mpi_type, 
                &mem_regs[indexRegion],
                dest_rank,
                &send_reqs[creqSendCurr])
            );
        }

        if(gdasync == GDASYNC_SA)
        {
            COMM_CHECK(comm_wait_ready_on_stream(dest_rank, (cudaStream_t)hStream));
            COMM_CHECK(comm_isend_on_stream(send_buf, 
                size, 
                mpi_type, 
                &mem_regs[indexRegion],
                dest_rank,
                &send_reqs[creqSendCurr],
                (cudaStream_t)hStream) //ToDo: Assume default stream?
            );
        }

        creqSendCurr=(creqSendCurr+1)%MAX_CREQS;
    }

    return 0;
}

int gdacomm::iRecv(
    void *recv_buf, int size, 
    MPI_Datatype mpi_type, int source_rank, 
    MPI_Comm comm, 
    MPI_Request * mpi_req,
    int mpi_tag,
    comm_memreg_it comm_reg_it
    )
{
    assert(recv_buf);
    assert(size);
    int ret=0;
    
    if(gdasync == GDASYNC_NO)
    {
        assert(mpi_req!=NULL);
        ret = MPI_Irecv(recv_buf,
            size,
            mpi_type,
            source_rank,
            mpi_tag,
            comm,
            mpi_req);
    }
    else
    {
        assert(comm_initiated==true);
        int indexRegion=0;
        //if(comm_reg_it == comm_reg_it.end()) Alloc new region?
        indexRegion = comm_reg_it->second;
        Log("iRecv - gdasync=%d, creqRecvCurr=%d, #region=%d", gdasync, creqRecvCurr, indexRegion);

        COMM_CHECK(comm_irecv(recv_buf, 
            size, 
            mpi_type, 
            &mem_regs[indexRegion],
            source_rank,
            &recv_reqs[creqRecvCurr])
        );
        
        if(gdasync == GDASYNC_SA)
            COMM_CHECK(comm_send_ready_on_stream(source_rank, 
                            &ready_reqs[creqReadyCurr],
                            (cudaStream_t)hStream));
        else
            COMM_CHECK(comm_send_ready(source_rank, 
                                &ready_reqs[creqReadyCurr]
                            )
                        );

        creqRecvCurr=(creqRecvCurr+1)%MAX_CREQS;
        creqReadyCurr=(creqReadyCurr+1)%MAX_CREQS;
    }

    return 0;
}

int gdacomm::WaitAll(
        MPI_Request * mpi_req,
        int num_requests
    )
{
    int ret=0;

    if(gdasync == GDASYNC_NO)
    {
        assert(mpi_req!=NULL);
        ret = MPI_Waitall(num_requests, mpi_req, MPI_STATUSES_IGNORE);
    }
    
    return 0;
}

int gdacomm::FlushAll(bool force_flush)
{
    int ret=0;

    if(gdasync != GDASYNC_NO)
    {
        assert(comm_initiated==true);
        if(
            force_flush == true || 
            (force_flush == false &&
                (creqSendStart + creqReadyStart >= MAX_FLUSH_CREQS) || 
                (creqRecvStart >= MAX_FLUSH_CREQS)
            )
        )
        {
            Log("WaitAll - gdasync=%d, creqSendStart=%d, creqRecvStart=%d creqReadyStart=%d, creqSendCurr=%d, creqRecvCurr=%d creqReadyCurr=%d", 
                gdasync, creqSendStart, creqRecvStart, creqReadyStart, creqSendCurr, creqRecvCurr, creqReadyCurr);

            comm_flush();
            creqReadyStart=0;
            creqReadyCurr=0;
            creqRecvStart=0;
            creqRecvCurr=0;
            creqSendStart=0;
            creqSendCurr=0;
        }
    }

    return 0;
}


int gdacomm::AsyncProgress()
{
    assert(comm_initiated==true);
    Log("gdasync=%d, creqSendCurr=%d, creqRecvCurr=%d creqReadyCurr=%d", 
            gdasync, creqSendCurr, creqRecvCurr, creqReadyCurr);
    comm_progress();
    return 0;
}

int gdacomm::AsyncWaitAllRecv()
{
    assert(comm_initiated==true);
    if(gdasync == GDASYNC_SA)
    {
        int numReqs = creqRecvCurr-creqRecvStart;
        if(numReqs > 0)
        {
            Log("gdasync=%d, creqRecvCurr=%d creqRecvStart=%d numReqs=%d", 
                gdasync, creqRecvCurr, creqRecvStart, numReqs);

            COMM_CHECK(comm_wait_all_on_stream(numReqs, 
                                            recv_reqs+creqRecvStart, 
                                            (cudaStream_t)hStream));
            creqRecvStart = creqRecvCurr;
        }
    }

    return 0;
}

int gdacomm::AsyncWaitAllReady()
{
    assert(comm_initiated==true);
    if(gdasync == GDASYNC_SA)
    {
        int numReqs = creqReadyCurr-creqReadyStart;
        if(numReqs > 0)
        {
            Log("gdasync=%d, creqReadyCurr=%d creqReadyStart=%d numReqs=%d", 
                gdasync, creqReadyCurr, creqReadyStart, numReqs);
            COMM_CHECK(comm_wait_all_on_stream(numReqs, 
                                                ready_reqs+creqReadyStart, 
                                                (cudaStream_t)hStream));
            creqReadyStart = creqReadyCurr;
        }
    }

    return 0;
}

int gdacomm::AsyncWaitAllSend()
{
    assert(comm_initiated==true);
    if(gdasync == GDASYNC_SA)
    {
        int numReqs = creqSendCurr-creqSendStart;
        if(numReqs > 0)
        {
            Log("gdasync=%d, creqSendCurr=%d creqSendStart=%d numReqs=%d", 
                gdasync, creqSendCurr, creqSendStart, numReqs);
            COMM_CHECK(comm_wait_all_on_stream(numReqs, 
                                                send_reqs+creqSendStart, 
                                                (cudaStream_t)hStream));
            creqSendStart = creqSendCurr;        
        }
    }
    
    return 0;
}

void gdacomm::newMemRegion(uintptr_t buffer, int size) {

    Log("newMemRegion - Looking for buffer %lx...", buffer);
    if(comm_regs.empty() || comm_regs.find(buffer) == comm_regs.end())
    {
        COMM_CHECK(comm_register((void*)buffer, (size_t)size, &mem_regs[mregCurr]));
        Log("newMemRegion - Associating buffer %lx to memreg %d (%p)", 
            buffer, mregCurr, &mem_regs[mregCurr]);

        comm_regs.insert ( std::pair<uintptr_t,uintptr_t>(buffer, mregCurr));
        mregCurr++;
    }
    Log("newMemRegion - region size=%d", comm_regs.size());
}

comm_memreg_it gdacomm::getMemRegion(uintptr_t buffer, int size) {
    Log("getMemRegion - region size=%d", comm_regs.size());
    
    if(comm_regs.empty() || comm_regs.find(buffer) == comm_regs.end())
        newMemRegion(buffer, size);

    comm_memreg_it iter = comm_regs.find(buffer);
    if(iter != comm_regs.end())
        Log("Found buffer %lx associated with memreg #%d", buffer, (iter->second));

    return iter;
}

/* ========== COUNTERS & INFO ========== */
int gdacomm::isAsync() {
    if(gdasync != GDASYNC_NO && gdasync != GDASYNC_SY)
        return 1;
    return 0;
}

int gdacomm::isMPI() {
    if(gdasync == GDASYNC_NO)
        return 1;
    return 0;
}

int gdacomm::countSendReqs() {
    return creqSendCurr;
}

int gdacomm::countRecvReqs() {
    return creqRecvCurr;
}

int gdacomm::pinMemory(void * ptr, int size) {
    if( pin_mem.empty() || ( std::find(pin_mem.begin(), pin_mem.end(), reinterpret_cast<std::uintptr_t>(ptr)) == pin_mem.end() ) )
    {
        assert(ptr);
        assert(size);
        //ToDo add cuda check
        cudaHostRegister ( ptr, (size_t) size, cudaHostRegisterPortable);
        pin_mem.push_back(reinterpret_cast<std::uintptr_t>(ptr));
        Log("pinMemory - ptr=%p (%lx) just pinned!", ptr, reinterpret_cast<std::uintptr_t>(ptr));
    }
    else
        Log("pinMemory - ptr=%p (%lx) already pinned!", ptr, reinterpret_cast<std::uintptr_t>(ptr));
    return 0;
}



