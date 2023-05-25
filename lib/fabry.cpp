#include <cassert>
#include <iostream>
#include <mpi.h>

#include <fabry/fabry.hpp>

namespace fabry {

namespace {
	template<typename T>
	opaque_handle decay(T x) {
		int cx = x;
		return static_cast<opaque_handle>(cx);
	}

	template<typename T>
	T lift(opaque_handle p) {
		return static_cast<int>(p);
	}
}

template<> opaque_handle get_type<int>() { return decay(MPI_INT); }

template<> opaque_handle get_type<unsigned long>() { return decay(MPI_UNSIGNED_LONG); }

communicator::communicator()
: p_{decay(MPI_COMM_NULL)}, rank_{0}, n_ranks_{0} { }

communicator::communicator(opaque_handle p)
: p_{p} {
	MPI_Comm_rank(lift<MPI_Comm>(p_), &rank_);
	MPI_Comm_size(lift<MPI_Comm>(p_), &n_ranks_);
}

communicator::communicator(world_tag)
: p_{decay(MPI_COMM_WORLD)} {
	MPI_Comm_rank(lift<MPI_Comm>(p_), &rank_);
	MPI_Comm_size(lift<MPI_Comm>(p_), &n_ranks_);
}

bool pollable::poll(opaque_handle h) {
	assert(s_ == stage::in_progress);
	MPI_Request req = lift<MPI_Request>(h);
	int done;
	MPI_Test(&req, &done, MPI_STATUS_IGNORE);
	if(done) {
		s_ = stage::complete;
		return true;
	}
	return false;
}

namespace internal {
	void initialize_fabric(int *argcp, char ***argvp) {
		MPI_Init(argcp, argvp);
	}

	void finalize_fabric() {
		MPI_Finalize();
	}

	void initiate_blocking(barrier_icb &b) {
		MPI_Barrier(lift<MPI_Comm>(b.com->get_handle()));
	}

	opaque_handle initiate_nonblocking(barrier_icb &b) {
		MPI_Request req;
		MPI_Ibarrier(lift<MPI_Comm>(b.com->get_handle()), &req);
		return decay(req);
	}

	void initiate_blocking(bcast_root_icb &b) {
		MPI_Bcast(const_cast<void *>(b.in), b.n, lift<MPI_Datatype>(b.dtype), b.com->rank(),
				lift<MPI_Comm>(b.com->get_handle()));
	}

	void initiate_blocking(bcast_nonroot_icb &b) {
		MPI_Bcast(b.out, b.n, lift<MPI_Datatype>(b.dtype), b.rrk,
				lift<MPI_Comm>(b.com->get_handle()));
	}

	opaque_handle initiate_nonblocking(bcast_root_icb &b) {
		MPI_Request req;
		MPI_Ibcast(const_cast<void *>(b.in), b.n, lift<MPI_Datatype>(b.dtype), b.com->rank(),
				lift<MPI_Comm>(b.com->get_handle()), &req);
		return decay(req);
	}

	opaque_handle initiate_nonblocking(bcast_nonroot_icb &b) {
		MPI_Request req;
		MPI_Ibcast(b.out, b.n, lift<MPI_Datatype>(b.dtype), b.rrk,
				lift<MPI_Comm>(b.com->get_handle()), &req);
		return decay(req);
	}

	void initiate_blocking(gather_root_icb &b) {
		MPI_Gather(b.in, b.n, lift<MPI_Datatype>(b.dtype),
				b.out, b.n, lift<MPI_Datatype>(b.dtype), b.com->rank(),
				lift<MPI_Comm>(b.com->get_handle()));
	}

	void initiate_blocking(gather_nonroot_icb &b) {
		MPI_Gather(b.in, b.n, lift<MPI_Datatype>(b.dtype),
				nullptr, 0, lift<MPI_Datatype>(b.dtype), b.rrk,
				lift<MPI_Comm>(b.com->get_handle()));
	}

	opaque_handle initiate_nonblocking(reduce_root_icb &b) {
		MPI_Request req;
		MPI_Ireduce(const_cast<void *>(b.in), b.out, b.n,
				lift<MPI_Datatype>(b.dtype), MPI_SUM, b.com->rank(),
				lift<MPI_Comm>(b.com->get_handle()), &req);
		return decay(req);
	}

	opaque_handle initiate_nonblocking(reduce_nonroot_icb &b) {
		MPI_Request req;
		MPI_Ireduce(b.in, nullptr, b.n,
				lift<MPI_Datatype>(b.dtype), MPI_SUM, b.rrk,
				lift<MPI_Comm>(b.com->get_handle()), &req);
		return decay(req);
	}
}

passive_rdma_base::passive_rdma_base()
: p_{decay(MPI_WIN_NULL)}, size_{0}, mapping_{nullptr} { }

passive_rdma_base::passive_rdma_base(no_root_tag, communicator &com, size_t unit, size_t size)
: com_{&com}, size_{size} {
	MPI_Win win;
	MPI_Win_allocate(size * unit, unit, MPI_INFO_NULL,
			lift<MPI_Comm>(com.get_handle()), &mapping_, &win);
	p_ = decay(win);
}

passive_rdma_base::~passive_rdma_base() {
	if(p_ != decay(MPI_WIN_NULL)) {
		std::cerr << "fabry: ~passive_rdma_base() called on live object" << std::endl;
		abort();
	}
}

void passive_rdma_base::dispose(no_root_tag) {
	auto win = lift<MPI_Win>(p_);
	MPI_Win_free(&win);
	p_ = decay(MPI_WIN_NULL);
}

void passive_rdma_base::do_get_sync(opaque_handle dtype, int rk, void *out) {
	MPI_Win_lock(MPI_LOCK_SHARED, rk, MPI_MODE_NOCHECK, lift<MPI_Win>(p_));
	MPI_Get(out, size_, lift<MPI_Datatype>(dtype),
			rk, 0, size_, lift<MPI_Datatype>(dtype), lift<MPI_Win>(p_));
	MPI_Win_unlock(rk, lift<MPI_Win>(p_));
}

void passive_rdma_base::do_put_sync(opaque_handle dtype, int rk, const void *in) {
	MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rk, MPI_MODE_NOCHECK, lift<MPI_Win>(p_));
	MPI_Put(in, size_, lift<MPI_Datatype>(dtype),
			rk, 0, size_, lift<MPI_Datatype>(dtype), lift<MPI_Win>(p_));
	MPI_Win_unlock(rk, lift<MPI_Win>(p_));
}

passive_rdma_base_scope::passive_rdma_base_scope(passive_rdma_base &r)
: r_{&r} {
	MPI_Win_lock(MPI_LOCK_EXCLUSIVE, r_->com_->rank(), MPI_MODE_NOCHECK, lift<MPI_Win>(r_->p_));
}

passive_rdma_base_scope::~passive_rdma_base_scope() {
	MPI_Win_unlock(r_->com_->rank(), lift<MPI_Win>(r_->p_));
}

} // namespace fabry
