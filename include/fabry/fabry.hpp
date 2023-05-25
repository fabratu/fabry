#pragma once

#include <cassert>
#include <cstddef>
#include <utility>

#include <fabry/internal-base.hpp>
#include <fabry/internal-icb.hpp>

namespace fabry {

struct program {
	program(int *argcp, char ***argvp) {
		internal::initialize_fabric(argcp, argvp);
	}

	program(const program &) = delete;

	~program() {
		internal::finalize_fabric();
	}

	program &operator= (const program &) = delete;
};

template<typename T>
opaque_handle get_type();

struct no_root_tag {
};

struct this_root_tag {
};

struct other_root_tag {
	int rk;
};

static constexpr no_root_tag collective;
static constexpr this_root_tag this_root;
static constexpr other_root_tag zero_root{0};

struct world_tag {
};

static constexpr world_tag world;

struct communicator {
	friend void swap(communicator &x, communicator &y) {
		std::swap(x.p_, y.p_);
		std::swap(x.rank_, y.rank_);
		std::swap(x.n_ranks_, y.n_ranks_);
	}

	communicator();

	communicator(world_tag);

	explicit communicator(opaque_handle p);

	communicator(const communicator &) = delete;

	communicator(communicator &&other)
	: communicator() {
		swap(*this, other);
	}

	explicit operator bool ();

	communicator &operator= (communicator other) {
		swap(*this, other);
		return *this;
	}

	opaque_handle get_handle() {
		return p_;
	}

	int rank() {
		return rank_;
	}
	int n_ranks() {
		return n_ranks_;
	}

	bool is_rank_zero() {
		return !rank();
	}

	inline internal::barrier_icb barrier(no_root_tag) {
		return {this};
	}

	template<typename T>
	inline internal::bcast_root_icb bcast(this_root_tag, const T *in, int n) {
		return {this, get_type<T>(), n, in};
	}

	template<typename T>
	inline internal::bcast_root_icb bcast(this_root_tag, const T *in) {
		return {this, get_type<T>(), 1, in};
	}

	template<typename T>
	inline internal::bcast_nonroot_icb bcast(other_root_tag root, T *out, int n) {
		return {this, get_type<T>(), root.rk, n, out};
	}

	template<typename T>
	inline internal::bcast_nonroot_icb bcast(other_root_tag root, T *out) {
		return {this, get_type<T>(), root.rk, 1, out};
	}

	template<typename T>
	inline internal::gather_root_icb gather(this_root_tag, const T *in, int n, T *out) {
		return {this, get_type<T>(), n, in, out};
	}

	template<typename T>
	inline internal::gather_root_icb gather(this_root_tag, const T *in, T *out) {
		return {this, get_type<T>(), 1, in, out};
	}

	template<typename T>
	inline internal::gather_nonroot_icb gather(other_root_tag root, const T *in, int n) {
		return {this, get_type<T>(), root.rk, n, in};
	}

	template<typename T>
	inline internal::gather_nonroot_icb gather(other_root_tag root, const T *in) {
		return {this, get_type<T>(), root.rk, 1, in};
	}

	template<typename T>
	inline internal::reduce_root_icb reduce(this_root_tag, const T *in, int n, T *out) {
		return {this, get_type<T>(), n, in, out};
	}

	template<typename T>
	inline internal::reduce_nonroot_icb reduce(other_root_tag root, const T *in, int n) {
		return {this, get_type<T>(), root.rk, n, in};
	}

	template<typename T>
	inline internal::all_reduce_icb all_reduce(const T *in, int n, T *out) {
		return {this, get_type<T>(), n, in, out};
	}

	communicator split_shared(no_root_tag);
	communicator split_color(no_root_tag, int color);
	communicator split_color(no_root_tag);

private:
	opaque_handle p_;
	int rank_;
	int n_ranks_;
};

//---------------------------------------------------------------------------------------
// Active RDMA.
//---------------------------------------------------------------------------------------

struct active_rdma_base {
	friend void swap(active_rdma_base &x, active_rdma_base &y) {
		std::swap(x.com_, y.com_);
		std::swap(x.p_, y.p_);
		std::swap(x.size_, y.size_);
		std::swap(x.mapping_, y.mapping_);
	}

	active_rdma_base();

	explicit active_rdma_base(no_root_tag, communicator &com, size_t unit, size_t size);

	active_rdma_base(const active_rdma_base &) = delete;

	active_rdma_base(active_rdma_base &&other)
	: active_rdma_base() {
		swap(*this, other);
	}

	~active_rdma_base();

	active_rdma_base &operator= (active_rdma_base other) {
		swap(*this, other);
		return *this;
	}

public:
	void pre_fence(no_root_tag);
	void fence(no_root_tag);
	void post_fence(no_root_tag);

	void dispose(no_root_tag);

protected:
	void *raw_data() {
		return mapping_;
	}

	void do_get_sync(opaque_handle dtype, int rk, void *out);
	void do_put_sync(opaque_handle dtype, int rk, const void *in);
	void do_accumulate_sync(opaque_handle dtype, int rk, const void *in);

private:
	communicator *com_;
	opaque_handle p_;
	size_t size_;
	void *mapping_;
};

template<typename T>
struct active_rdma_array : active_rdma_base {
	explicit active_rdma_array(no_root_tag, communicator &com, size_t size)
	: active_rdma_base{collective, com, sizeof(T), size} { }

	T *data() {
		return reinterpret_cast<T *>(raw_data());
	}

	void get_sync(int rk, T *out) {
		do_get_sync(get_type<T>(), rk, out);
	}

	void put_sync(int rk, const T *in) {
		do_put_sync(get_type<T>(), rk, in);
	}

	void accumulate_sync(int rk, const T *in) {
		do_accumulate_sync(get_type<T>(), rk, in);
	}
};

//---------------------------------------------------------------------------------------
// Passive RDMA.
//---------------------------------------------------------------------------------------

struct passive_rdma_base {
	friend struct passive_rdma_base_scope;

	friend void swap(passive_rdma_base &x, passive_rdma_base &y) {
		std::swap(x.com_, y.com_);
		std::swap(x.p_, y.p_);
		std::swap(x.size_, y.size_);
		std::swap(x.mapping_, y.mapping_);
	}

	passive_rdma_base();

	explicit passive_rdma_base(no_root_tag, communicator &com, size_t unit, size_t size);

	passive_rdma_base(const passive_rdma_base &) = delete;

	passive_rdma_base(passive_rdma_base &&other)
	: passive_rdma_base() {
		swap(*this, other);
	}

	~passive_rdma_base();

	passive_rdma_base &operator= (passive_rdma_base other) {
		swap(*this, other);
		return *this;
	}

public:
	void dispose(no_root_tag);

protected:
	void do_get_sync(opaque_handle dtype, int rk, void *out);
	void do_put_sync(opaque_handle dtype, int rk, const void *in);

private:
	communicator *com_;
	opaque_handle p_;
	size_t size_;
	void *mapping_;
};

template<typename T>
struct passive_rdma_array : passive_rdma_base {
	explicit passive_rdma_array(no_root_tag, communicator &com, size_t size)
	: passive_rdma_base{collective, com, sizeof(T), size} { }

/*
	T *data() {
		return reinterpret_cast<T *>(raw_data());
	}
*/

	void get_sync(int rk, T *out) {
		do_get_sync(get_type<T>(), rk, out);
	}

	void put_sync(int rk, const T *in) {
		do_put_sync(get_type<T>(), rk, in);
	}
};

struct passive_rdma_base_scope {
	passive_rdma_base_scope(passive_rdma_base &r);

	passive_rdma_base_scope(const passive_rdma_base_scope &) = delete;

	~passive_rdma_base_scope();

	passive_rdma_base_scope &operator= (const passive_rdma_base_scope &) = delete;

protected:
	passive_rdma_base *get_rdma() {
		return r_;
	}

protected:
	void *raw_data() {
		return r_->mapping_;
	}

private:
	passive_rdma_base *r_;
};

template<typename T>
struct passive_rdma_array_scope : passive_rdma_base_scope {
	passive_rdma_array_scope(passive_rdma_array<T> &r)
	: passive_rdma_base_scope{r} { }

	T *data() {
		return reinterpret_cast<T *>(raw_data());
	}
};

struct pollable {
private:
	enum stage {
		null,
		in_progress,
		complete
	};

public:
	pollable()
	: s_{stage::null} { }

	template<typename ICB>
	pollable(ICB b) {
		h_ = internal::initiate_nonblocking(b);
		s_ = stage::in_progress;
	}

	pollable(const pollable &) = delete;

	pollable &operator= (const pollable &) = delete;

	template<typename ICB>
	void issue(ICB b) {
		assert(s_ = stage::null);
		h_ = internal::initiate_nonblocking(b);
		s_ = stage::in_progress;
	}

	bool done() {
		return poll(h_);
	}

private:
	bool poll(opaque_handle h);

	stage s_ = stage::in_progress;
	opaque_handle h_;
};

template<typename ICB>
void post(ICB b) {
	internal::initiate_blocking(b);
}

} // namespace fabry
