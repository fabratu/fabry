#pragma once

#include <fabry/internal-base.hpp>

namespace fabry::internal {

struct barrier_icb {
	communicator *com;
};

void initiate_blocking(barrier_icb &b);
opaque_handle initiate_nonblocking(barrier_icb &b);

struct bcast_root_icb {
	communicator *com;
	opaque_handle dtype;
	int n;
	const void *in;
};

struct bcast_nonroot_icb {
	communicator *com;
	opaque_handle dtype;
	int rrk;
	int n;
	void *out;
};

void initiate_blocking(bcast_root_icb &b);
void initiate_blocking(bcast_nonroot_icb &b);
opaque_handle initiate_nonblocking(bcast_root_icb &b);
opaque_handle initiate_nonblocking(bcast_nonroot_icb &b);

struct gather_root_icb {
	communicator *com;
	opaque_handle dtype;
	int n;
	const void *in;
	void *out;
};

struct gather_nonroot_icb {
	communicator *com;
	opaque_handle dtype;
	int rrk;
	int n;
	const void *in;
};

void initiate_blocking(gather_root_icb &b);
void initiate_blocking(gather_nonroot_icb &b);

struct reduce_root_icb {
	communicator *com;
	opaque_handle dtype;
	int n;
	const void *in;
	void *out;
};

struct reduce_nonroot_icb {
	communicator *com;
	opaque_handle dtype;
	int rrk;
	int n;
	const void *in;
};

opaque_handle initiate_nonblocking(reduce_root_icb &b);
opaque_handle initiate_nonblocking(reduce_nonroot_icb &b);

} // namespace fabry::internal
