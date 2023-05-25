#pragma once

namespace fabry {

struct communicator;

enum class opaque_handle : int { };

namespace internal {

void initialize_fabric(int *argcp, char ***argvp);
void finalize_fabric();

} // namespace fabry::internal

} // namespace fabry
