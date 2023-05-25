#include <iostream>

#include <fabry/fabry.hpp>

int main(int argc, char **argv) {
	fabry::program program{&argc, &argv};
	fabry::communicator world{fabry::world};

	fabry::post(world.barrier(fabry::collective));

	int x = world.rank();
	if(world.is_rank_zero()) {
		int out[2];
		fabry::post(world.gather(fabry::this_root, &x, out));
		for(int i = 0; i < 2; i++)
			std::cout << "There is a rank " << out[i] << std::endl;
	}else{
		fabry::post(world.gather(fabry::zero_root, &x));
	}

	// Try out some RDMA.
	{
		fabry::passive_rdma_array<int> win{fabry::collective, world, 1};
		//win.get_sync(rk, &rk);
		*win.data() = world.rank();
		if(world.is_rank_zero()) {
			int r1;
			win.get_sync(1, &r1);
			std::cout << "rank 1's value is " << r1 << std::endl;
		}
		win.dispose(fabry::collective);
	}
}
