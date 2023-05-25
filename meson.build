project('fabry', 'c', 'cpp')

pkgconfig = import('pkgconfig')

mpi_dep = dependency('mpi', language: 'c')

install_headers(
	[
		'include/fabry/fabry.hpp',
		'include/fabry/internal-base.hpp',
		'include/fabry/internal-icb.hpp',
	],
	subdir: 'fabry'
)

incl = include_directories('include')

fabry_lib = library('fabry',
	'lib/fabry.cpp',
	include_directories: incl,
	dependencies: mpi_dep,
	install: true
)

fabry_dep = declare_dependency(
	include_directories: incl,
	link_with: fabry_lib
)

pkgconfig.generate(fabry_lib)
