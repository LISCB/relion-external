import os
from .micrographs_star import read_micrographs_star
from .particles_star import read_particles_star
from .coords_star import read_coords_star

def read_star(starfile_location):
    basename = os.path.basename(starfile_location)
    if basename.startswith('micrographs'):
        return read_micrographs_star(starfile_location)
    elif basename.startswith('particles'):
        return read_particles_star(starfile_location)
    else:
        raise NotImplementedError('Automatic detection currently limited to micrograph_*.star files.')