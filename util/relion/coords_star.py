"""
    This file is part of the relion-external suite that allows integration of
    arbitrary software into Relion 3.1.

    Copyright (C) 2020 University of Leicester

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see www.gnu.org/licenses/gpl-3.0.html.

    Written by TJ Ragan (tj.ragan@leicester.ac.uk),
    Leicester Institute of Structural and Chemical Biology (LISCB)

"""


import os
import shutil

from .micrographs_star import read_micrographs_star
from .pick_star import read_pick_star
from .particles_star import Particle


RELION_STAR_PRINTTABLE = os.environ.get('RELION_EXTERNAL_PRINTTABLE', shutil.which('relion_star_printtable'))


# TODO: use default callable and add commandline parameter to specify
def read_coords_star(starfile_location):
    starfile_dirname = os.path.dirname(starfile_location)
    ext = starfile_location.split('coords_suffix')[1]
    with open(starfile_location) as f:
        lines = f.readlines()
    if len(lines) != 1:
        raise NotImplementedError('Only a single entry in the coord starfile supported.')
    original_mics_location = lines[0].strip()
    original_mics = read_micrographs_star((original_mics_location))
    coords = []
    for micrograph in original_mics['micrographs']:
        micrograph_dir, micrograph_name = os.path.split(micrograph.rlnMicrographName)
        pick_star_filename = os.path.join(starfile_dirname, os.path.basename(micrograph_dir), os.path.splitext(micrograph_name)[0]) + ext
        picks = read_pick_star(pick_star_filename)
        for pick in picks:
            # coords.append(Pick(rlnMicrographName=micrograph.rlnMicrographName,
            #                    rlnCoordinateX=pick.rlnCoordinateX,
            #                    rlnCoordinateY=pick.rlnCoordinateY,
            #                    rlnAutopickFigureOfMerit=pick.rlnAutopickFigureOfMerit,
            #                    rlnClassNumber=pick.rlnClassNumber,
            #                    rlnAnglePsi=pick.rlnAnglePsi))
            coords.append(Particle(rlnMicrographName=micrograph.rlnMicrographName,
                               rlnCoordinateX=pick.rlnCoordinateX,
                               rlnCoordinateY=pick.rlnCoordinateY,
                               rlnOpticsGroup=micrograph.rlnOpticsGroup,
                               rlnImageName=None))
    return {'optics groups': original_mics['optics groups'],
            'coords': coords}
