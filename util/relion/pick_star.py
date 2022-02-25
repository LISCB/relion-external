"""
    This file is part of the relion-external suite that allows integration of
    arbitrary software into Relion 3.1.

    Copyright (C) 2020 TJ Ragan (tj.ragan@leicester.ac.uk)
    for The Leicester Institute of Structural and Chemical Biology (LISCB),
    Univeristy of Leicester

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

"""


import subprocess
import os
import tempfile
from collections import namedtuple
import shutil


RELION_STAR_PRINTTABLE = os.environ.get('RELION_EXTERNAL_PRINTTABLE', shutil.which('relion_star_printtable'))

PICKS_FIELDS = ['rlnMicrographName',
                'rlnCoordinateX', 'rlnCoordinateY',
                'rlnAutopickFigureOfMerit', 'rlnClassNumber', 'rlnAnglePsi']

Pick = namedtuple('Pick', PICKS_FIELDS)


def read_pick_star(starfile_location):
    with tempfile.TemporaryDirectory() as tmpdirname:
        curr_dir = os.getcwd()
        abs_starfile_location = os.path.abspath(starfile_location)
        picks = []
        try:
            os.chdir(tmpdirname)
            pick_fields_txt = ' _'.join(PICKS_FIELDS)
            pick_fields_txt = ' _' + pick_fields_txt
            picks_proc = subprocess.run(
                f'relion_star_printtable {abs_starfile_location} data_ {pick_fields_txt}',
                check=True, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            for p in picks_proc.stdout.decode().splitlines():
                split_p = p.strip().split()
                picks.append(Pick(rlnMicrographName=None,
                                  rlnCoordinateX=split_p[0], rlnCoordinateY=int(float(split_p[1])),
                                  rlnAutopickFigureOfMerit=float(split_p[2]),
                                  rlnClassNumber=float(split_p[3]),
                                  rlnAnglePsi=float(split_p[4])))
        finally:
            os.chdir(curr_dir)
        return picks
