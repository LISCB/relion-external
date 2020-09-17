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


import subprocess
import os
import tempfile
from collections import namedtuple
import shutil


RELION_STAR_PRINTTABLE = os.environ.get('RELION_EXTERNAL_PRINTTABLE', shutil.which('relion_star_printtable'))


OG_FIELDS = ['rlnOpticsGroupName', 'rlnOpticsGroup', 'rlnMicrographPixelSize', 'rlnVoltage',
             'rlnSphericalAberration', 'rlnAmplitudeContrast']
MICROGRAPH_FIELDS = ['rlnMicrographName',  'rlnOpticsGroup']

OpticsGroup = namedtuple('OpticsGroup', OG_FIELDS)
Micrograph = namedtuple('Micrograph', MICROGRAPH_FIELDS)


# TODO: use default callable and add commandline parameter to specify
def read_micrographs_star(starfile_location):
    with tempfile.TemporaryDirectory() as tmpdirname:
        curr_dir = os.getcwd()
        abs_starfile_location = os.path.abspath(starfile_location)
        ogs = {}
        micrographs = []
        try:
            os.chdir(tmpdirname)
            og_fields_txt = ' _'.join(OG_FIELDS)
            og_fields_txt = ' _' + og_fields_txt
            optics_proc = subprocess.run(
                f'{RELION_STAR_PRINTTABLE} {abs_starfile_location} data_optics {og_fields_txt}',
                check=True, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            for og in optics_proc.stdout.decode().splitlines():
                split_og = og.strip().split()
                ogs[split_og[0]] = OpticsGroup(rlnOpticsGroupName=split_og[0],
                                               rlnOpticsGroup=int(split_og[1]),
                                               rlnMicrographPixelSize=float(split_og[2]),
                                               rlnVoltage=float(split_og[3]),
                                               rlnSphericalAberration=float(split_og[4]),
                                               rlnAmplitudeContrast=float(split_og[5]),
                                               )

            micrograph_fields_txt = ' _'.join(MICROGRAPH_FIELDS)
            micrograph_fields_txt = ' _' + micrograph_fields_txt
            micrographs_proc = subprocess.run(
                f'relion_star_printtable {abs_starfile_location} data_micrographs {micrograph_fields_txt}',
                check=True, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            for m in micrographs_proc.stdout.decode().splitlines():
                split_m = m.strip().split()
                micrographs.append(Micrograph(rlnMicrographName=split_m[0], rlnOpticsGroup=int(split_m[1])))
        finally:
            os.chdir(curr_dir)
        return {'optics groups': ogs, 'micrographs': micrographs}
