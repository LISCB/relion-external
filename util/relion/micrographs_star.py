import subprocess
import os
import tempfile
from collections import namedtuple


RELION_STAR_PRINTTABLE = '/net/prog/relion3/2085051-icc-cpu-opt/bin/relion_star_printtable'

OG_FIELDS = ['rlnOpticsGroupName', 'rlnOpticsGroup', 'rlnMicrographPixelSize', 'rlnVoltage',
             'rlnSphericalAberration']
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
                                               rlnSphericalAberration=float(split_og[4])
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