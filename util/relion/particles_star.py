import subprocess
import os
import tempfile
from collections import namedtuple


RELION_STAR_PRINTTABLE = '/net/prog/relion3/2085051-icc-cpu-opt/bin/relion_star_printtable'

OG_FIELDS = ['rlnOpticsGroupName', 'rlnOpticsGroup', 'rlnMicrographOriginalPixelSize',
             'rlnVoltage', 'rlnSphericalAberration', 'rlnAmplitudeContrast',
             'rlnImagePixelSize', 'rlnImageSize', 'rlnImageDimensionality']
PARTICLES_FIELDS = ['rlnMicrographName',  'rlnOpticsGroup',
                    'rlnImageName', 'rlnCoordinateX', 'rlnCoordinateY']

OpticsGroup = namedtuple('OpticsGroup', OG_FIELDS)
Particle = namedtuple('Particle', PARTICLES_FIELDS)


def read_particles_star(starfile_location):
    with tempfile.TemporaryDirectory() as tmpdirname:
        curr_dir = os.getcwd()
        abs_starfile_location = os.path.abspath(starfile_location)
        ogs = {}
        particles = []
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
                                               rlnMicrographOriginalPixelSize=float(split_og[2]),
                                               rlnVoltage=float(split_og[3]),
                                               rlnSphericalAberration=float(split_og[4]),
                                               rlnAmplitudeContrast=float(split_og[5]),
                                               rlnImagePixelSize=float(split_og[5]),
                                               rlnImageSize=float(split_og[5]),
                                               rlnImageDimensionality=float(split_og[8]),
                                               )

            particle_fields_txt = ' _'.join(PARTICLES_FIELDS)
            particle_fields_txt = ' _' + particle_fields_txt
            particles_proc = subprocess.run(
                f'relion_star_printtable {abs_starfile_location} data_particles {particle_fields_txt}',
                check=True, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            for p in particles_proc.stdout.decode().splitlines():
                split_p = p.strip().split()
                particles.append(Particle(rlnMicrographName=split_p[0], rlnOpticsGroup=int(split_p[1]),
                                          rlnImageName=split_p[2],
                                          rlnCoordinateX=float(split_p[3]), rlnCoordinateY=float(split_p[4])))
        finally:
            os.chdir(curr_dir)
        return {'optics groups': ogs, 'particles': particles}