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


from pathlib import Path
import os
from glob import glob

from util.relion import read_particles_star
from util.relion import read_micrographs_star, read_coords_star
from util.framework.job import RelionJob


class ParticlesTo(RelionJob):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parser.add_argument('--in_parts', metavar='STARFILE',
                                 help='Input particles .star file. DO NOT USE IF RUNNING FROM RELION.')
        self.parser.add_argument('--in_mics', metavar='STARFILE',
                                 help='Input micrographs .star file.  Must be paired with a coords star file. DO NOT USE IF RUNNING FROM RELION.')
        self.parser.add_argument('--in_coords', metavar='STARFILE',
                                 help='Input coords .star file. Must be paired with a micrographs star file. DO NOT USE IF RUNNING FROM RELION.')



    @staticmethod
    def get_micrographs_in_particles_starfile(starfile):
        micrographs = set()
        for p in starfile['particles']:
            micrographs.add(p.rlnMicrographName)
        return micrographs


    # def write_relion_output_nodes(self):
    #     with open(os.path.join(self.relion_job_dir, 'RELION_OUTPUT_NODES.star'), 'w') as f:
    #         f.write('data_output_nodes\n')
    #         f.write('loop_\n')
    #         f.write('_rlnPipeLineNodeName #1\n')
    #         f.write('_rlnPipeLineNodeType #2\n')
    #         f.write(f'None 20\n')


    def assemble_particles_starfile(self, in_parts=None, in_mics=None, in_coords=None):
        if in_parts:
            return read_particles_star(in_parts)
        elif self.args.in_parts:
            return read_particles_star(self.args.in_parts)
        else:
            from util.relion.particles_star import Particle
            coords_star_file = read_coords_star(self.args.in_coords)
            if self.args.in_mics:
                micrographs_star_file = read_micrographs_star(self.args.in_mics)
                ogs = micrographs_star_file['optics groups']
                micrographs = micrographs_star_file['micrographs']
                micrographs_to_ogs = {m.rlnMicrographName: m.rlnOpticsGroup for m in micrographs}
            else:
                ogs = coords_star_file['optics groups']
            particles = []
            for part in coords_star_file['coords']:
                if self.args.in_mics:
                    og = micrographs_to_ogs[part.rlnMicrographName]
                else:
                    og = part.rlnOpticsGroup
                particles.append(Particle(rlnMicrographName=part.rlnMicrographName,
                                          rlnOpticsGroup=og,
                                          rlnImageName=None,
                                          rlnCoordinateX=part.rlnCoordinateX,
                                          rlnCoordinateY=part.rlnCoordinateY))
            return {'optics groups': ogs,
                    'particles': particles}


    def run(self):
        self.initial_run()
        try:
            self.input_particles_starfile = self.assemble_particles_starfile()
            # print(self.input_particles_starfile['optics groups'])
            # print(self.input_particles_starfile['particles'])
            # import sys; sys.exit()
            working_top_dir = self.make_working_tree()
            expected_count, expected_preproc_count = self.worker_setup_function(self)

            worker_procs = set()
            self.worker_dirs = glob(os.path.join(self.working_top_dir, 'worker*'))
            for d in self.worker_dirs:
                try:
                    os.chdir(d)
                    env = os.environ.copy()
                    if self.gpu_ids:
                        if self.parallelizable:
                            env.update({'CUDA_VISIBLE_DEVICES': d[-1]})
                        else:
                            env.update({'CUDA_VISIBLE_DEVICES': ' '.join([str(i) for i in self.gpu_ids])})

                    proc = self.worker_run_function(self, env=env,
                                                    # parsed_args=self.args,
                                                    # total_workers=len(worker_dirs)
                                                    )
                    worker_procs.add(proc)
                finally:
                    os.chdir(self.top_dir)
            self.monitor_job_progress(worker_procs, total_count=expected_count,
                                      total_preproc_count=expected_preproc_count)

            if self.worker_output_file_converter_function:
                self.worker_output_file_converter_function(self)
            self.recover_output_files()

            if self.worker_output_analysis_function is not None:
                self.worker_output_analysis_function(self)
            if self.worker_cleanup_function is not None:
                self.worker_cleanup_function(self)

            self.write_relion_output_starfile()
            self.write_relion_output_nodes()

            Path(os.path.join(self.relion_job_dir, 'RELION_JOB_EXIT_SUCCESS')).touch()
            print(f' Done!\n')

        except Exception as e:
            if os.path.exists(os.path.join(self.relion_job_dir, 'RELION_JOB_ABORT_NOW')):
                os.remove(os.path.join(self.relion_job_dir, 'RELION_JOB_ABORT_NOW'))
                Path(os.path.join(self.relion_job_dir, 'RELION_JOB_EXIT_ABORTED')).touch()
                raise e
            else:
                Path('RELION_JOB_EXIT_FAILURE').touch()
                raise e

        except Exception as e:
            raise e
