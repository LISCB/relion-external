import time
import signal
from pathlib import Path
import os
from glob import glob

from util.relion import read_star
from util.framework.job import RelionJob


class ParticlesTo(RelionJob):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parser.add_argument('--in_parts', metavar='STARFILE', required=True,
                                 help='Input particles .star file. DO NOT USE IF RUNNING FROM RELION.')


    def recover_output_files(self):
        pass


    def write_relion_output(self):
        pass


    @staticmethod
    def get_micrographs_in_particles_starfile(starfile):
        micrographs = set()
        for p in starfile['particles']:
            micrographs.add(p.rlnMicrographName)
        return micrographs


    def write_relion_output(self):
        with open(os.path.join(self.relion_job_dir, 'RELION_OUTPUT_NODES.star'), 'w') as f:
            f.write('data_output_nodes\n')
            f.write('loop_\n')
            f.write('_rlnPipeLineNodeName #1\n')
            f.write('_rlnPipeLineNodeType #2\n')
            f.write(f'None 20\n')


    def run(self):
        self.initial_run()
        try:
            self.input_particles_starfile = read_star(self.args.in_parts)
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
            self.write_relion_output()

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
