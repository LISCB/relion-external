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
import time
from collections import defaultdict
import argparse
import os
from glob import glob
import shutil
from subprocess import check_output, CalledProcessError, PIPE
import signal


MOUSE = r'~~( ϵ:>'
CHEESE = r'[oo]'
VERSION = 0.5

class RelionJob:

    def __init__(self, name,
                 extra_text='** NON-COMMERCIAL USE ONLY. **',
                 worker_setup_function=None,
                 worker_run_function=None,
                 worker_output_file_converter_function=None,
                 worker_output_analysis_function=None,
                 worker_cleanup_function=None,
                 preproc_dir=False,
                 parallelizable=True,
                 output_star_template=None,):
        self.name = name
        self.extra_text = extra_text
        self.worker_setup_function = worker_setup_function
        self.worker_run_function = worker_run_function
        self.worker_output_file_converter_function=worker_output_file_converter_function
        self.worker_output_analysis_function = worker_output_analysis_function
        self.worker_cleanup_function = worker_cleanup_function or self.cleanup
        self.preproc_dir = preproc_dir
        self.parallelizable = parallelizable
        self.num_workers = 1
        self.args = None
        self._out_ext = None
        if output_star_template:
            self.output_star_name = output_star_template.format(self.name)
        else:
            self.output_star_name = None
        self._relion_output_node_int = -1

        self.working_top_dir = None
        self.total_count = 0
        self.start_time = None
        self.current_count = 0
        self.current_time = None
        self.top_dir = os.getcwd()
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--o', metavar='DIRECTORY', required=True,
                                 help='Job output directory. DO NOT USE IF RUNNING FROM RELION.')
        self.parser.add_argument('--j', metavar="NUM_CPU", type=int, default=len(os.sched_getaffinity(0)),
                                 help="Total threads. (Default: All available cores.)")

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


    @staticmethod
    def get_gpu_info():
        try:
            HW_GPU_LIST = subprocess.run('lspci | grep "VGA compatible controller: NVIDIA"', shell=True,
                                         stdout=subprocess.PIPE).stdout.decode().splitlines()
            CUDA_VISIBLE_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES')
            if CUDA_VISIBLE_DEVICES:
                CUDA_VISIBLE_DEVICE_IDS = [int(g) for g in CUDA_VISIBLE_DEVICES.split(',')]
                gpu_ids = CUDA_VISIBLE_DEVICE_IDS
            else:
                gpu_ids = list(range(len(HW_GPU_LIST)))
            return gpu_ids
        except:
            return []


    @staticmethod
    def relion_path_mapping(input_micrographs_starfile, relion_job_dir=None,
                            output_ext='',
                            reduce=False, output=True):
        '''
        reduce should be False, None, or extension of file to check
        '''
        mapping = defaultdict(list)
        for micrograph in input_micrographs_starfile['micrographs']:
            input_dir = os.path.dirname(micrograph.rlnMicrographName)
            output_dir = os.path.basename(input_dir)
            file_name = os.path.basename(micrograph.rlnMicrographName)
            if relion_job_dir and reduce:
                target_path = os.path.join(relion_job_dir, output_dir,
                                           os.path.splitext(file_name)[0]+output_ext)
                if os.path.exists(target_path):
                    continue
            if output is True:
                mapping[output_dir].append(file_name)
            else:
                mapping[input_dir].append(file_name)
        return mapping


    @staticmethod
    def _flatten_output_mapping(output_mapping):
        flattened = []
        for k, v in output_mapping.items():
            for m in v:
                flattened.append(os.path.join(k, m))
        return flattened


    @staticmethod
    def _find_default_input_to_output_mapping(working_top_dir, out_ext='.mrc'):
        io_mapping = {}
        worker_dirs = glob(os.path.join(working_top_dir, 'worker_*'))
        for d in worker_dirs:
            worker_dir_name = os.path.basename(d)
            for input_micrograph_directory in glob(os.path.join(d, 'input', '*')):
                for input_micrograph_path in glob(
                        os.path.join(input_micrograph_directory, '*.mrc')):
                    output_file = os.path.join(os.path.basename(working_top_dir), worker_dir_name, 'output',
                                               os.path.relpath(input_micrograph_path, os.path.join(d, 'input')))
                    output_file = os.path.splitext(output_file)[0] + out_ext
                    io_mapping[os.path.relpath(input_micrograph_path, os.path.dirname(working_top_dir))] = output_file
        return io_mapping


    @staticmethod
    def make_progress_bar(start_time=None, total_count=0, current_count=0, current_time=None):
        TOTAL_DOTS = 62
        end = ''
        dots = '' * TOTAL_DOTS
        spaces = ' ' * (TOTAL_DOTS + len(MOUSE))
        if total_count < 1:  # No idea, just make a holding bar
            return f' ???/??? ??? {spaces}{CHEESE}'
        elif start_time:  # We have a total count and a start time, now we can start filling in the times.
            current_time = current_time or time.time()
            progress_time = current_time - start_time
            estimated_time_text = '???'
            if current_count:
                estimated_time = int((current_time - start_time) / (current_count / total_count))
                if estimated_time < 60:
                    units = 'sec'
                    estimated_time_text = f'{estimated_time:3d}'
                    progress_time_text = f'{int(progress_time):3d}'
                elif estimated_time < 3600:
                    units = 'min'
                    estimated_time_text = f'{estimated_time / 60:.3f}'[:3]
                    progress_time_text = f'{progress_time / 60:.3f}'[:3]
                else:
                    units = 'hrs'
                    estimated_time_text = f'{estimated_time / 3600:.1f}'
                    progress_time_text = f'{progress_time / 3600:.1f}'
                dot_count = int(round((TOTAL_DOTS + len(CHEESE)) * current_count / total_count))
                dots = '.' * dot_count
            else:
                if progress_time < 60:
                    units = 'sec'
                    progress_time_text = f'{int(progress_time):3d}'
                elif progress_time < 3600:
                    units = 'min'
                    progress_time_text = f'{progress_time / 60:.3f}'[:3]
                else:
                    units = 'hrs'
                    progress_time_text = f'{progress_time / 3600:.1f}'
            return f'\r {progress_time_text}/{estimated_time_text} {units} {dots}{MOUSE}'


    def count_done_preproc_items(self, job_object, preproc_ext='.mrc'):
        preproc_dirs = os.path.join(self.working_top_dir, 'worker*', 'preproc')
        try:
            find_output = check_output(f'find -L {preproc_dirs} -name "*{preproc_ext}"', shell=True,
                                       stderr=PIPE, universal_newlines=True).splitlines()
            return len(find_output)
        except CalledProcessError:
            return 0
    # def count_done_preproc_items(self, job_object):
    #     try:
    #         preproc_items = check_output(f'find {os.path.join(job_object.working_top_dir, "worker*", "preproc")} -name "*.mrc"',
    #                                  stderr=PIPE, shell=True).splitlines()
    #         return len(preproc_items)
    #     except CalledProcessError:
    #         return 0


    def _parse_args(self):
        if self.args is None:
            self.parser.add_argument('--cache', type=str, default='',
                                     help="Cache directory, typically /ssd/$SLURM_JOB_ID. (Default: No cache.)")
            self.parser.add_argument('--keep_preproc', action='store_false',
                                     help="Retain intermediate preprocessing files. (Default: False.)")
            self.args = self.parser.parse_args()
            self.relion_job_dir = self.args.o


    # def _monitor_sub_job_progress(self, slaves, total_count,
    #                               start_time=None, previous_done_count=None,
    #                               preproc=False):
    #     return {}
    def _monitor_sub_job_progress(self, slaves, total_count,
                                  start_time=None, previous_done_count=None,
                                  preproc=False):
        if previous_done_count is None:
            print(self.make_progress_bar(), flush=True, end='\r')
            previous_done_count = 0
        for i, proc in enumerate(slaves):
            poll = proc.poll()
            if (poll is not None) and (poll > 0):
                raise Exception('External job failed. Exiting.')
        abort = self.check_abort_signal()
        if abort:
            for proc in slaves:
                os.kill(proc.pid, signal.SIGTERM)
            raise Exception('Relion RELION_JOB_ABORT_NOW file seen. Terminating.')

        if preproc:
            done_count = self.count_done_preproc_items(self)
        else:
            done_count = self.count_done_output_items(self)

        if done_count > previous_done_count:
            print(self.make_progress_bar(start_time, total_count, done_count), flush=True, end='')
        elif done_count == 0:
            start_time = start_time or time.time()
            print(self.make_progress_bar(start_time, total_count, done_count), flush=True, end='')
        return {'start_time': start_time, 'previous_done_count': done_count}


    def monitor_job_progress(self, slaves, total_count, total_preproc_count=None):
        slaves = [proc for proc in slaves if proc is not None]
        if len(slaves) == 0:
            return
        total_preproc_count = total_preproc_count or total_count
        if self.preproc_dir:
            print('Preprocessing:', flush=True)
            monitor_status = {}
            while any([proc.poll() is None for proc in slaves]):
                monitor_status = self._monitor_sub_job_progress(slaves=slaves,
                                                                total_count=total_preproc_count,
                                                                preproc=True,
                                                                **monitor_status)
                if monitor_status.get('previous_done_count', 0) == total_preproc_count:
                    break
                time.sleep(1)
            print('\nProcessing:', flush=True)

        monitor_status = {}
        while any([proc.poll() is None for proc in slaves]):
            monitor_status = self._monitor_sub_job_progress(slaves=slaves,
                                                            total_count=total_count,
                                                            **monitor_status)
            time.sleep(1)

        print()


    def make_working_tree(self):
        i = 2
        working_top_dir = os.path.join(self.relion_job_dir, self.name)
        while os.path.exists(working_top_dir):
            working_top_dir = os.path.join(self.relion_job_dir, self.name) + f'-{i}'
            i += 1
        self.working_top_dir = working_top_dir
        for n in range(self.num_workers):
            worker_dir = os.path.join(working_top_dir, f'worker_{n}')
            input_dir_path = os.path.join(worker_dir, 'input')
            os.makedirs(input_dir_path, exist_ok=True)
            os.makedirs(os.path.join(worker_dir, 'output'), exist_ok=True)
        return working_top_dir


    def cleanup(self, job_object):
        if job_object.preproc_dir and not job_object.args.keep_preproc:
            for d in glob(os.path.join(job_object.working_top_dir, 'worker*')):
                try:
                    shutil.rmtree(os.path.join(d, 'preproc'))
                except FileNotFoundError:
                    pass


    def check_abort_signal(self):
        if os.path.exists(os.path.join(self.relion_job_dir, 'RELION_JOB_ABORT_NOW')):
            return True
        return False


    def print_header_text(self):
        print('LISCB External program wrapper for Relion 3.1.', flush=True)
        print('(Leicester Institute of Structural and Chemical Biology)\n', flush=True)
        print('Copyright (C) 2020 University of Leicester\n', flush=True)
        print('This program comes with ABSOLUTELY NO WARRANTY; for details go to www.gnu.org/licenses/gpl-3.0.html', flush=True)
        print('This is free software, and you are welcome to redistribute it', flush=True)
        print('under certain conditions; go to www.gnu.org/licenses/gpl-3.0.html for details.\n', flush=True)
        print(f'Wrapper Version: {VERSION}\n', flush=True)

        if callable(self.extra_text):
            print(self.extra_text(self), flush=True)
        else:
            print(self.extra_text, flush=True)

        # print(f'Print compute info here.', flush=True)
        # print(f"Using {GPU_COUNT} GPUs: {GPU_IDS}.", flush=True)
        print(f'=================', flush=True)


    def write_relion_output_nodes(self):
        if self._relion_output_node_int >= 0:
            with open(os.path.join(self.relion_job_dir, 'RELION_OUTPUT_NODES.star'), 'w') as f:
                f.write('data_output_nodes\n')
                f.write('loop_\n')
                f.write('_rlnPipeLineNodeName #1\n')
                f.write('_rlnPipeLineNodeType #2\n')
                f.write(f'{os.path.join(self.relion_job_dir, self.output_star_name)}    {self._relion_output_node_int}\n')


    def write_relion_output_starfile(self):
        pass

    def recover_output_files(self):
        pass

    def initial_run(self):
        self._parse_args()
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

        for f in ('RELION_JOB_EXIT_FAILURE', 'RELION_JOB_EXIT_SUCCESS'):
            try: os.remove(os.path.join(self.args.o, f))
            except FileNotFoundError: pass

        self.print_header_text()
        self.gpu_ids = self.get_gpu_info()

        if self.parallelizable:
            self.num_workers = max(1, len(self.gpu_ids))

        # working_top_dir = self.make_working_tree()
