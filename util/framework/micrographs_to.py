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
import shutil

from util.relion import read_micrographs_star
from util.framework.job import RelionJob


class MicrographsTo(RelionJob):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parser.add_argument('--in_mics', metavar='STARFILE', required=True,
                                 help='Input micrographs .star file. DO NOT USE IF RUNNING FROM RELION.')
        self.io_mappings = None


    def make_output_tree(self):
        micrograph_source_dirs = self.relion_path_mapping(self.input_micrographs_starfile).keys()
        for d in micrograph_source_dirs:
            os.makedirs(os.path.join(self.relion_job_dir, d), exist_ok=True)


    def count_done_output_items(self, job_object):
        potential_done_items = self.io_mappings.values()
        done_count = 0
        for potential_micrograph in potential_done_items:
            pth = os.path.join(self.relion_job_dir, potential_micrograph)
            if os.path.exists(pth):
                done_count += 1
        return done_count


    # def count_done_preproc_items(self):
    #     potential_micrographs = [m.replace('/input/', '/preproc/') for m in self.io_mappings.keys()]
    #     done_count = 0
    #     for potential_micrograph in potential_micrographs:
    #         pth = os.path.join(self.relion_job_dir, potential_micrograph)
    #         print(f'preproc count: checking for {pth}')
    #         if os.path.exists(pth):
    #             done_count += 1
    #     return done_count


    # def _monitor_sub_job_progress(self, slaves, total_count,
    #                               start_time=None, previous_done_count=None,
    #                               preproc=False):
    #     if previous_done_count is None:
    #         print(self.make_progress_bar(), flush=True, end='\r')
    #         previous_done_count = 0
    #     for i, proc in enumerate(slaves):
    #         poll = proc.poll()
    #         if (poll is not None) and (poll > 0):
    #             raise Exception('External job failed. Exiting.')
    #     abort = self.check_abort_signal()
    #     if abort:
    #         for proc in slaves:
    #             os.kill(proc.pid, signal.SIGTERM)
    #         raise Exception('Relion RELION_JOB_ABORT_NOW file seen. Terminating.')
    #
    #     if preproc:
    #         done_count = self.count_done_preproc_items()
    #     else:
    #         done_count = self.count_done_output_items()
    #
    #     if done_count > previous_done_count:
    #         print(self.make_progress_bar(start_time, total_count, done_count), flush=True, end='')
    #     elif done_count == 0:
    #         start_time = start_time or time.time()
    #         print(self.make_progress_bar(start_time, total_count, done_count), flush=True, end='')
    #     return {'start_time': start_time, 'previous_done_count': done_count}


    # def monitor_job_progress(self, slaves):
    #     if self.preproc_dir:
    #         print('Preprocessing:', flush=True)
    #         monitor_status = {}
    #         while any([proc.poll() is None for proc in slaves]):
    #             monitor_status = self._monitor_sub_job_progress(slaves=slaves,
    #                                                             preproc=True,
    #                                                             **monitor_status)
    #             if monitor_status.get('previous_done_count', 0) == len(self.io_mappings):
    #                 break
    #             time.sleep(1)
    #         print('\nProcessing:', flush=True)
    #     monitor_status = {}
    #     while any([proc.poll() is None for proc in slaves]):
    #         monitor_status = self._monitor_sub_job_progress(slaves=slaves,
    #                                                         **monitor_status)
    #         time.sleep(1)
    #     print()


    def recover_output_files(self):
        if self.io_mappings is None:
            io_mappings = {}
            for working_dir in sorted(glob(os.path.join(self.relion_job_dir, self.name+'*')), key=os.path.getmtime):
                io_mappings.update(self._find_default_input_to_output_mapping(working_dir, out_ext=self._out_ext))
        else:
            io_mappings = self.io_mappings

        for input, output in io_mappings.items():
            input_output_target = output.split('/output/')[1]  # HACK!
            relion_output_target = os.path.join(self.relion_job_dir, input_output_target)
            try:
                if os.path.exists(os.path.join(self.relion_job_dir, output)):
                    os.symlink(os.path.join('..', output), relion_output_target)
            except FileExistsError:
                pass


    def make_and_populate_working_tree(self):
        reduced_input_micrographs_starfile = self.relion_path_mapping(self.input_micrographs_starfile,
                                                                     self.relion_job_dir,
                                                                     output_ext=self._out_ext,
                                                                     reduce=True,
                                                                     output=False)
        reduced_micrographs_to_symlink = self._flatten_output_mapping(reduced_input_micrographs_starfile)
        micrograph_count = len(reduced_micrographs_to_symlink)
        if micrograph_count == 0:
            return None
        working_top_dir = self.make_working_tree()
        for n in range(min(self.num_workers, micrograph_count)):
            self._populate_worker_dir(reduced_micrographs_to_symlink, worker_number=n)
        return working_top_dir


    def _populate_worker_dir(self, reduced_micrographs_to_symlink, worker_number=0):
        worker_dir = os.path.join(self.working_top_dir, f'worker_{worker_number}')
        micrographs_to_symlink = reduced_micrographs_to_symlink[worker_number::self.num_workers]

        input_dir_path = os.path.join(worker_dir, 'input')
        preproc_dir_path = os.path.join(worker_dir, 'preproc')
        # os.makedirs(input_dir_path, exist_ok=True)
        # os.makedirs(os.path.join(worker_dir, 'output'), exist_ok=True)

        micrographs_to_symlink_abspath = [os.path.abspath(m) for m in micrographs_to_symlink]
        for micrograph_to_symlink_abspath in micrographs_to_symlink_abspath:
            input_subpath = os.path.basename(os.path.dirname(micrograph_to_symlink_abspath))
            full_input_subpath = os.path.join(input_dir_path, input_subpath)
            os.makedirs(full_input_subpath, exist_ok=True)
            if self.preproc_dir:
                # os.makedirs(preproc_dir_path, exist_ok=True)
                if self.args.cache:
                    src = os.path.join(self.args.cache, preproc_dir_path)
                    shutil.rmtree(src, ignore_errors=True)
                    os.makedirs(src, exist_ok=True)
                    try:
                        os.symlink(src, preproc_dir_path)
                    except FileExistsError:
                        pass
                os.makedirs(os.path.join(preproc_dir_path, input_subpath), exist_ok=True)
            try:
                os.symlink(os.path.relpath(micrograph_to_symlink_abspath, full_input_subpath),
                           os.path.join(full_input_subpath,
                                        os.path.basename(micrograph_to_symlink_abspath)))
            except FileExistsError:
                pass


    def run(self):
        self.initial_run()
        try:
            self.input_micrographs_starfile = read_micrographs_star(self.args.in_mics)
            self.make_output_tree()
            # self.working_top_dir = self.make_and_populate_working_tree()
            self.make_and_populate_working_tree()

            try:
                if self.working_top_dir:
                    io_mappings = {}
                    if self.worker_setup_function:
                        current_dir = os.getcwd()
                        for d in glob(os.path.join(self.working_top_dir, 'worker_*')):
                            try:
                                worker_relpath = os.path.relpath(d, self.relion_job_dir)
                                os.chdir(d)
                                local_io_mapping = self.worker_setup_function(self)  # in_mics_abspath = os.path.abspath(self.args.in_mics)
                                if local_io_mapping is not None:
                                    io_mappings.update(
                                        {os.path.join(worker_relpath, k): os.path.join(worker_relpath, v)
                                         for k, v in local_io_mapping.items()})
                            finally:
                                os.chdir(current_dir)
                    self.io_mappings = io_mappings or self._find_default_input_to_output_mapping(self.working_top_dir,
                                                                                           out_ext=self._out_ext)
                    worker_procs = set()
                    current_dir = os.getcwd()
                    self.worker_dirs = glob(os.path.join(self.working_top_dir, 'worker*'))
                    for d in self.worker_dirs:
                        try:
                            os.chdir(d)
                            env = os.environ.copy()
                            if self.gpu_ids:
                                env.update({'CUDA_VISIBLE_DEVICES': d[-1]})
                            proc = self.worker_run_function(self, env=env,
                                                            # parsed_args=self.args,
                                                            # total_workers=len(worker_dirs)
                                                            )
                            worker_procs.add(proc)
                        finally:
                            os.chdir(current_dir)
                    self.monitor_job_progress(worker_procs, total_count=len(self.io_mappings))
            finally:
                if self.worker_output_file_converter_function:
                    self.worker_output_file_converter_function(self)
                self.recover_output_files()

            if self.worker_output_analysis_function is not None:
                self.worker_output_analysis_function(self)
            if self.worker_cleanup_function is not None:
                try:
                    self.worker_cleanup_function(self)
                except:
                    pass

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
