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


# TODO: make downscaling params more sensible/settable

import os
from subprocess import Popen, check_output, PIPE
from collections import defaultdict
from glob import glob
import shutil
import sys

from util.framework.micrographs2starfiles import Micrographs2Starfiles


GPU_PYTHON = os.environ.get('TOPAZ_GPU_PYTHON', os.environ.get('TOPAZ_PYTHON', shutil.which('python3')))
TOPAZ_EXE = os.environ.get('TOPAZ_EXECUTABLE', shutil.which('topaz'))
TOPAZ = ' '.join((GPU_PYTHON, TOPAZ_EXE))

class Topaz_Picker(Micrographs2Starfiles):
    '''
    This is an example of how to setup a job via subclassing.
    Note that setup is now in the __init__() method, and we augment functionality by overriding methods in the superclass,
    and then use super() to call them...
    '''

    OUTPUT_FILENAME = 'picks.topaz'
    RECEPTIVE_FIELD_SIZES = {'resnet16': 91,
                             'resnet8': 71,
                             'conv127': 127,
                             'conv63': 6,
                             'conv31': 31}

    def __init__(self):
        super().__init__(name='topaz-pick',
                         extra_text=self.derive_extra_text,
                         worker_run_function=self.run_worker,
                         worker_output_file_converter_function=self.convert_output_files,
                         worker_output_analysis_function=self.worker_output_analysis_function,
                         worker_cleanup_function=self.cleanup,
                         preproc_dir=True,
                         parallelizable=True)

        preproc_args = self.parser.add_argument_group('Preprocessing Parameters')
        preproc_args.add_argument('-s', '--scale', default=8, type=int,
                                  help='downsample images by this factor.  See topaz webpage for appropriate choice (default: 8)')
        preproc_args.add_argument('--affine', action='store_true',
                                  help='use standard normalization (x-mu)/std of whole image rather than GMM normalization')
        # preproc_args.add_argument('--keep_preproc', action='store_true',
        #                        help='Keep the pre-processed micrographs (Default: Off.)')

        pick_args = self.parser.add_argument_group('Picking Parameters')
        pick_args.add_argument('-r', '--radius', default=14, type=int,
                               help='radius of the regions to extract in downsampled pixels. (Default: 14)')
        #TODO: add an exclusive with radius minimum inter particle distance in Å
        pick_args.add_argument('-t', '--threshold', default=0, type=float,
                               help='log-likelihood score threshold at which to terminate region extraction, -6 is p>=0.0025.  Lower numbers mean more picking. (Default: 0)')
        pick_args.add_argument('-m', '--model', default='resnet16',
                               help='path to trained subimage classifier, or pretrained network name.' \
                                    ' Available pretrained networks are: resnet16, resnet8, conv127, conv63, conv31.  (Default: pretrained resnet16)')
        pick_args.add_argument('--batch-size', default=4, type=int,
                               help='batch size for scoring micrographs with model. (Default: 4)')


    def _get_model_receptive_field(self):
        try:
            output = check_output(f'{TOPAZ} train -m {self.args.model} --describe 2>&1',
                                  universal_newlines=True, shell=True, stderr=PIPE).splitlines()
            for line in output:
                if 'Receptive field' in line:
                    receptive_field = line.split('# Receptive field:')[-1].strip()
                    return int(receptive_field)
        finally:
            return None


    def derive_extra_text(self, job_object):
        # TODO: can we use the --describe method to get the receptive field?
        text = []
        text.append('Topaz Picker v0.2.4')
        text.append('Commercial and non-commercial use allowed.')
        text.append('Citation:')
        text.append('Bepler, T., Morin, A., Brasch, J., Shapiro, L., Noble, A.J., Berger, B. (2019).')
        text.append('Positive-unlabeled convolutional neural networks for particle picking in cryo-electron micrographs. Nature Methods.')
        text.append('https://doi.org/10.1038/s41592-019-0575-8')
        text.append('\n*** Note that Topaz only outputs once finished, so the picking progress bar is not very useful.')
        text.append('*** You can estimate the Processing time required as: 10 sec + ( (Number of Micrographs / Number of GPUs) ) * 2 sec')
        # TODO: convert this to Å
        model = job_object.args.model


        if model not in ['resnet16', 'resnet8', 'conv127', 'conv63', 'conv31'] and not os.path.isfile(model):
            # Path('RELION_JOB_EXIT_FAILURE').touch()
            raise (FileNotFoundError(f'The model is not one of the pre-bundled models and {model} was not found.'))

        receptive_field = self.RECEPTIVE_FIELD_SIZES.get(model, self._get_model_receptive_field())
        if receptive_field:
            text.append(f'\nMaximum particle pick size: {receptive_field * job_object.args.scale} pixels.')
        else:
            text.append('\nUnknown model receptive field; cannot determine largest pick size.')
        return '\n'.join(text)


    def count_done_output_items(self, job_object):
        outputs = defaultdict(list)
        for io_mapping in self.io_mappings.values():
            d = os.path.dirname(io_mapping)
            f = os.path.basename(io_mapping)
            outputs[os.path.basename(d)].append(os.path.splitext(f)[0])
        done_count = 0
        worker_paths = glob(os.path.join(self.working_top_dir, 'worker_*'))
        for worker_path in worker_paths:
            for d in outputs.keys():
                pth = os.path.join(worker_path, 'output', d, self.OUTPUT_FILENAME)
                try:
                    with open(pth) as f:
                        seen_micrographs = set()
                        for line in f:
                            seen_micrographs.add(line.split()[0])
                        done_count += len(seen_micrographs)
                except FileNotFoundError:
                    pass
        return done_count


    def convert_output_files(self, job_object):
        working_top_dir_root = os.path.join(self.relion_job_dir, self.name)
        for worker_dir in sorted(glob(os.path.join(working_top_dir_root+'*', 'worker_*')), key=os.path.getmtime):
            topaz_output_files = set()
            output = check_output(f'find {worker_dir}/input -name "*.mrc"', shell=True,
                                  universal_newlines=True).splitlines()
            for pth in output:
                topaz_output_files.add(os.path.join(os.path.dirname(pth.replace('/input/', '/output/')), self.OUTPUT_FILENAME))
            for topaz_output_file in topaz_output_files:
                picks = defaultdict(list)
                with open(topaz_output_file) as f:
                    next(f)  # skip header line
                    for line in f:
                        splitline = line.split()
                        if float(splitline[3]) > self.args.threshold:
                            picks[splitline[0]].append(tuple(splitline[1:]))
                for k, v in picks.items():
                    self._write_pick_star_file(topaz_output_file, k, v)


    def _write_pick_star_file(self, topaz_output_file, micrograph_id, picks):
        with open(os.path.join(os.path.dirname(topaz_output_file), micrograph_id) + f'_{self.name}.star', 'w') as f:
            f.write('\n# version 30001\n')
            f.write('\ndata_\n')
            f.write('\nloop_\n')
            f.write('_rlnCoordinateX #1\n')
            f.write('_rlnCoordinateY #2\n')
            f.write('_rlnAutopickFigureOfMerit #3\n')
            f.write('_rlnClassNumber #4\n')
            f.write('_rlnAnglePsi #5\n')
            for pick in picks:
                f.write(f' {pick[0]:>7} {pick[1]:>7} {pick[2]:>12}  0  0.000000\n')  #TODO: convert log-likelyhood to FOM


    def run_worker(self, job_object, **kwargs):
        parsed_args = self.args
        env = kwargs.get('env', {})
        output = check_output('find input -name "*.mrc"', shell=True, universal_newlines=True).splitlines()
        input_dirs = set([os.path.dirname(f.split('input/')[-1]) for f in output])
        threads = int(int(parsed_args.j) / int(len(job_object.worker_dirs)))
        if threads < 1:
            threads = 1
        if 'CUDA_VISIBLE_DEVICES' in env:
            gpu_id = 0
        else:
            gpu_id = -1
        radius = parsed_args.radius
        scale = parsed_args.scale
        affine = parsed_args.affine
        model = parsed_args.model
        if model not in ['resnet16', 'resnet8', 'conv127', 'conv63', 'conv31']:
            model_abspath = os.path.abspath(os.path.join(job_object.top_dir, parsed_args.model))
            if os.path.isfile(model_abspath):
                model = model_abspath

        for d in input_dirs:
            os.makedirs(os.path.join('output', d), exist_ok=True)
            preproc_cmd = ''
            if gpu_id != -1:
                preproc_cmd += 'OMP_NUM_THREADS=2 '  # Pytorch seems to be ignoring the number of threads
            preproc_cmd += f'{TOPAZ} preprocess --scale {scale} '
            preproc_cmd += f'--device -1 --num-workers {min(threads, 6)} '
            if affine:
                preproc_cmd += ' --affine '
            preproc_cmd += f' --destdir preproc/{d} input/{d}/*.mrc'
            preproc_cmd += f' >preproc.out 2>preproc.err'
            pick_cmd = f'{TOPAZ} extract --device {gpu_id} --radius {radius} --up-scale {scale} --model {model} --threshold -6'
            pick_cmd += f' --output output/{d}/{self.OUTPUT_FILENAME} preproc/{d}/*.mrc'
            pick_cmd += f' >pick.out 2>pick.err'
            # pick_cmd = ' echo Skip pick'
            # print(f"Executing: {preproc_cmd + ' && sleep 1 && ' + pick_cmd}", file=sys.stderr, flush=True)
            proc = Popen(preproc_cmd + ' && sleep 1 && ' + pick_cmd, shell=True, env=env,
                         # stdout=PIPE, stderr=PIPE
                         )
            return proc


    def _get_header_length(self, star_lines):
        #TODO: convert to use relion_print_table
        header_lines = 0
        try:
            while not star_lines[header_lines].startswith('loop_'):
                header_lines += 1
            header_lines += 1
            while star_lines[header_lines].strip().startswith('_'):
                header_lines += 1
            return header_lines
        except IndexError:
            return len(star_lines)


    def worker_output_analysis_function(self, job_object):
        micrograph_source_dirs = self.relion_path_mapping(self.input_micrographs_starfile).keys()
        header_linecount = None
        pick_count = 0
        micrograph_count = 0
        for micrograph_output_dir in micrograph_source_dirs:
            relion_output_directory = os.path.join(self.relion_job_dir, micrograph_output_dir)
            for star_filename in glob(os.path.join(relion_output_directory, '*.star')):
                # print(read_star(star_filename))
                with open(star_filename) as f:
                    lines = f.readlines()
                    header_linecount = header_linecount or self._get_header_length(lines)
                    pick_count += len(lines) - header_linecount
                    micrograph_count += 1
        print(f'\n Total number of particles from {micrograph_count} micrographs is {pick_count}')
        try:
            print(f' i.e. on average there were {int(round(pick_count/micrograph_count))} particles per micrograph.')
        except ZeroDivisionError:
            pass


    def cleanup(self, job_object):
        if job_object.args.keep_preproc:
            for d in glob(os.path.join(job_object.working_top_dir, 'worker*')):
                shutil.rmtree(os.path.join(d, 'preproc'))


if __name__ == '__main__':
    job = Topaz_Picker()
    job.run()
