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
from pathlib import Path
from subprocess import Popen, check_output, PIPE, CalledProcessError
from glob import glob
import json
import shutil

from util.framework.micrographs2starfiles import Micrographs2Starfiles


CRYOLO_BIN_DIR = os.environ.get('CRYOLO_BIN_DIR', os.path.dirname(shutil.which('python3')))
CPU_PYTHON = os.environ.get('CRYOLO_CPU_PYTHON', os.environ.get('CRYOLO_PYTHON', shutil.which('python3')))
GPU_PYTHON = os.environ.get('CRYOLO_GPU_PYTHON', os.environ.get('CRYOLO_PYTHON', shutil.which('python3')))
CRYOLO_PHOSNET_LOCATION = os.environ.get('CRYOLO_GENERAL_MODEL', '/net/common/cryolo/gmodel_phosnet_202005_N63_c17.h5')
CRYOLO_PHOSNET_NN_LOCATION = os.environ.get('CRYOLO_GENERAL_NN_MODEL', '/net/common/cryolo/gmodel_phosnet_202005_nn_N63_c17.h5')


def setup_worker(job_object):
    parsed_args = job_object.args
    cryolo_config = {
        'model': {
            'architecture': 'PhosaurusNet',
            'input_size': 1024,
            'max_box_per_image': 700,
            'norm': parsed_args.norm,
            'filter': [0.1, 'preproc/']
        },
        'other': {
            'log_path': 'logs/'
        }

    }
    with open('cryolo_config.json', 'w') as f:
        json.dump(cryolo_config, f)


def run_worker(job_object, **kwargs):
    parsed_args = job_object.args
    env = kwargs.get('env', {})
    output = check_output('find input -name "*.mrc"', shell=True,
                          universal_newlines=True).splitlines()
    input_dirs = set([os.path.dirname(f.split('input/')[-1]) for f in output])
    threads = int(int(parsed_args.j) / int(len(job_object.worker_dirs)))
    if threads < 1:
        threads = 1
    if 'CUDA_VISIBLE_DEVICES' in env:
        PYTHON = GPU_PYTHON
    else:
        PYTHON = CPU_PYTHON
    cryolo_cmds = []
    weights = parsed_args.weights
    abs_weights = os.path.abspath(os.path.join(job_object.top_dir, parsed_args.weights))
    if os.path.isfile(abs_weights):
        weights = abs_weights

    for i, d in enumerate(input_dirs):
        one_cryolo_cmd = ' '
        # one_cryolo_cmd = 'OMP_NUM_THREADS=1 '
        one_cryolo_cmd += ' '.join((PYTHON, os.path.join(CRYOLO_BIN_DIR, 'cryolo_predict.py')))
        one_cryolo_cmd += ' --write_empty --conf cryolo_config.json '
        one_cryolo_cmd += f' --weights {weights}'
        one_cryolo_cmd += f' --input input/{d} --output output/{d} '
        one_cryolo_cmd += f' --num_cpu {threads} '
        for arg in vars(parsed_args):
            if arg not in ['o', 'in_mics', 'j', 'cache', 'keep_preproc', 'weights',
                'filament', 'filament_width', 'mask_width', 'box_distance', 'minimum_number_boxes', 'search_range_factor',
                'otf', 'norm'
                           ]:
                one_cryolo_cmd += f' --{arg} {getattr(parsed_args, arg)} '
            elif arg in ['otf', 'filament']:
                if getattr(parsed_args, arg):
                    one_cryolo_cmd += f' --{arg} '
        one_cryolo_cmd += f' >cryolo_{i}.out 2>cryolo_{i}.err '
        cryolo_cmds.append(one_cryolo_cmd)
        # cryolo_cmd = 'sleep 10'
    # print(f'Executing: {cryolo_cmds}', file=sys.stderr, flush=True)
    proc = Popen(' & '.join(cryolo_cmds), shell=True, env=env,
                stdout=PIPE, stderr=PIPE,
                )
    return proc


def count_done_output_items(job_object):
    output_dirs = os.path.join(job_object.working_top_dir, 'worker*', 'output')
    try:
        find_output = check_output(f'find {output_dirs} -name "*.cbox"', shell=True,
                                   stderr=PIPE, universal_newlines=True).splitlines()
        return len(find_output)
    except CalledProcessError:
        return 0


def cbox_2_star(cbox_pth, threshold=0):
    star_header = '''
# version 30001

data_

loop_
_rlnCoordinateX #1
_rlnCoordinateY #2
_rlnAutopickFigureOfMerit #3
_rlnClassNumber #4
_rlnAnglePsi #5
'''
    picks = []
    sizes = []
    with open(cbox_pth) as f:
        for line in f:
            splitline = line.split()
            if float(splitline[4]) >= threshold:
                picks.append(
                    f' {splitline[0]:>7} {splitline[1]:>7} {splitline[4]:>12}  0  0.000000')
                sizes.append((int(splitline[5]), int(splitline[6])))
    loop_body = '\n'.join(picks)
    return star_header + loop_body


def cryolo_output_2_star(job_object):
    threshold = job_object.args.threshold
    output_subdirs = glob(os.path.join(job_object.relion_job_dir, f'{job_object.name}*',
                                       'worker*', 'output', '*'))
    for subdir in output_subdirs:
        for cbox_pth in glob(os.path.join(subdir, 'CBOX', '*.cbox')):
            star_pth = cbox_pth.replace('/CBOX/', '/')
            star_pth = os.path.splitext(star_pth)[0] + f'_{job_object.name}.star'
            star_text = cbox_2_star(cbox_pth, threshold)
            with open(star_pth, 'w') as f:
                f.write(star_text)


def analyze_cbox(job_object):
    threshold = job_object.args.threshold
    cboxes = {}
    for working_dir in sorted(glob(os.path.join(job_object.relion_job_dir,
                                                job_object.name + '*')),
                              key=os.path.getmtime):
        for cbox_pth in glob(os.path.join(working_dir,
                                           'worker*/output/*/CBOX/*.cbox')):
            cboxes[os.path.basename(cbox_pth)] = cbox_pth
    particle_sizes = []
    for cbox_pth in cboxes.values():
        with open(cbox_pth) as f:
            for line in f:
                splitline = line.split()
                if float(splitline[4]) >= threshold:
                    particle_sizes.append((int(splitline[5]), int(splitline[6])))

    if len(particle_sizes) > 0:
        min_sizes = sorted([min(s) for s in particle_sizes])
        max_sizes = sorted([max(s) for s in particle_sizes])
        sizes_len = len(min_sizes)
        fifth_percentile_index = int(sizes_len / 20)
        twentyfifth_percentile_index = int(sizes_len / 4)
        median_index = int(sizes_len / 2)
        print('\n Short axis statistics:')
        print(f'   Absolute minimum: {min_sizes[0]} pix')
        print(f'   5%-Quantile: {min_sizes[fifth_percentile_index]} pix')
        print(f'   25%-Quantile: {min_sizes[twentyfifth_percentile_index]} pix')
        print(
            f'   Median: {min_sizes[median_index]} pix (Mean: {(sum(min_sizes) / sizes_len):.1f} pix)')
        print(f'   75%-Quantile: {min_sizes[-twentyfifth_percentile_index]} pix')
        print(f'   95%-Quantile: {min_sizes[-fifth_percentile_index]} pix')
        print(f'   Absolute maximum: {min_sizes[-1]} pix')
        print(' Long axis statistics:')
        print(f'   Absolute minimum: {max_sizes[0]} pix')
        print(f'   5%-Quantile: {max_sizes[fifth_percentile_index]} pix')
        print(f'   25%-Quantile: {max_sizes[twentyfifth_percentile_index]} pix')
        print(
            f'   Median: {max_sizes[median_index]} pix (Mean: {(sum(max_sizes) / sizes_len):.1f} pix)')
        print(f'   75%-Quantile: {max_sizes[-twentyfifth_percentile_index]} pix')
        print(f'   95%-Quantile: {max_sizes[-fifth_percentile_index]} pix')
        print(f'   Absolute maximum: {max_sizes[-1]} pix')
        print()
    print(f' Total number of particles from {len(cboxes)} micrographs is {len(particle_sizes)}')
    try:
        print(f' i.e. on average there were {int(round(len(particle_sizes) / len(cboxes)))} particles per micrograph')
    except ZeroDivisionError:
        pass

def remove_preproc(job_object):
    pass
    # print('Remove preproc files (?)')


if __name__ == '__main__':
    job = Micrographs2Starfiles(name='cryolo-pick',
                                worker_setup_function=setup_worker,
                                worker_run_function=run_worker,
                                worker_output_file_converter_function=cryolo_output_2_star,
                                worker_output_analysis_function=analyze_cbox,
                                worker_cleanup_function=remove_preproc,
                                preproc_dir=True,
                                parallelizable=True)

    job.extra_text = 'crYOLO Version: 1.7.2\n' \
    '*** Not licensed for commercial use. ***'

    job.parser.add_argument('--weights', default=CRYOLO_PHOSNET_LOCATION,
                            help='Trained weights.  (Default: General PhosaurusNet Model.)')
    job.parser.add_argument('--threshold', type=float, default=0.3,
                            help='Picking threshold.  (Default: 0.3.  Lower means pick more.)')
    job.parser.add_argument('--norm', default='STANDARD',
                            choices=['GMM', 'STANDARD'],
                            help='Normalization that is applied to the images. '
                                 'STANDARD will subtract the image mean and divide by the standard deviation. '
                                 'Experimental: Gaussian Mixture Models (GMM) fit a 2 component GMM to you image data '
                                 'and normalizes according the brighter component. '
                                 'This ensures that it always normalize with respect to ice but slows down the training. '
                                 '(Default: STANDARD)')
    job.parser.add_argument('--prediction_batch_size', type=int, default=3,
                            help='Images per batch, lower values will help with memory issues.  (Default: 3)')
    job.parser.add_argument('--gpu_fraction', type=float, default=1.0,
                            help='Fraction of GPU memory to use.  (Default: 1.0)')
    job.parser.add_argument('--otf', action='store_true',
                            help='On-The-Fly pre-filtering.  Not currently multi-threaded. (Default: off)')
    job.parser.add_argument('--norm_margin', type=float, default=0.2,
                            help='Relative margin size for normalization. (Default: 0.2)')
    job.parser.add_argument('--filament', action='store_true',
                            help='Use filament mode.  (default: off)')
    job.parser.add_argument('--filament_width', type=int, required=False,
                            help='Filament width in pixels.')
    job.parser.add_argument('--mask_width', type=int, default=100,
                            help='Mask width in pixels.')
    job.parser.add_argument('--box_distance', type=int, default=0,
                            help='Distance in pixel between two boxes.')
    job.parser.add_argument('--minimum_number_boxes', type=int, default=0,
                            help='Minimum number of boxes per filament.')
    job.parser.add_argument('--search_range_factor', type=float, default=1.41,
                            help='The search range for connecting boxes is the box size times this factor.')

    job.count_done_output_items = count_done_output_items

    job._parse_args()

    if not os.path.isfile(job.args.weights):
        # Path('RELION_JOB_EXIT_FAILURE').touch()
        raise(FileNotFoundError(f'The model file {job.args.weights} was not found.'))

    job.run()
