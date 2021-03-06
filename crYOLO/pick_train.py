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

# TODO: use ssd as temp file

import os
from subprocess import Popen
import shutil
from collections import defaultdict
import json

from util.framework.particles_to import ParticlesTo


CRYOLO_BIN_DIR = os.environ.get('CRYOLO_BIN_DIR', os.path.dirname(shutil.which('python3')))
CPU_PYTHON = os.environ.get('CRYOLO_CPU_PYTHON', os.environ.get('CRYOLO_PYTHON', shutil.which('python3')))
GPU_PYTHON = os.environ.get('CRYOLO_GPU_PYTHON', os.environ.get('CRYOLO_PYTHON', shutil.which('python3')))
CRYOLO_PHOSNET_LOCATION = os.environ.get('CRYOLO_GENERAL_MODEL', '/net/common/cryolo/gmodel_phosnet_202005_N63_c17.h5')


def particles_star_to_cryolo(starfile, output_path, size):
    by_micrograph = defaultdict(list)
    for particle in starfile['particles']:
        micrograph_name = os.path.splitext(os.path.basename(particle.rlnMicrographName))[0]
        by_micrograph[micrograph_name].append(f'{int(particle.rlnCoordinateX)} {int(particle.rlnCoordinateY)} {size} {size}')
    for micrograph_name, coords in by_micrograph.items():
        with open(os.path.join(output_path, micrograph_name+'.box'), 'w') as f:
            f.write('\n'.join(coords))


def setup_worker(job_object):
    try:
        assert len(job_object.input_particles_starfile['optics groups']) == 1
    except AssertionError:
        raise NotImplementedError('Training only supports one optics group right now.')

    micrographs = job_object.get_micrographs_in_particles_starfile(job_object.input_particles_starfile)

    input_symlink_target_dir = os.path.join(job_object.working_top_dir, 'worker_0', 'input')

    for m in micrographs:
        fname = os.path.basename(m)
        abs_m_path = os.path.abspath(m)
        src = os.path.relpath(abs_m_path, input_symlink_target_dir)
        dest = os.path.join(input_symlink_target_dir, fname)
        os.symlink(src, dest)

    particles_star_to_cryolo(job_object.input_particles_starfile,
                             os.path.join(job_object.working_top_dir, 'worker_0', 'input'),
                             job_object.args.box_size)

    parsed_args = job_object.args
    cryolo_config = {
        'model': {
            'architecture': 'PhosaurusNet',
            'input_size': 1024,
            'anchors': [job_object.args.box_size, job_object.args.box_size],
            'max_box_per_image': 700,
            'norm': parsed_args.norm,
            'filter': [0.1, 'preproc/']
        },
        "train": {
            "train_image_folder": "input",
            "train_annot_folder": "input",
            "train_times": 10,
            "pretrained_weights": f"{job_object.args.weights}",
            "batch_size": 3,
            "learning_rate": 0.0001,
            "nb_epoch": parsed_args.num_epochs,
            "object_scale": 5.0,
            "no_object_scale": 1.0,
            "coord_scale": 1.0,
            "class_scale": 1.0,
            "saved_weights_name": "cryolo_model.h5",
            "debug": True
        },
        'valid': {
            'valid_image_folder': '',
            'valid_annot_folder': '',
            'valid_times': 1
        },
        'other': {
            'log_path': 'logs/'
        }

    }
    with open(os.path.join(job_object.working_top_dir, 'worker_0', 'cryolo_config.json'), 'w') as f:
        json.dump(cryolo_config, f)
    return job_object.args.num_epochs, len(micrographs)


def count_done_output_items(job_object):
    last_epoch = 0
    with open(os.path.join(job_object.working_top_dir, 'worker_0', 'cryolo.out')) as f:
        for line in f:
            if line.strip().startswith('Epoch ') and '/' not in line:
                last_epoch = line.split('Epoch ')[1]
                last_epoch = last_epoch.split(':')[0]
                last_epoch = int(last_epoch)
    return last_epoch


def run_worker(job_object, **kwargs):
    parsed_args = job_object.args
    env = kwargs.get('env', {})
    if 'CUDA_VISIBLE_DEVICES' in env:
        PYTHON = GPU_PYTHON
        # gpu_string = f" -g {env['CUDA_VISIBLE_DEVICES']} "  # multi-gpu training broken
        gpu_string = f" -g 0 "
    else:
        PYTHON = CPU_PYTHON
        gpu_string = ""
    cryolo_cmd = ' '.join((PYTHON, os.path.join(CRYOLO_BIN_DIR, 'cryolo_train.py'), gpu_string))
    cryolo_cmd += ' --use_multithreading '
    cryolo_cmd += f' --warmup {parsed_args.warmup} --conf cryolo_config.json '
    cryolo_cmd += ' > cryolo.out 2> cryolo.err '
    proc = Popen(cryolo_cmd, shell=True, env=env)
    return proc



if __name__ == '__main__':
    job = ParticlesTo(name='cryolo-train',
                                worker_setup_function=setup_worker,
                                worker_run_function=run_worker,
                                preproc_dir=True,
                                parallelizable=False)

    job.extra_text = 'crYOLO Version: 1.7.2\n' \
    '*** Not licensed for commercial use. ***'

    job.parser.add_argument('--box_size', type=int, default=200,
                            help='Smalles box size (in pixels) containin your particles.  (Default: 200.)')
    job.parser.add_argument('--warmup', type=int, default=5,
                            help='Number of warmup epochs. Ignored if you turn on fine-tuning.  (Default: 5.)')
    job.parser.add_argument('--num_epochs', type=int, default=200,
                            help='Number of training epochs. (Default: 200)')
    job.parser.add_argument('--weights', default=CRYOLO_PHOSNET_LOCATION,
                            help='Trained weights.  (Default: General PhosaurusNet Model.)')
    job.parser.add_argument('--norm', default='GMM',
                            choices=['GMM', 'STANDARD'],
                            help='Normalization that is applied to the images. '
                                 'STANDARD will subtract the image mean and divide by the standard deviation. '
                                 'Experimental: Gaussian Mixture Models (GMM) fit a 2 component GMM to you image data '
                                 'and normalizes according the brighter component. '
                                 'This ensures that it always normalize with respect to ice but slows down the training. '
                                 '(Default: GMM)')
    job.parser.add_argument('-e', '--early', type=int, default=10,
                            help='Early stop patience. If the validation loss did not improve longer than the early stop patience, '
                                 'the training is stopped. (default: 10)')
    job.parser.add_argument('--fine_tune', action='store_true',
                            help='Set it to true if you only want to use the fine tune mode. '
                                 'When using the fine tune mode, only the last layers of your network '
                                 'are trained and you have to specify pretrained_weights. '
                                 'You typically use a general model as pretrained weights. (default: False)')
    job.parser.add_argument('--lft', type=int, default=2,
                            help='Layers to be trained when using fine tuning. (default: 2)')
    # TODO: train_times, batch_size, learning_rate, object_scale, no_object_scale, coord_scale, class_scale

    job.count_done_output_items = count_done_output_items


    job.run()
