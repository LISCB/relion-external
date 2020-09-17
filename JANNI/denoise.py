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
import shutil
from subprocess import Popen
from functools import partial
import argparse

from util.framework.micrographs2micrographs import Micrographs2Micrographs


CPU_PYTHON = os.environ.get('JANNI_CPU_PYTHON', os.environ.get('JANNI_PYTHON', shutil.which('python3')))
GPU_PYTHON = os.environ.get('JANNI_GPU_PYTHON', os.environ.get('JANNI_PYTHON', shutil.which('python3')))
JANNI_LOCATION = os.environ.get('JANNI_DEFAULT_MODEL', '/net/common/janni/gmodel_janni_20190703.h5')


def run_worker(job_object, **kwargs):
    env = kwargs.get('env')
    args = job_object.args
    cmd = os.path.abspath(__file__) + ' --slave'
    for arg in vars(args):
        if arg not in ['slave', 'gpu', 'j', 'keep_preproc', 'cache']:
            cmd += f' --{arg} {getattr(args, arg)}'
    if 'CUDA_VISIBLE_DEVICES' in env:
        PYTHON = GPU_PYTHON
        cmd += f' --g {env["CUDA_VISIBLE_DEVICES"]}'
    else:
        PYTHON = CPU_PYTHON
    cmd += ' >janni.out 2>janni.err'
    proc = Popen(PYTHON + ' ' + cmd, shell=True)
    return proc


def run_as_slave(self, *args, **kwargs):
    from glob import glob
    from janni import predict

    for input_dir in glob('input/*'):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.args.g)
        predict.predict(input_path=input_dir,
                        output_path='output',
                        model_path=JANNI_LOCATION,
                        batch_size=2,
                        # output_resize_to=None, squarify=None,
                        # **janni_params
                        )


if __name__ == '__main__':
    extra_text = 'JANNI Denoiser v0.1.4\n' \
                 '*** NON-COMMERCIAL USE ONLY ***\n' \
                 'Citation:\n' \
                 'DOI: 10.5281/zenodo.3378300\n' \
                 'https://zenodo.org/badge/latestdoi/192689060'
    job = Micrographs2Micrographs(name='janni-denoise',
                                  worker_run_function=run_worker,
                                  parallelizable=True,
                                  extra_text=extra_text)

    job.parser.add_argument('--slave', action='store_true', help=argparse.SUPPRESS)
    job.parser.add_argument('--g', metavar="GPU_ID", type=int, default=0,
                             help="Which GPU to use. (Default: 0)")
    job._parse_args()
    if job.args.slave:
        job.run = partial(run_as_slave, job)

    job.run()
