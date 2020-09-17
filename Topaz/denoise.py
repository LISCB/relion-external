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
from subprocess import Popen, check_output
import shutil

from util.framework.micrographs2micrographs import Micrographs2Micrographs


GPU_PYTHON = os.environ.get('TOPAZ_GPU_PYTHON', os.environ.get('TOPAZ_PYTHON', shutil.which('python3')))
TOPAZ_EXE = os.environ.get('TOPAZ_EXECUTABLE', shutil.which('topaz') )
TOPAZ = ' '.join((GPU_PYTHON, TOPAZ_EXE))

def run_worker(job_object, **kwargs):
    env = kwargs.get('env')
    output = check_output('find . -name "*.mrc"', shell=True, universal_newlines=True).splitlines()

    gpu_id = env.get('CUDA_VISIBLE_DEVICES', -1)
    cmd = f'{TOPAZ} denoise --patch-size 1024 --device {gpu_id} '

    dirs = set([os.path.dirname(d) for d in output])
    for d in dirs:
        basename = os.path.basename(d)
        cmd += f'--output output/{basename} {d}/*.mrc'
        cmd += f'>run.{basename}.out 2>run.{basename}.err'
        cmd += ' ; '
    proc = Popen(cmd, shell=True)
    return proc


if __name__ == '__main__':
    job = Micrographs2Micrographs(name='topaz-denoise',
                                  worker_run_function=run_worker,
                                  parallelizable=True)


    extra_text = 'Topaz Denoiser v0.2.4a\n' \
                 'Commercial and non-commercial use allowed.\n' \
                 'Citation:\n' \
                 'Bepler, T., Noble, A.J., Berger, B. (2019). Topaz-Denoise: general deep denoising models for cryoEM.\n' \
                 'bioRxiv. https://www.biorxiv.org/content/10.1101/838920v1\n'
    job.extra_text = extra_text

    job.run()
