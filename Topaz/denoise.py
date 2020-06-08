import os
from subprocess import Popen, check_output, PIPE

from util.framework.micrographs2micrographs import Micrographs2Micrographs


TOPAZ = '/net/prog/anaconda3/envs/topaz/bin/python /net/prog/anaconda3/envs/topaz/bin/topaz'


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
