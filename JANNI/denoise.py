#! /net/prog/anaconda3/envs/cryolo/bin/python

from collections import defaultdict
import shutil
import glob
import sys
import os
import argparse
import subprocess
import time
from pathlib import Path
import signal
from random import random
import tempfile


PROGRAM_NAME = 'janni'
CPU_PYTHON = '/net/prog/anaconda3/envs/cryolo/bin/python'
GPU_PYTHON = '/net/prog/anaconda3/envs/cryolo-gpu/bin/python'
JANNI_LOCATION = '/net/common/janni/gmodel_janni_20190703.h5'

SUFFIX_STAR_FILENAME = f'coords_suffix_{PROGRAM_NAME}.star'

HW_GPU_LIST = subprocess.run('lspci | grep "VGA compatible controller: NVIDIA"', shell=True,
                               stdout=subprocess.PIPE).stdout.decode().splitlines()
CUDA_VISIBLE_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES')
if CUDA_VISIBLE_DEVICES:
    CUDA_VISIBLE_DEVICE_IDS = [int(g) for g in CUDA_VISIBLE_DEVICES.split(',')]
    GPU_IDS = CUDA_VISIBLE_DEVICE_IDS
else:
    GPU_IDS = list(range(len(HW_GPU_LIST)))
GPU_COUNT = len(GPU_IDS)


TOP_DIR = os.getcwd()
# MOUSE = r'~~(,_,">'
MOUSE = r'~~( ϵ:>'
CHEESE = r'[oo]'

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--o', metavar='DIRECTORY',
                        help='Job output directory. DO NOT USE IF RUNNING FROM RELION.')
    parser.add_argument('--in_mics', metavar='STARFILE',
                        help='Input micrographs .star file. DO NOT USE IF RUNNING FROM RELION.')
    parser.add_argument('--model', default=JANNI_LOCATION,
                        help='Pretrained model.  (Default: General JANNI U-net.)')
    parser.add_argument('--patch_size', type=int, default=1024,
                        help='Denoising patch size.  (Default: 1024.  Lower means pick more.)')
    parser.add_argument('--batch_size', type=int,
                        default=4, help='Patches per batch, lower values will help with memory issues.  (Default: 4)')
    parser.add_argument('--padding', type=int, default=15,
                        help='Patch padding.  (Default: 15)')
    parser.add_argument('--j', metavar="NUM_CPU", type=int, default=1,
                        help="Threads per job. (Default: 1 per GPU)")
    parser.add_argument('-g', '--gpu', type=int, default=0, help="Which GPU to use.")
    # TODO: make it work with more than one GPU
    parser.add_argument('--slave', action='store_true', help=argparse.SUPPRESS)

    args = parser.parse_args()
    if not (args.o and args.in_mics):
        print('ERROR: Input and output are required.')
        sys.exit(1)
    return args


def get_input_micrograph_paths(starfile_location):
    metadata = get_input_micrograph_metadata(starfile_location)
    micrographs = metadata['micrographs']
    return [m[0] for m in micrographs]


def get_input_micrograph_metadata(starfile_location):
    # TODO: use default callable and add commandline parameter to specify
    with tempfile.TemporaryDirectory() as tmpdirname:
        curr_dir = os.getcwd()
        abs_starfile_location = os.path.abspath(starfile_location)
        ogs = {}
        micrographs = []
        try:
            os.chdir(tmpdirname)
            og_fields = ['_rlnOpticsGroupName', '_rlnOpticsGroup', '_rlnMicrographPixelSize', '_rlnVoltage', '_rlnSphericalAberration']
            og_fields = ' '.join(og_fields)
            optics_proc = subprocess.run(
                f'relion_star_printtable {abs_starfile_location} data_optics {og_fields}',
                check=True, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            for og in optics_proc.stdout.decode().splitlines():
                split_og = og.strip().split()
                ogs[split_og[0]] = {'rlnOpticsGroup': int(split_og[1]),
                                    'rlnMicrographPixelSize': float(split_og[2]),
                                    'rlnVoltage': float(split_og[3]),
                                    'rlnSphericalAberration': float(split_og[4])}

            micrographs_proc = subprocess.run(
                f'relion_star_printtable {abs_starfile_location} data_micrographs _rlnMicrographName _rlnOpticsGroup',
                check=True, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            for m in micrographs_proc.stdout.decode().splitlines():
                split_m = m.strip().split()
                micrographs.append((split_m[0], int(split_m[1])))
        finally:
            os.chdir(curr_dir)
        return {'optics groups': ogs, 'micrographs': micrographs}


def call_slaves(args):
    #TODO: This should check for already done micrographs, split them up, then partition them to slaves using tempfiles
    my_env = os.environ.copy()
    if HW_GPU_LIST and not CUDA_VISIBLE_DEVICES:
        my_env['CUDA_VISIBLE_DEVICES'] = ','.join([str(id) for id in GPU_IDS])
    slave_cmd = os.path.abspath(__file__) + ' --slave'
    for arg in vars(args):
        if arg not in ['slave', 'gpu', 'j']:
            slave_cmd += f' --{arg} {getattr(args, arg)}'
    if GPU_IDS:
        j = max(int(args.j / len(GPU_IDS)), 1)
        slave_cmd = GPU_PYTHON + ' ' + slave_cmd
        # [print(f'{slave_cmd} --gpu {g}') for g in GPU_IDS]
        return tuple([subprocess.Popen(f'{slave_cmd} --gpu {g}', env=my_env, shell=True,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE
                              ) for g in GPU_IDS])
    else:
        slave_cmd = CPU_PYTHON + ' ' + slave_cmd
        # print(slave_cmd)
        print(' Initialising denoiser(s).  (This can a minute or two, please wait.)')
        return (subprocess.Popen(slave_cmd, env=my_env, shell=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE
                                 ), )


def check_abort_signal(abs_job_dir):
    # return True
    if os.path.exists(os.path.join(abs_job_dir, 'RELION_JOB_ABORT_NOW')):
        return True
    return False


def micrographs_to_process(relion_job_dir, in_mics):
    output_dirs = _get_output_dirs(get_input_micrograph_paths(in_mics))
    micrographs_to_process = []
    for output_dir, micrograph_list in output_dirs.items():
        for micrograph in micrograph_list:
            if not os.path.isfile(os.path.join(relion_job_dir, output_dir, os.path.basename(micrograph))):
                micrographs_to_process.append(micrograph)
    return micrographs_to_process


def update_progress_bar(start_time=None, total_count=0, current_count=0, current_time=None):
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
            #TODO: Fix: this currently overruns the cheese by 3~4 spaces
            dot_count = int(round((TOTAL_DOTS+len(CHEESE)) * current_count / total_count))
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


def monitor_slaves(args, slaves):
    relion_job_dir = args.o
    total_micrographs_to_process = len(micrographs_to_process(args.o, args.in_mics))
    previous_remaining_micrographs_to_process = total_micrographs_to_process
    start_time = None
    print(update_progress_bar(), end='\r')
    while any([proc.poll() is None for proc in slaves]):
        for i, proc in enumerate(slaves):
            poll = proc.poll()
            if (poll is not None) and (poll > 0):
                raise Exception('crYOLO job failed. Exiting.')
        abort = check_abort_signal(relion_job_dir)
        if abort:
            for proc in slaves:
                os.kill(proc.pid, signal.SIGTERM)
            raise Exception('Relion RELION_JOB_ABORT_NOW file seen. Terminating.')

        remaining_micrographs_to_process = len(micrographs_to_process(args.o, args.in_mics))
        if remaining_micrographs_to_process < previous_remaining_micrographs_to_process:
            previous_remaining_micrographs_to_process = remaining_micrographs_to_process
            start_time = start_time or time.time()
            print(update_progress_bar(start_time, total_micrographs_to_process, total_micrographs_to_process-remaining_micrographs_to_process), flush=True, end='')
        time.sleep(1)
    print()


def make_output_tree(args):
    relion_job_dir = args.o
    os.makedirs(relion_job_dir, exist_ok=True)
    output_dirs = _get_output_dirs(get_input_micrograph_paths(args.in_mics))
    output_dirs = output_dirs.keys()
    for output_dir in output_dirs:
        os.makedirs(os.path.join(relion_job_dir, output_dir), exist_ok=True)
    return relion_job_dir


def write_relion_star_final(relion_job_dir, input_star_file):
    input_micrograph_metadata = get_input_micrograph_metadata(input_star_file)
    optics_groups = input_micrograph_metadata['optics groups']
    input_micrographs = input_micrograph_metadata['micrographs']

    input_micrograph_paths = get_input_micrograph_paths(input_star_file)
    output_image_paths = []
    for output_dir in _get_output_dirs(input_micrograph_paths):
        output_image_paths += glob.glob(os.path.join(relion_job_dir, output_dir, '*.mrc'))

    with open(os.path.join(relion_job_dir, "denoised_micrographs.star"), 'w') as f:
        f.write('\n# version 30001\n\n')
        f.write('data_optics\n\n')
        f.write('loop_\n')
        f.write('_rlnOpticsGroupName  # 1\n')
        f.write('_rlnOpticsGroup  # 2\n')
        f.write('_rlnVoltage  # 3\n')
        f.write('_rlnSphericalAberration  # 4\n')
        f.write('_rlnMicrographPixelSize  # 5\n')
        for og_name, og in optics_groups.items():
            f.write(f"{og_name} ")
            f.write(f" {og['rlnOpticsGroup']:4d} ")
            f.write(f" {og['rlnVoltage']:4.3f} ")
            f.write(f" {og['rlnSphericalAberration']:2.3f} ")
            f.write(f" {og['rlnMicrographPixelSize']:3.5f} ")
            f.write("\n")

        f.write('\n# version 30001\n\n')
        f.write('data_micrographs\n\n')
        f.write('loop_\n')
        f.write('_rlnMicrographName  # 1\n')
        f.write('_rlnOpticsGroup  # 2\n')

        for micrograph, og_number in input_micrographs:
            for output_image_path in output_image_paths:
                output_image_basename = os.path.basename(output_image_path)
                if os.path.basename(micrograph) == output_image_basename:
                    f.write(f'{output_image_path}  {og_number:3d}\n')
                    break
        # for output_image_path in output_image_paths:
        #     output_image_basename = os.path.basename(output_image_path)
        #     for micrograph, og_number in input_micrographs:
        #         if os.path.basename(micrograph) == output_image_basename:
        #             f.write(f'{output_image_path}  {og_number:3d}\n')
        #             break

    with open(os.path.join(relion_job_dir, 'RELION_OUTPUT_NODES.star'), 'w') as f:
        f.write('data_output_nodes\n')
        f.write('loop_\n')
        f.write('_rlnPipeLineNodeName #1\n')
        f.write('_rlnPipeLineNodeType #2\n')
        f.write(f'{os.path.join(relion_job_dir, "denoised_micrographs.star")}    1\n')


def _get_denoised_micrograph_count(relion_job_dir, input_star_file):
    micrograph_paths = get_input_micrograph_paths(input_star_file)
    output_image_count = 0
    for output_dir in _get_output_dirs(micrograph_paths):
        output_image_count += len(glob.glob(os.path.join(relion_job_dir, output_dir, '*.mrc')))
    return output_image_count

def print_results_summary(relion_job_dir, input_star_file, initial_micrograph_count=0):
    output_image_count = _get_denoised_micrograph_count(relion_job_dir, input_star_file)
    print(f'\nFinished denoising {output_image_count-initial_micrograph_count} micrographs.')
    print(f'{output_image_count} total denoised micrographs available.')


def run_as_master(args):
    for f in ('RELION_JOB_EXIT_FAILURE', 'RELION_JOB_EXIT_SUCCESS'):
        try: os.remove(os.path.join(args.o, f))
        except FileNotFoundError: pass

    print('JANNI Wrapper for Relion v3.1', flush=True)
    print('Written by TJ Ragan (LISCB, University of Leicester)\n', flush=True)

    print('JANNI Version: ???', flush=True)
    print('Wrapper Version: 0.1\n', flush=True)

    print(f'Print compute info here.', flush=True)
    if GPU_COUNT:
        print(f"Using {GPU_COUNT} GPUs: {GPU_IDS}.", flush=True)
    else:
        print(f"Using CPUs only.  This will be much slower than running on GPUs!", flush=True)

    print(f'=================', flush=True)

    try:
        relion_job_dir = make_output_tree(args)

        initial_denoised_count = _get_denoised_micrograph_count(relion_job_dir, args.in_mics)

        slaves = call_slaves(args)
        monitor_slaves(args, slaves)
        Path(os.path.join(relion_job_dir, 'RELION_JOB_EXIT_SUCCESS')).touch()

        print_results_summary(relion_job_dir, args.in_mics, initial_denoised_count)
        write_relion_star_final(relion_job_dir, args.in_mics)

    except Exception as e:
        relion_job_dir = args.o
        if os.path.exists(os.path.join(relion_job_dir, 'RELION_JOB_ABORT_NOW')):
            os.remove(os.path.join(relion_job_dir, 'RELION_JOB_ABORT_NOW'))
            Path(os.path.join(relion_job_dir, 'RELION_JOB_EXIT_ABORTED')).touch()
        else:
            print('boom!')
            Path('RELION_JOB_EXIT_FAILURE').touch()
            raise e

    Path(os.path.join(relion_job_dir, 'RELION_JOB_EXIT_SUCCESS')).touch()
    print(f'\n Done!\n')


def run_as_slave(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    janni_params = {'patch_size': (args.patch_size,)*2,  # Square patches
                    'batch_size': args.batch_size,
                    'padding': args.padding,
                    }

    relion_job_dir = args.o
    input_star_file = args.in_mics
    micrograph_paths = get_input_micrograph_paths(input_star_file)
    if CUDA_VISIBLE_DEVICES:
        my_micrograph_paths = micrograph_paths[args.gpu::GPU_COUNT]
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    else:
        my_micrograph_paths = micrograph_paths
    denoise(relion_job_dir, my_micrograph_paths, os.path.abspath(args.model), janni_params)


def _get_output_dirs(micrograph_paths):
    output_dirs = defaultdict(list)
    for micrograph_path in micrograph_paths:
        pth, fname = os.path.split(micrograph_path)
        output_dirs[os.path.basename(pth)].append(micrograph_path)
    return output_dirs


def denoise(relion_job_dir, micrograph_paths, model_path, janni_params, force=False):
    output_dirs = _get_output_dirs(micrograph_paths)
    micrographs_to_do = defaultdict(list)
    for output_dir, micrographs in output_dirs.items():
        for micrograph in micrographs:
            if not os.path.isfile(os.path.join(relion_job_dir, output_dir, os.path.basename(micrograph))):
                micrographs_to_do[output_dir].append(micrograph)
    if micrographs_to_do:
        from janni import predict
        from janni import models
        model = models.get_model_unet(input_size=janni_params['patch_size'])
        model.load_weights(model_path)
        for output_dir, micrographs in micrographs_to_do.items():
            predict.predict_list(image_paths=micrographs, model=model,
                                 output_path=relion_job_dir,
                                 output_resize_to=None, squarify=None,
                                 **janni_params)


if __name__ == '__main__':
    args = parse_args()

    if not args.slave:
        run_as_master(args)
    else:
        run_as_slave(args)
