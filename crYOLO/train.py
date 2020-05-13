#! /net/prog/anaconda3/envs/cryolo/bin/python

#/net/prog/anaconda3/envs/cryolo-gpu/bin/cryolo_train.py -c config_cryolo2.json -w 5
#/net/prog/anaconda3/envs/cryolo-gpu/bin/cryolo_train.py -c config_cryolo2_weights.json -w 0 -lft 2 --fine_tune

''' Relion Polish training output:
 + Reading Refine3D/job070/run_data.star...
 + Using movie pixel size from MotionCorr/job002/Micrographs/20170629_00021_frameImage.star: 0.885 A
 + Using coord. pixel size from MotionCorr/job002/Micrographs/20170629_00021_frameImage.star: 0.885 A
 + Using dose per frame from MotionCorr/job002/Micrographs/20170629_00021_frameImage.star: 1.277 e/A^2
 + Reading references ...
 + The names of the reference half maps and the mask were taken from the PostProcess STAR file.
   - Half map 1: Refine3D/job070/run_half1_class001_unfil.mrc
   - Half map 2: Refine3D/job070/run_half2_class001_unfil.mrc
   - Mask: MaskCreate/job071/mask.mrc
 + Masking references ...
 + Transforming references ...
 + Initializing motion estimator ...
 + Initializing motion parameter estimator ...
 + estimating motion parameters for all optics groups simultaneously ...
 + maximum frequency to consider for alignment: 4.08462 A (78 ref. px)
 + frequency range to consider for evaluation:  4.08462 - 2.89636 A (78 - 110 ref. px)
 + micrographs randomly selected for parameter optimization:
        15: MotionCorr/job002/Micrographs/20170629_00040_frameImage.mrc
        16: MotionCorr/job002/Micrographs/20170629_00042_frameImage.mrc
        13: MotionCorr/job002/Micrographs/20170629_00037_frameImage.mrc
        19: MotionCorr/job002/Micrographs/20170629_00045_frameImage.mrc
        11: MotionCorr/job002/Micrographs/20170629_00035_frameImage.mrc
        7: MotionCorr/job002/Micrographs/20170629_00028_frameImage.mrc
        6: MotionCorr/job002/Micrographs/20170629_00027_frameImage.mrc
        3: MotionCorr/job002/Micrographs/20170629_00024_frameImage.mrc
        4: MotionCorr/job002/Micrographs/20170629_00025_frameImage.mrc
        1: MotionCorr/job002/Micrographs/20170629_00022_frameImage.mrc
        8: MotionCorr/job002/Micrographs/20170629_00029_frameImage.mrc

 + 2066 particles found in 11 micrographs
 + preparing alignment data...
        micrograph 1 / 11: 209 particles [209 total]
        micrograph 2 / 11: 264 particles [473 total]
        micrograph 3 / 11: 131 particles [604 total]
        micrograph 4 / 11: 157 particles [761 total]
        micrograph 5 / 11: 146 particles [907 total]
        micrograph 6 / 11: 200 particles [1107 total]
        micrograph 7 / 11: 216 particles [1323 total]
        micrograph 8 / 11: 157 particles [1480 total]
        micrograph 9 / 11: 193 particles [1673 total]
        micrograph 10 / 11: 184 particles [1857 total]
        micrograph 11 / 11: 209 particles [2066 total]
   done

it: 	 s_vel: 	 s_div: 	 s_acc: 	 fsc:

0: 	 0.90000  	 10000.00000  	 3.00000  	 0.005414960030
1: 	 0.90000  	 10000.00000  	 3.00000  	 0.005414960030
2: 	 0.90000  	 10000.00000  	 3.00000  	 0.005414960030
3: 	 0.90000  	 10000.00000  	 3.00000  	 0.005414960030
4: 	 0.90000  	 10000.00000  	 3.00000  	 0.005414960030
5: 	 0.90000  	 10000.00000  	 3.00000  	 0.005414960030
6: 	 0.90000  	 10000.00000  	 3.00000  	 0.005414960030
7: 	 0.85000  	 8500.00000  	 2.50000  	 0.005415314909
8: 	 0.85000  	 8500.00000  	 2.50000  	 0.005415314909
9: 	 0.85000  	 8500.00000  	 2.50000  	 0.005415314909
10: 	 0.85000  	 8500.00000  	 2.50000  	 0.005415314909
11: 	 0.92743  	 9635.41667  	 2.23264  	 0.005415453703
12: 	 0.92743  	 9635.41667  	 2.23264  	 0.005415453703
13: 	 0.92743  	 9635.41667  	 2.23264  	 0.005415453703
14: 	 0.92743  	 9635.41667  	 2.23264  	 0.005415453703
15: 	 0.92743  	 9635.41667  	 2.23264  	 0.005415453703
16: 	 0.90342  	 8491.97049  	 2.23647  	 0.005415467155
17: 	 0.90342  	 8491.97049  	 2.23647  	 0.005415467155
18: 	 0.89630  	 8860.39979  	 2.28972  	 0.005415467965
19: 	 0.90723  	 9052.57965  	 2.18620  	 0.005415471246
20: 	 0.91487  	 9218.53332  	 2.23505  	 0.005415473107
21: 	 0.90478  	 8767.90404  	 2.23673  	 0.005415485061
22: 	 0.90478  	 8767.90404  	 2.23673  	 0.005415485061
23: 	 0.92029  	 9048.78629  	 2.22761  	 0.005415485613
24: 	 0.92029  	 9048.78629  	 2.22761  	 0.005415485613
25: 	 0.92029  	 9048.78629  	 2.22761  	 0.005415485613
26: 	 0.92029  	 9048.78629  	 2.22761  	 0.005415485613
27: 	 0.91114  	 8909.45687  	 2.22416  	 0.005415485937
28: 	 0.91645  	 9033.73888  	 2.20598  	 0.005415486088
29: 	 0.91645  	 9033.73888  	 2.20598  	 0.005415486088
30: 	 0.91645  	 9033.73888  	 2.20598  	 0.005415486088
31: 	 0.91459  	 8958.03812  	 2.21579  	 0.005415486249
32: 	 0.92109  	 9096.76701  	 2.20502  	 0.005415486486
33: 	 0.92109  	 9096.76701  	 2.20502  	 0.005415486486
34: 	 0.91897  	 9030.44490  	 2.20514  	 0.005415486490
35: 	 0.91897  	 9030.44490  	 2.20514  	 0.005415486490
36: 	 0.91897  	 9030.44490  	 2.20514  	 0.005415486490
37: 	 0.91897  	 9030.44490  	 2.20514  	 0.005415486490
38: 	 0.92204  	 9105.21714  	 2.19979  	 0.005415486522
39: 	 0.92204  	 9105.21714  	 2.19979  	 0.005415486522
40: 	 0.92121  	 9090.20575  	 2.20369  	 0.005415486543
41: 	 0.92121  	 9090.20575  	 2.20369  	 0.005415486543
42: 	 0.92121  	 9090.20575  	 2.20369  	 0.005415486543

good parameters: --s_vel 0.921 --s_div 9090 --s_acc 2.205

written to Polish/job074/opt_params_all_groups.txt
'''

from collections import defaultdict
import shutil
import sys
import os
import argparse
import subprocess
from pathlib import Path
import json
import mrcfile
import time
import signal


PROGRAM_NAME = 'cryolo'
CPU_EXECUTABLE = '/net/prog/anaconda3/envs/cryolo/bin/cryolo_train.py'
GPU_EXECUTABLE = '/net/prog/anaconda3/envs/cryolo-gpu/bin/cryolo_train.py'
CRYOLO_PHOSNET_NN_LOCATION = '/net/common/cryolo/gmodel_phosnet_202003_nn_N63.h5'
CRYOLO_PHOSNET_LOCATION = '/net/common/cryolo/gmodel_phosnet_202002_N63.h5'


HW_GPU_LIST = subprocess.run('lspci | grep "VGA compatible controller: NVIDIA"', shell=True,
                               stdout=subprocess.PIPE).stdout.decode().splitlines()
SLURM_GPU_IDS = os.environ.get('CUDA_VISIBLE_DEVICES')
if SLURM_GPU_IDS:
    GPU_IDS = [int(g) for g in SLURM_GPU_IDS.split(',')]
else:
    GPU_IDS = list(range(len(HW_GPU_LIST)))
GPU_COUNT = len(GPU_IDS)

EXECUTABLE = GPU_EXECUTABLE if GPU_COUNT else CPU_EXECUTABLE


TOP_DIR = os.getcwd()


def parse_args():
    #TODO: add switch to train starting from pretrained weights
    parser = argparse.ArgumentParser()

    parser.add_argument('--o', metavar='DIRECTORY',
                        help='Job output directory')
    parser.add_argument('--in_mics', metavar='MICS_STARFILE',
                        help='Input micrographs .star file.')
    parser.add_argument('--in_coords', metavar='COORDS_STARFILE',
                        help='Input coordinates .star file.')
    parser.add_argument('--in_parts', metavar='PARTS_STARFILE',
                        help='Input particles .star file. (This will override the coordinates/micrographs inputs.)')
    parser.add_argument('--mask_size', type=int, default=200,
                        help='Mask size.  Should be *just* big enough to contain particles.  (Default: 200 pix)')
    parser.add_argument('--warmup', type=int, default=5,
                        help="Warmup epochs. (Default: 5, Ignored for fine_tune.  Don't know what to do if you refine the general model.")
    parser.add_argument('--fine_tune', action='store_true',
                        help='Fine tune initial model. (Default: False)')
    parser.add_argument('-lft', '--layers_fine_tune', type=int, default=2,
                        help='Fine tune layers. (Default: 2)')
    parser.add_argument('--filter', default=0.1,
                        help="Downsampling filter.  Allowed values 0 - 0.5 .  (Default: 0.1 .)")
    parser.add_argument('--weights', default=CRYOLO_PHOSNET_LOCATION,
                        help="Initializaiton weights.  (Default: General PhosaurusNet Model weights.)")
    parser.add_argument('--use_default_weights', action='store_true',
                        help='Initialize network with random weights.  (Default: False.)')
    parser.add_argument('--early', type=int, default=10,
                        help='If the validation loss does not improve after this many iterations, training will be stopped early.  (Default: 10)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of full passes through the data.  (Default: 200)')
    parser.add_argument('--gpu_fraction', type=float, default=1.0,
                        help='Fraction of GPU memory to use.  (Default: 1.0)')
    parser.add_argument('--j', metavar="NUM_CPU", type=int, default=-1,
                        help="Threads. (Default: Use all available)")

    args = parser.parse_args()
    if not (args.in_parts or (args.o and args.in_mics) and args.in_coords):
        print('ERROR: Either particles or input micrographs and coordinates, as well as output directory are required.')
        sys.exit(1)
    return args


def setup_temp_dir(relion_job_dir):
    cryolo_working_directory = os.path.join(relion_job_dir, 'crYOLO')
    shutil.rmtree(cryolo_working_directory, ignore_errors=True)
    os.makedirs(cryolo_working_directory, exist_ok=True)
    os.makedirs(os.path.join(cryolo_working_directory, 'train_box_files'), exist_ok=True)
    os.makedirs(os.path.join(cryolo_working_directory, 'train_image_files'), exist_ok=True)

    return cryolo_working_directory



def get_input_info_from_particles_star_file(starfile_location):
    '''
    Read particles star file

    Return a dictionary of {Micrograph path : [Micrograph names}
    i.e. the keys are the different micrograph containing directories, and the value is a list of microgaph filenames within.
    '''
    ogs = subprocess.check_output(f'relion_star_printtable {starfile_location} data_optics _rlnMicrographOriginalPixelSize',
                                    universal_newlines=True, shell=True).splitlines()
    try:
        assert len(ogs) == 1
        original_pixel_size = float(ogs[0])
    except AssertionError:
        raise NotImplementedError('Only one optics group allowed.')
    lines = subprocess.check_output(f'relion_star_printtable {starfile_location} data_particles _rlnCoordinateX _rlnCoordinateY _rlnImageName _rlnMicrographName',
                                    universal_newlines=True, shell=True)
    micrograph_paths = defaultdict(list)
    px, nx = None, None
    for line in lines.splitlines():
        x, y, part, micrograph = line.split()
        part_index, part_file = part.split('@')
        with mrcfile.open(part_file, permissive=True) as mrc:
            if px is None:
                px = float(mrc.voxel_size.x)
                assert px == float(mrc.voxel_size.y)
                nx = mrc.data.shape[2]
                assert nx == mrc.data.shape[1]
            else:
                assert px == float(mrc.voxel_size.x)
                assert nx == mrc.data.shape[2]
        micrograph_paths[micrograph].append((int(float(x)), int(float(y))))
    return micrograph_paths, nx, px, original_pixel_size


def write_picks_box_file(filename, picks, size):
    with open(filename, 'w') as f:
        for pick in picks:
            f.write(f' {pick[0]:>7} {pick[1]:>7} {size:>6} {size:>6}\n')


def make_input_files(cryolo_working_directory, input_parts_star_file=None,
                    input_mics_star_file=None, input_coords_star_file=None,
                    box_size=None):
    if input_parts_star_file is not None:
        micrograph_paths, nx, px, original_pixel_size = get_input_info_from_particles_star_file(input_parts_star_file)
    else:
        raise NotImplementedError('requires parts star file for now.')

    full_size_box = box_size or int(nx * px/original_pixel_size)
    for micrograph_path, picks in micrograph_paths.items():
        micrograph_name = os.path.basename(micrograph_path)
        os.symlink(os.path.abspath(micrograph_path), os.path.join(cryolo_working_directory, 'train_image_files', micrograph_name))
        box_file_name = os.path.join(cryolo_working_directory, 'train_box_files', micrograph_name)
        box_file_name = os.path.splitext(box_file_name)[0] + '.box'
        write_picks_box_file(box_file_name, picks, full_size_box)


def make_config_file(cryolo_working_directory, cryolo_params):
    config_dict = {'model': {"filter": [0.1, "filtered_tmp/"],
                            "input_size": 1024,
                            "max_box_per_image": 700,
                            },
                  'train': {"train_image_folder": "train_image_files",
                            "train_annot_folder": "train_box_files",
                            "train_times": 1,
                            "batch_size": 4,
                            "learning_rate": 0.0001,
                            "object_scale": 5.0,
                            "no_object_scale": 1.0,
                            "coord_scale": 1.0,
                            "class_scale": 1.0,
                            "pretrained_weights": "cryolo_model.h5",
                            "saved_weights_name": "cryolo_model.h5",
                            "debug": True
                            },
                  'valid': {"valid_image_folder": "",
                            "valid_annot_folder": "",
                            "valid_times": 1
                            },
                  'other': {'log_path': 'logs/'}
                  }
    config_dict['model']["architecture"] = "PhosaurusNet"
    config_dict['model']["anchors"] = [cryolo_params['mask size'],] * 2
    config_dict['model']["filter"][0] = cryolo_params['filter']
    config_dict['train']["nb_epoch"] = cryolo_params['epochs']
    if not cryolo_params['use_default_weights']:
        config_dict['train']["pretrained_weights"] = cryolo_params['weights']
    with open(os.path.join(cryolo_working_directory, 'config_cryolo.json'), 'w') as f:
        json.dump(config_dict, f)


def check_abort_signal(abs_job_dir):
    if os.path.exists(os.path.join(abs_job_dir, 'RELION_JOB_ABORT_NOW')):
        return True
    return False


def run_cryolo(relion_job_dir, cryolo_params):
    #TODO: Re-route output and print more useful info to stdout and stderr.
    abs_job_dir = os.path.abspath(relion_job_dir)
    cryolo_working_directory = os.path.join(relion_job_dir, 'crYOLO')
    try:
        os.chdir(cryolo_working_directory)
        cmd = f"{EXECUTABLE} --conf config_cryolo.json --warmup {cryolo_params['warmup']} --use_multithreading"
        if cryolo_params['fine_tune']:
            cmd += f" --fine_tune --layers_fine_tune {cryolo_params['layers_fine_tune']} "
        cmd += f" --num_cpu {cryolo_params['num_cpu']}"
        cmd += f" --early {cryolo_params['early_stop_patience']}"
        print(f'RUNNING: {cmd}', flush=True)
        proc = subprocess.Popen(cmd, shell=True)
        while proc.poll() is None:
            abort = check_abort_signal(abs_job_dir)
            if abort:
                os.kill(proc.pid, signal.SIGTERM)
                raise Exception('Relion RELION_JOB_ABORT_NOW file seen. Terminating.')
            time.sleep(1)
        if proc.poll():
            raise subprocess.SubprocessError('Train job died.  Exiting.')
    finally:
        os.chdir(TOP_DIR)


def assimilate_results(relion_job_dir):
    cryolo_working_directory = os.path.join(relion_job_dir, 'crYOLO')
    src_model = os.path.abspath(os.path.join(cryolo_working_directory, 'cryolo_model.h5'))
    dest_model = os.path.join(relion_job_dir, 'cryolo_model.h5')
    os.symlink(src_model, dest_model)


if __name__ == '__main__':
    print('Deal with moving model .h5 file and relion communication files')
    # sys.exit()
    try:
        args = parse_args()

        for f in ('RELION_JOB_EXIT_FAILURE', 'RELION_JOB_EXIT_SUCCESS'):
            try: os.remove(f)
            except FileNotFoundError: pass

        cryolo_params = {'weights_location': CRYOLO_PHOSNET_LOCATION,
                         'num_cpu': args.j,
                         'warmup': args.warmup,
                         'mask size': args.mask_size,
                         'gpu_fraction': args.gpu_fraction,
                         'fine_tune': args.fine_tune,
                         'layers_fine_tune': args.layers_fine_tune,
                         'weights': args.weights,
                         'use_default_weights': args.use_default_weights,
                         'early_stop_patience': args.early,
                         'epochs': args.epochs,
                         'filter': args.filter,
                         }

        print('Cryolo Wrapper for Relion v3.1', flush=True)
        print('Written by TJ Ragan (LISCB, University of Leicester)\n', flush=True)

        print('crYOLO Version: 1.6.1', flush=True)
        print('Wrapper Version: 0.3\n', flush=True)

        print(f'Print compute info here.', flush=True)
        print(f"Using {GPU_COUNT} GPUs: {GPU_IDS}.", flush=True)
        print(f'=================', flush=True)

        relion_job_dir = args.o
        input_mics_star_file = args.in_mics
        input_coords_star_file = args.in_coords

        os.makedirs(relion_job_dir, exist_ok=True)
        cryolo_working_directory = setup_temp_dir(relion_job_dir)
        make_input_files(cryolo_working_directory, input_parts_star_file=args.in_parts,
                         input_mics_star_file=None, input_coords_star_file=None,
                         box_size=args.mask_size)
        make_config_file(cryolo_working_directory, cryolo_params)
        run_cryolo(relion_job_dir, cryolo_params)
        assimilate_results(relion_job_dir)
        #TODO: Assimilate and print results.

        Path(os.path.join(relion_job_dir, 'RELION_JOB_EXIT_SUCCESS')).touch()
        print(f'\n Done!\n')

    except Exception as e:
        if os.path.exists(os.path.join(relion_job_dir, 'RELION_JOB_ABORT_NOW')):
            os.remove(os.path.join(relion_job_dir, 'RELION_JOB_ABORT_NOW'))
            Path(os.path.join(relion_job_dir, 'RELION_JOB_EXIT_ABORTED')).touch()
        else:
            Path('RELION_JOB_EXIT_FAILURE').touch()
            raise e

