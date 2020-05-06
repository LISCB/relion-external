#! /net/prog/anaconda3/envs/cryolo/bin/python

from collections import defaultdict
import shutil
import glob
import sys
import os
import argparse
import subprocess
import threading
import time
from pathlib import Path
import signal
from functools import partial


PROGRAM_NAME = 'cryolo'
CPU_EXECUTABLE = '/net/prog/anaconda3/envs/cryolo/bin/cryolo_predict.py'
GPU_EXECUTABLE = '/net/prog/anaconda3/envs/cryolo-gpu/bin/cryolo_predict.py'
CRYOLO_PHOSNET_LOCATION = '/net/common/cryolo/gmodel_phosnet_202002_N63.h5'
CRYOLO_PHOSNET_NN_LOCATION = '/net/common/cryolo/gmodel_phosnet_202003_nn_N63.h5'
JANNI_LOCATION = '/net/common/janni/gmodel_janni_20190703.h5'

SUFFIX_STAR_FILENAME = f'coords_suffix_{PROGRAM_NAME}.star'


gpu_count = len(subprocess.run('lspci | grep "VGA compatible controller: NVIDIA"', shell=True,
                           stdout=subprocess.PIPE).stdout.decode().splitlines())
EXECUTABLE = GPU_EXECUTABLE if gpu_count else CPU_EXECUTABLE


TOP_DIR = os.getcwd()
# MOUSE = r'~~(,_,">'
MOUSE = r'~~( ϵ:>'


class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self,  *args, **kwargs):
        super(StoppableThread, self).__init__(*args, **kwargs)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--o', metavar='DIRECTORY', help='Job output directory')
    parser.add_argument('--in_mics', metavar='STARFILE', help='Input micrographs .star file')
    parser.add_argument('--threshold', type=float, default=0.3, help='Picking threshold.  (Default: 0.3.  Lower means pick more)')
    parser.add_argument('--prediction_batch_size', type=int, default=3, help='Images per batch, lower values will help with memory issues.  (Default: 3)')
    parser.add_argument('--gpu_fraction', type=float, default=1.0, help='Fraction of GPU memory to use.  (Default: 1.0)')
    parser.add_argument('--otf', action='store_true', help='On-The-Fly pre-filtering.  Not currently multi-threaded. (Default: off)')
    parser.add_argument('--overwrite', action='store_true', help='Force complete re-picking.  (default: off)')
    parser.add_argument('--j', metavar="NUM_CPU", type=int, default=-1, help="Threads. (Default: Use all available)")

    args = parser.parse_args()
    if not (args.o and args.in_mics):
        print('ERROR: Input and output are required.')
        sys.exit(1)
    return args


def get_input_micrograph_paths(starfile_location):
    '''
    Read star file

    Return a dictionary of {Micrograph path : [Micrograph names}
    i.e. the keys are the different micrograph containing directories, and the value is a list of microgaph filenames within.
    '''

    #TODO: use default callable and add commandline parameter to specify
    rlnMicrographNames = subprocess.check_output(f'relion_star_printtable {starfile_location} data_micrographs _rlnMicrographName',
                                                universal_newlines=True, shell=True)
    micrograph_paths = defaultdict(list)
    for micrograph_path in rlnMicrographNames.splitlines():
        pth, fname = os.path.split(micrograph_path.strip())
        micrograph_paths[pth].append(fname)
    return micrograph_paths


def setup_temp_dir(relion_job_dir, micrograph_paths, force=False):
    print(' + Running crYOLO on the following micrographs:', flush=True)
    for k,v in micrograph_paths.items():
        relion_micrograph_directory = os.path.basename(k)
        relion_output_directory = os.path.join(relion_job_dir, relion_micrograph_directory)
        cryolo_working_directory = os.path.join(relion_output_directory, 'crYOLO')
        input_symlinks_dir = os.path.join(cryolo_working_directory, 'input')
        log_dir = os.path.join(cryolo_working_directory, 'logs', 'cmdlogs')
        if force:
            shutil.rmtree(relion_output_directory, ignore_errors=True)
        else:
            try:
                lastlog = glob.glob(os.path.join(log_dir, '*.txt'))
                lastlog = sorted(lastlog, key=os.path.getmtime)[-1]
                old_path_suffix = os.path.splitext(lastlog.split('-')[-1])[0]
                shutil.move(cryolo_working_directory, cryolo_working_directory + '-' + old_path_suffix)
            except IndexError:
                pass
        os.makedirs(input_symlinks_dir, exist_ok=True)
        source_directory = os.path.join(TOP_DIR, k)
        for m in v:
            if not force:
                prefix = os.path.splitext(m)[0]
                dest_star = os.path.join(relion_output_directory, prefix + f'_{PROGRAM_NAME}.star')
                if os.path.isfile(dest_star):
                    continue

            src = os.path.join(source_directory, m)
            dest = os.path.join(input_symlinks_dir, m)
            print(f'   * {os.path.join(k,m)}', flush=True)
            try:
                os.symlink(src, dest)
            except FileExistsError:
                pass


def make_config_files(relion_job_dir, micrograph_paths, cryolo_params):
    cryolo_config = '''{
    "model": {
        "architecture": "PhosaurusNet",
        "input_size": 1024,
        "max_box_per_image": 700,
        "filter": [
            0.1,
            "filtered_tmp/"
        ]
    },
    "other": {
        "log_path": "logs/"
    }
    }
    '''.strip()
    for k in micrograph_paths.keys():
        relion_micrograph_directory = os.path.basename(k)
        cryolo_working_directory = os.path.join(relion_job_dir, relion_micrograph_directory, 'crYOLO')
        with open(os.path.join(cryolo_working_directory, 'cryolo_config.json'), 'w') as f:
            f.write(cryolo_config)


def _make_updated_progress_bar(progress_time, estimated_time, frac_done):
    if estimated_time < 60:
        progress = f'{int(progress_time):3d}'
    elif estimated_time < 3600:
        progress = f'{progress_time / 60:.3f}'[:3]
    else:
        progress = f'{progress_time / 3600:.1f}'
    if frac_done < 0.1:
        estimated_time_text = '??? sec'
    else:
        if estimated_time < 60:
            estimated_time_text = f'{estimated_time:3d} sec'
        elif estimated_time < 3600:
            estimated_time_text = f'{estimated_time / 60:.3f}'[:3] + ' min'
        else:
            estimated_time_text = f'{estimated_time / 3600:.1f}' + ' hrs'

    p_char = int(frac_done * 62)
    progress_text = f'\r {progress}/{estimated_time_text} '
    progress_text += '.' * p_char
    progress_text += MOUSE
    progress_text += ' ' * (58 - p_char)
    cheese_truncation = 0
    if p_char > 55:
        cheese_truncation = p_char - 55
    progress_text += '[oo]'[cheese_truncation:]
    return progress_text


def check_abort_signal(abs_job_dir):
    if os.path.exists(os.path.join(abs_job_dir, 'RELION_JOB_ABORT_NOW')):
        return True
    return False


def _update_preproc_progress_bar(kwargs=None):
    if kwargs is None:
        kwargs = {'count': len(os.listdir('input')),
                  'start_preproc_time': time.time(),
                  'initial_preproc_count': None,
                  'prior_preproc_count': 0
                  }
        print(' Preprocessing ...', flush=True)
        print(f' 000/??? sec {MOUSE}                                                          [oo]',
              end='', flush=True)
    output = subprocess.run('find filtered_tmp -name "*.mrc"',
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            shell=True).stdout.decode().splitlines()
    if kwargs['initial_preproc_count'] is None:
        kwargs['initial_preproc_count'] = set(output)
    preproc_count = len(set(output))
    if preproc_count > kwargs['prior_preproc_count']:
        kwargs['prior_preproc_count'] = preproc_count
        frac_preproc = max(len(set(output) - kwargs['initial_preproc_count']) / kwargs['count'], 0.000001)
        progress_time = time.time() - kwargs['start_preproc_time']
        estimated_time = int(progress_time / frac_preproc)
        progress_text = _make_updated_progress_bar(progress_time, estimated_time, frac_preproc)
        print(progress_text, end='', flush=True)
    if preproc_count == kwargs['count']:
        print('')
        return _update_pick_progress_bar
    else:
        return lambda: _update_preproc_progress_bar(kwargs)


def _update_pick_progress_bar(kwargs=None):
    if kwargs is None:
        kwargs = {'count': len(os.listdir('input')),
                  'start_pick_time': time.time(),
                  'initial_picks': None,
                  'prior_pick_count': 0
                  }
        print(' Autopicking ...', flush=True)
        print(f' 000/??? sec {MOUSE}                                                          [oo]',
              end='', flush=True)
    output = subprocess.run('find output/CBOX -name "*.cbox"',
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            shell=True).stdout.decode().splitlines()
    if kwargs['initial_picks'] is None:
        kwargs['initial_picks'] = set(output)
    pick_count = len(set(output))
    if pick_count > kwargs['prior_pick_count']:
        kwargs['prior_pick_count'] = pick_count
        frac_picked = max(len(set(output) - kwargs['initial_picks']) / kwargs['count'], 0.000001)
        progress_time = time.time() - kwargs['start_pick_time']
        estimated_time = int(progress_time / frac_picked)
        progress_text = _make_updated_progress_bar(progress_time, estimated_time, frac_picked)
        print(progress_text, end='', flush=True)
    return lambda: _update_pick_progress_bar(kwargs)


def run_cryolo(**kwargs):
    weights_location = kwargs.get('weights_location', CRYOLO_PHOSNET_LOCATION)
    cryolo_cmd = f'{EXECUTABLE} -c cryolo_config.json --weights {weights_location} --input input --output output'
    for opt in ['num_cpu', 'gpu_fraction', 'prediction_batch_size']:
        value = kwargs.get(opt)
        if value is not None:
            cryolo_cmd += f" --{opt} {value}"
    if kwargs.get('otf'):
        cryolo_cmd += f" --otf"
    cryolo_cmd += ' >cryolo.out 2>cryolo.err'
    return subprocess.Popen(cryolo_cmd, shell=True)


def run_all_cryolo(relion_job_dir, micrograph_paths, num_jobs=1, **kwargs):
    print(' Autopicking with crYOLO ...', flush=True)
    abs_job_dir = os.path.abspath(relion_job_dir)
    cwd = os.getcwd()
    update_progress_bar = _update_preproc_progress_bar

    for k in micrograph_paths.keys():
        relion_micrograph_subdirectory = os.path.basename(k)
        cryolo_working_directory = os.path.join(abs_job_dir, relion_micrograph_subdirectory, 'crYOLO')
        try:
            os.chdir(cryolo_working_directory)
            count = len(os.listdir('input'))
            if count < 1:
                continue
            proc = run_cryolo(**kwargs)
            while proc.poll() is None:
                update_progress_bar = update_progress_bar()
                abort = check_abort_signal(abs_job_dir)
                if abort:
                    os.kill(proc.pid, signal.SIGTERM)
                    raise Exception('Relion RELION_JOB_ABORT_NOW file seen. Terminating.')
                time.sleep(1)
        finally:
            os.chdir(cwd)


def cbox2star(cbox_path, threshold=0):
    #TODO: add size columns and check if relion can handle it.
    picks = []
    sizes = []
    with open(cbox_path) as f:
        for line in f:
            splitline = line.split()
            if float(splitline[4]) >= threshold:
                picks.append(f' {splitline[0]:>7} {splitline[1]:>7} {splitline[4]:>12}  0  0.000000')
                sizes.append((int(splitline[5]), int(splitline[6])))
    loop_body = '\n'.join(picks)
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
    return (star_header + loop_body, sizes)


def assimilate_results(relion_job_dir, micrograph_paths, remove_temp_images=True, threshold=0, **kwargs):
    postfix_ID = ''
    print('\n Assimilating results ...', flush=True)
    sizes = []
    for k in micrograph_paths.keys():
        relion_micrograph_directory = os.path.basename(k)
        relion_output_directory = os.path.join(relion_job_dir, relion_micrograph_directory)
        cryolo_working_directory = os.path.join(relion_output_directory, 'crYOLO')
        for src in sorted(glob.glob(os.path.join(cryolo_working_directory, 'logs', 'cmdlogs', '*.txt')), key=os.path.getmtime):
            shutil.copy2(src, relion_output_directory)
            postfix_ID = os.path.splitext(src)[0].split('command_predict_')[-1]
        for src in glob.glob(os.path.join(cryolo_working_directory, 'output', 'DISTR', '*')):
            shutil.copy2(src, relion_output_directory)
        output_dirs = sorted(glob.glob(os.path.join(cryolo_working_directory+'*')),
                             key=os.path.getmtime)
        output_cbox_files = {}
        for output_dir in output_dirs:
            for cbox_file in glob.glob(os.path.join(output_dir, 'output', 'CBOX', '*.cbox')):
                output_cbox_files[os.path.basename(cbox_file)] = output_dir
        for f_name, d_name in output_cbox_files.items():
            prefix = os.path.splitext(f_name)[0]
            dest = os.path.join(relion_output_directory, prefix + f'_{PROGRAM_NAME}.star')
            with open(dest, 'w') as f:
                star_text, s = cbox2star(os.path.join(d_name, 'output', 'CBOX', f_name), threshold=threshold)
                sizes += s
                f.write(star_text + '\n')
        try:
            shutil.copy2(os.path.join(cryolo_working_directory, 'cryolo.out'),
                         os.path.join(relion_output_directory, f'cryolo_{postfix_ID}.out'))
            shutil.copy2(os.path.join(cryolo_working_directory, 'cryolo.err'),
                         os.path.join(relion_output_directory, f'cryolo_{postfix_ID}.err'))
        except FileNotFoundError:
            pass
        if remove_temp_images:
            shutil.rmtree(os.path.join(cryolo_working_directory, 'filtered_tmp'), ignore_errors=True)

    return postfix_ID, sizes

def write_relion_star_final(relion_job_dir, input_star_file):
    with open(os.path.join(relion_job_dir, SUFFIX_STAR_FILENAME), 'w') as f:
        f.write(f'{input_star_file}\n')

    with open(os.path.join(relion_job_dir, 'RELION_OUTPUT_NODES.star'), 'w') as f:
        f.write('data_output_nodes\n')
        f.write('loop_\n')
        f.write('_rlnPipeLineNodeName #1\n')
        f.write('_rlnPipeLineNodeType #2\n')
        f.write(f'{os.path.join(relion_job_dir, SUFFIX_STAR_FILENAME)}    2\n')


def print_results_summary(relion_job_dir, particle_sizes):
    for k in micrograph_paths.keys():
        relion_micrograph_directory = os.path.basename(k)
        relion_output_directory = os.path.join(relion_job_dir, relion_micrograph_directory)
        wc = subprocess.run(f'wc -l {relion_output_directory}/*.star',
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                shell=True).stdout.decode().splitlines()[:-1]
        kept_particle_count = 0
        for line in wc:
            kept_particle_count += (int(line.strip().split()[0]) - 11)  #11 header lines per star file
        print(f' Total number of particles from {len(wc)} micrographs is {kept_particle_count}')
        if len(particle_sizes) > 0:
            print(f' i.e. on average there were {int(round(kept_particle_count/len(wc)))} particles per micrograph')
            min_sizes = sorted([min(s) for s in particle_sizes])
            max_sizes = sorted([max(s) for s in particle_sizes])
            sizes_len = len(min_sizes)
            fifth_percentile_index = int(sizes_len/20)
            twentyfifth_percentile_index = int(sizes_len/4)
            median_index = int(sizes_len/2)
            print('\n Short axis statistics:')
            print(f'   Absolute minimum: {min_sizes[0]} pix')
            print(f'   5%-Quantile: {min_sizes[fifth_percentile_index]} pix')
            print(f'   25%-Quantile: {min_sizes[twentyfifth_percentile_index]} pix')
            print(f'   Median: {min_sizes[median_index]} pix (Mean: {(sum(min_sizes)/sizes_len):.1f} pix)')
            print(f'   75%-Quantile: {min_sizes[-twentyfifth_percentile_index]} pix')
            print(f'   95%-Quantile: {min_sizes[-fifth_percentile_index]} pix')
            print(f'   Absolute maximum: {min_sizes[-1]} pix')
            print(' Long axis statistics:')
            print(f'   Absolute minimum: {max_sizes[0]} pix')
            print(f'   5%-Quantile: {max_sizes[fifth_percentile_index]} pix')
            print(f'   25%-Quantile: {max_sizes[twentyfifth_percentile_index]} pix')
            print(f'   Median: {max_sizes[median_index]} pix (Mean: {(sum(max_sizes)/sizes_len):.1f} pix)')
            print(f'   75%-Quantile: {max_sizes[-twentyfifth_percentile_index]} pix')
            print(f'   95%-Quantile: {max_sizes[-fifth_percentile_index]} pix')
            print(f'   Absolute maximum: {max_sizes[-1]} pix')


if __name__ == '__main__':
    try:
        args = parse_args()

        for f in ('RELION_JOB_EXIT_FAILURE', 'RELION_JOB_EXIT_SUCCESS'):
            try: os.remove(f)
            except FileNotFoundError: pass

        cryolo_params = {'weights_location': CRYOLO_PHOSNET_LOCATION,
                         'threshold': args.threshold,
                         'num_cpu': args.j,
                         'gpu_fraction': args.gpu_fraction,
                         'prediction_batch_size': args.prediction_batch_size,
                         'otf': args.otf,
                         }

        print('Cryolo Wrapper for Relion v3.1', flush=True)
        print('Written by TJ Ragan (LISCB, University of Leicester)\n', flush=True)

        print(f'Print compute info here.', flush=True)
        print(f'=================', flush=True)

        relion_job_dir = args.o
        input_star_file = args.in_mics

        os.makedirs(relion_job_dir, exist_ok=True)
        micrograph_paths = get_input_micrograph_paths(input_star_file)
        setup_temp_dir(relion_job_dir, micrograph_paths, args.overwrite)
        make_config_files(relion_job_dir, micrograph_paths, cryolo_params)
        run_all_cryolo(relion_job_dir, micrograph_paths, **cryolo_params)

        postfix_ID, particle_sizes, = assimilate_results(relion_job_dir, micrograph_paths, remove_temp_images=True, **cryolo_params)
        print_results_summary(relion_job_dir, particle_sizes)

        write_relion_star_final(relion_job_dir, input_star_file)


        Path(os.path.join(relion_job_dir, 'RELION_JOB_EXIT_SUCCESS')).touch()
        print(f' Done!')

    except Exception as e:
        if os.path.exists(os.path.join(relion_job_dir, 'RELION_JOB_ABORT_NOW')):
            os.remove(os.path.join(relion_job_dir, 'RELION_JOB_ABORT_NOW'))
            Path(os.path.join(relion_job_dir, 'RELION_JOB_EXIT_ABORTED')).touch()
        else:
            Path('RELION_JOB_EXIT_FAILURE').touch()
            raise e

