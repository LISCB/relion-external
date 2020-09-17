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
from subprocess import Popen, PIPE, CalledProcessError
import signal
import time
from glob import glob
import shutil

from util.framework.particles_to import ParticlesTo


GPU_PYTHON = os.environ.get('TOPAZ_GPU_PYTHON', os.environ.get('TOPAZ_PYTHON', shutil.which('python3')))
TOPAZ_EXECUTABLE = os.environ.get('TOPAZ_EXECUTABLE', shutil.which('topaz'))
TOPAZ = ' '.join((GPU_PYTHON, TOPAZ_EXECUTABLE))

class Topaz_Pick_Trainer(ParticlesTo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,
                         name='topaz-pick_train',
                         worker_setup_function=self.setup,
                         worker_run_function=self.run_worker,
                         # extra_text=self.extra_text,
                         worker_output_analysis_function=self.worker_output_analysis_function,
                         worker_cleanup_function=self.cleanup,
                         preproc_dir=True,
                         parallelizable=False,
                         )


        preproc_args = self.parser.add_argument_group('Preprocessing Parameters')
        preproc_args.add_argument('-s', '--scale', default=8, type=int,
                                  help='downsample images by this factor.  See topaz webpage for appropriate choice (default: 8)')
        preproc_args.add_argument('--affine', action='store_true',
                                  help='use standard normalization (x-mu)/std of whole image rather than GMM normalization')
        # preproc_args.add_argument('--keep_preproc', action='store_true',
        #                        help='Keep the pre-processed micrographs (Default: Off.)')

        train_args = self.parser.add_argument_group('Training Parameters')
        required_train_args = train_args.add_mutually_exclusive_group(required=True)
        required_train_args.add_argument('-n', '--num-particles', type=int,
                               help='Expected average number of particles per micrograph.  (Either this or PI must be set.)')
        required_train_args.add_argument('--pi', type=float,
                               help='parameter specifying fraction of data that is expected to be positive.  (Either this or num-particles must be set.)')
        train_args.add_argument('--num-epochs', default=10, type=int,
                               help='maximum number of training epochs (Default: 10)')
        train_args.add_argument('--epoch-size', default=1000, type=int,
                               help='number of parameter updates per epoch (Default: 1000)')
        train_args.add_argument('-r', '--radius', default=3, type=int,
                               help='pixel radius around particle centers to consider positive (Default: 3)')
        train_args.add_argument('--method', default='GE-binomial', choices=['PN', 'GE-KL', 'GE-binomial' ,'PU'],
                               help='objective function to use for learning the region classifier (default: GE-binomial)')
        train_args.add_argument('--slack', default=-1, type=float,
                               help='weight on GE penalty (default: 10 for GE-KL, 1 for GE-binomial)')  #TODO: fix defaults in runner
        train_args.add_argument('--autoencoder', default=0, type=float,
                               help='option to augment method with autoencoder. weight on reconstruction error (default: 0)')
        train_args.add_argument('--l2', default=0, type=float,
                               help='l2 regularizer on the model parameters (default: 0)')
        train_args.add_argument('--learning-rate', default=0.0002, type=float,
                               help='learning rate for the optimizer (default: 0.0002)')
        preproc_args.add_argument('--natural', action='store_true',
                               help='sample unbiasedly from the data to form minibatches rather than sampling particles and not particles at' \
                                    'ratio given by minibatch-balance parameter (Default: off.)')
        train_args.add_argument('--minibatch-size', default=256, type=int,
                               help='number of data points per minibatch (default: 256)')
        train_args.add_argument('--minibatch-balance', default=0.0625, type=float,
                               help='fraction of minibatch that is positive data points (default: 0.0625)')
        train_args.add_argument('-m', '--model', default='resnet16',
                               help='path to trained subimage classifier, or pretrained network name.' \
                                    ' Available pretrained networks are: resnet15, resnet8, conv127, conv63, conv31.  (Default: pretrained resnet16)')
        train_args.add_argument('--no-pretrained', action='store_true',
                               help="Don't initialize model with pretrained weights (if available). (Default: Use pretrained weights.)")
        train_args.add_argument('--units', default=32, type=int,
                               help='number of units model parameter (default: 32)')
        train_args.add_argument('--dropout', default=0, type=float,
                               help='dropout rate model parameter(default: 0.0)')
        train_args.add_argument('--bn', default='on', choices=['on', 'off'],
                               help='use batch norm in the model (default: on)')
        train_args.add_argument('--pooling', default=None,
                               help='pooling method to use (default: none)')
        train_args.add_argument('--unit-scaling', default=2, type=int,
                               help='scale the number of units up by this factor every pool/stride layer (default: 2)')
        train_args.add_argument('--ngf', default=32, type=int,
                               help='scaled number of units per layer in generative model, only used if autoencoder > 0 (default: 32)')
        train_args.add_argument('--test-batch-size', default=1, type=int,
                               help='batch size for calculating test set statistics (default: 1')

        xval_args = self.parser.add_argument_group('Cross Validation')
        xval_args.add_argument('-k', '--k-fold', default=-1, type=int,
                               help='option to split the training set into K folds for cross validation.  (Default: No cross validation.)')
        xval_args.add_argument('--fold', default=0, type=int,
                               help='when using K-fold cross validation, sets which fold is used as the heldout test set. (Default: 0.)')
        xval_args.add_argument('--cross-validation-seed', default=42, type=int,
                               help='option to split the training set into K folds for cross validation.  (Default: 42.)')

        self.extra_text = 'Topaz Picker v0.2.4\n' \
                          'Commercial and non-commercial use allowed.\n' \
                          'Citation:\n' \
                          'Bepler, T., Morin, A., Brasch, J., Shapiro, L., Noble, A.J., Berger, B. (2019).\n' \
                          'Positive-unlabeled convolutional neural networks for particle picking in cryo-electron micrographs. Nature Methods.\n' \
                          'https://doi.org/10.1038/s41592-019-0575-8'


    def setup(self, job_object):
        try:
            assert len(job_object.input_particles_starfile['optics groups']) == 1
        except AssertionError:
            raise NotImplementedError('Training only supports one optics group right now.')

        micrographs = self.get_micrographs_in_particles_starfile(job_object.input_particles_starfile)

        input_symlink_target_dir = os.path.join(job_object.working_top_dir, 'worker_0', 'input')

        for m in micrographs:
            fname = os.path.basename(m)
            abs_m_path = os.path.abspath(m)
            src = os.path.relpath(abs_m_path, input_symlink_target_dir)
            dest = os.path.join(input_symlink_target_dir, fname)
            os.symlink(src, dest)

        self.particles_star_to_topaz(self.input_particles_starfile,
                                     os.path.join(job_object.working_top_dir, 'worker_0', 'from_star.txt'))
        return job_object.args.num_epochs, len(micrographs)


    # def count_done_preproc_items(self, job_object):
    #     try:
    #         mrc_files = check_output(f'find {os.path.join(job_object.working_top_dir, "worker*", "preproc")} -name "*.mrc"',
    #                                  stderr=PIPE, shell=True).splitlines()
    #         return len(mrc_files)
    #     except CalledProcessError:
    #         return 0


    def count_done_output_items(self, job_object):
        try:
            epoch_save_files = glob(os.path.join(job_object.working_top_dir, "worker*", "output", "model_epoch*.sav"))
            return len(epoch_save_files)
        except CalledProcessError:
            return 0


    def _monitor_sub_job_progress(self, slaves, total_count,
                                  start_time=None, previous_done_count=None,
                                  preproc=False):
        if previous_done_count is None:
            print(self.make_progress_bar(), flush=True, end='')
            previous_done_count = 0
        for i, proc in enumerate(slaves):
            poll = proc.poll()
            if (poll is not None) and (poll > 0):
                raise Exception('External job failed. Exiting.')
        abort = self.check_abort_signal()
        if abort:
            for proc in slaves:
                os.kill(proc.pid, signal.SIGTERM)
            raise Exception('Relion RELION_JOB_ABORT_NOW file seen. Terminating.')

        if preproc:
            done_count = self.count_done_preproc_items(self)
        else:
            done_count = self.count_done_output_items(self)

        if done_count > previous_done_count:
            print(self.make_progress_bar(start_time, total_count, done_count), flush=True, end='')
        elif done_count == 0:
            start_time = start_time or time.time()
            print(self.make_progress_bar(start_time, total_count, done_count), flush=True, end='')
        return {'start_time': start_time, 'previous_done_count': done_count}


    @staticmethod
    def particles_star_to_topaz(starfile, output_path):
        with open(output_path, 'w') as f:
            f.write('image_name\tx_coord\ty_coord\n')
            for p in starfile['particles']:
                line = f'{os.path.splitext(os.path.basename(p.rlnMicrographName))[0]}'
                line += f'\t{int(p.rlnCoordinateX)}\t{int(p.rlnCoordinateY)}\n'
                f.write(line)


    def run_worker(self, job_object, **kwargs):

        parsed_args = self.args
        env = kwargs.get('env')

        scale = parsed_args.scale


        convert_cmd = f'{TOPAZ} convert --down-scale {scale}'
        convert_cmd += f' --output converted.txt from_star.txt'
        convert_proc = Popen(convert_cmd, shell=True, env=env,
                             stdout=PIPE, stderr=PIPE
                             )
        preproc_cmd = f'{TOPAZ} preprocess --num-workers 8 --scale {scale}'
        preproc_cmd += f' --destdir preproc/ input/*.mrc'
        train_cmd = f'{TOPAZ} train '  #TODO: --num-workers and --num-threads
        for arg in vars(parsed_args):
            if arg not in ['o', 'in_parts', 'in_coords', 'in_mics',
                           'j', 'cache', 'keep_preproc',
                           'no_pretrained', 'num_particles', 'pi',
                           'affine', 'natural', 'scale']:
                train_cmd += f'--{arg.replace("_", "-")} {getattr(parsed_args, arg)} '
            elif arg in ['no_pretrained', 'natural']:
                if getattr(parsed_args, arg):
                    train_cmd += f'--{arg.replace("_", "-")} '
        if parsed_args.num_particles:
            train_cmd += f' --num-particles {parsed_args.num_particles}'
        else:
            train_cmd += f' --pi {parsed_args.pi}'
        train_cmd += f' --train-images preproc'
        train_cmd += f' --train-targets converted.txt'
        train_cmd += f' --save-prefix output/model'
        train_cmd += f' -o output/model_training.txt'
        train_cmd += f' >topaz.out 2>topaz.err'
        proc = Popen(preproc_cmd + ' && sleep 1 && ' + train_cmd, shell=True, env=env,
                     stdout=PIPE, stderr=PIPE,
                     )
        while convert_proc.poll() is None:  # Potential race condition, but probably never happens
            time.sleep(1)
        return proc


    def recover_output_files(self):
        epoch_files = sorted(glob(os.path.join(self.working_top_dir, "worker_0/output/model_epoch*.sav")),
                             key=os.path.getmtime)
        shutil.copy2(epoch_files[-1], self.relion_job_dir)
        shutil.copy2(os.path.join(self.working_top_dir, "worker_0/output/model_training.txt"), self.relion_job_dir)


    def worker_output_analysis_function(self, job_object):
        model_files = sorted(glob(os.path.join(self.relion_job_dir, 'model_epoch*.sav')),
                             key=os.path.getmtime)
        model_file = os.path.basename(model_files[-1])
        print(f'model saved as: {model_file}')


if __name__ == '__main__':
    job = Topaz_Pick_Trainer()
    # print('*** Handle Continue!!! ***', file=sys.stderr)
    job.run()
