import os

from util.framework.micrographs_to import MicrographsTo


class Micrographs2Starfiles(MicrographsTo):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._out_ext = f'_{self.name}.star'


    def count_done_output_items(self):
        raise NotImplementedError('Need to change the output item counter to star files.')
        potential_micrographs = self.io_mappings.values()
        done_count = 0
        for potential_micrograph in potential_micrographs:
            pth = os.path.join(self.working_top_dir, potential_micrograph)
            if os.path.exists(pth):
                done_count += 1
        return done_count


    def write_relion_output(self):
        output_star_name = f'coords_suffix_{self.name}.star'
        with open(os.path.join(self.relion_job_dir, output_star_name), 'w') as f:
            f.write(self.args.in_mics)

        with open(os.path.join(self.relion_job_dir, 'RELION_OUTPUT_NODES.star'), 'w') as f:
            f.write('data_output_nodes\n')
            f.write('loop_\n')
            f.write('_rlnPipeLineNodeName #1\n')
            f.write('_rlnPipeLineNodeType #2\n')
            f.write(f'{os.path.join(self.relion_job_dir, output_star_name)}    2\n')


    def analyze_output(self, *args, **kwargs):
        pass
