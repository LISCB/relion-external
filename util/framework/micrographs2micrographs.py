import os
from glob import glob

from util.framework.micrographs_to import MicrographsTo


class Micrographs2Micrographs(MicrographsTo):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._out_ext = '.mrc'


    def write_relion_output(self):
        output_star_name = f'micrographs_{self.name}.star'
        optics_groups = self.input_micrographs_starfile['optics groups']
        input_micrographs = self.input_micrographs_starfile['micrographs']

        output_image_paths = []
        for output_dir in self.relion_path_mapping(self.input_micrographs_starfile).keys():
            output_image_paths += glob(os.path.join(self.relion_job_dir, output_dir, '*.mrc'))

        with open(os.path.join(self.relion_job_dir, output_star_name), 'w') as f:
            f.write('\n# version 30001\n\n')
            f.write('data_optics\n\n')
            f.write('loop_\n')
            f.write('_rlnOpticsGroupName  # 1\n')
            f.write('_rlnOpticsGroup  # 2\n')
            f.write('_rlnVoltage  # 3\n')
            f.write('_rlnSphericalAberration  # 4\n')
            f.write('_rlnMicrographPixelSize  # 5\n')
            for og in optics_groups.values():
                f.write(f"{og.rlnOpticsGroupName} ")
                f.write(f" {og.rlnOpticsGroup:4d} ")
                f.write(f" {og.rlnVoltage:4.3f} ")
                f.write(f" {og.rlnSphericalAberration:2.3f} ")
                f.write(f" {og.rlnMicrographPixelSize:3.5f} ")
                f.write("\n")

            f.write('\n# version 30001\n\n')
            f.write('data_micrographs\n\n')
            f.write('loop_\n')
            f.write('_rlnMicrographName  # 1\n')
            f.write('_rlnOpticsGroup  # 2\n')

            for micrograph in input_micrographs:
                for output_image_path in output_image_paths:
                    output_image_basename = os.path.basename(output_image_path)
                    if os.path.basename(micrograph.rlnMicrographName) == output_image_basename:
                        f.write(f'{output_image_path}  {micrograph.rlnOpticsGroup:3d}\n')
                        break

        with open(os.path.join(self.relion_job_dir, 'RELION_OUTPUT_NODES.star'), 'w') as f:
            f.write('data_output_nodes\n')
            f.write('loop_\n')
            f.write('_rlnPipeLineNodeName #1\n')
            f.write('_rlnPipeLineNodeType #2\n')
            f.write(f'{os.path.join(self.relion_job_dir, output_star_name)}    1\n')
