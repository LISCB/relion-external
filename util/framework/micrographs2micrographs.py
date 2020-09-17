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
from glob import glob

from util.framework.micrographs_to import MicrographsTo
from util.relion.constants import NODE_MICS


class Micrographs2Micrographs(MicrographsTo):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._out_ext = '.mrc'
        self._relion_output_node_int = int(NODE_MICS)


    def write_relion_output_starfile(self):
        self.output_star_name = f'micrographs_{self.name}.star'
        optics_groups = self.input_micrographs_starfile['optics groups']
        input_micrographs = self.input_micrographs_starfile['micrographs']

        output_image_paths = []
        for output_dir in self.relion_path_mapping(self.input_micrographs_starfile).keys():
            output_image_paths += glob(os.path.join(self.relion_job_dir, output_dir, '*.mrc'))

        with open(os.path.join(self.relion_job_dir, self.output_star_name), 'w') as f:
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

    # def write_relion_output_nodes(self):
    #     with open(os.path.join(self.relion_job_dir, 'RELION_OUTPUT_NODES.star'), 'w') as f:
    #         f.write('data_output_nodes\n')
    #         f.write('loop_\n')
    #         f.write('_rlnPipeLineNodeName #1\n')
    #         f.write('_rlnPipeLineNodeType #2\n')
    #         f.write(f'{os.path.join(self.relion_job_dir, self.output_star_name)}    {self._relion_output_node_int}\n')
