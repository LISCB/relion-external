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

from util.framework.micrographs_to import MicrographsTo
from util.relion.constants import NODE_MIC_COORDS

class Micrographs2Starfiles(MicrographsTo):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._out_ext = f'_{self.name}.star'
        self.output_star_name = f'coords_suffix_{self.name}.star'
        self._relion_output_node_int = int(NODE_MIC_COORDS)

    def count_done_output_items(self):
        raise NotImplementedError('Need to change the output item counter to star files.')
        potential_micrographs = self.io_mappings.values()
        done_count = 0
        for potential_micrograph in potential_micrographs:
            pth = os.path.join(self.working_top_dir, potential_micrograph)
            if os.path.exists(pth):
                done_count += 1
        return done_count


    def write_relion_output_starfile(self):
        with open(os.path.join(self.relion_job_dir, self.output_star_name), 'w') as f:
            f.write(self.args.in_mics)


    def analyze_output(self, *args, **kwargs):
        pass
