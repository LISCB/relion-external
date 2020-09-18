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


# from Relion/src/pipeline_jobs.h
NODE_NONE = -1  # No output node (not actually in Relion - this will not create a RELION_OUTPUT_NODES.star file)
NODE_MOVIES = 0  # 2D micrograph movie(s), e.g. Falcon001_movie.mrcs or micrograph_movies.star
NODE_MICS = 1   # 2D micrograph(s), possibly with CTF information as well, e.g. Falcon001.mrc or micrographs.star
NODE_MIC_COORDS = 2  # Suffix for particle coordinates in micrographs (e.g. autopick.star or .box)
NODE_PART_DATA = 3  # A metadata (STAR) file with particles (e.g. particles.star or run1_data.star)
NODE_MOVIE_DATA = 4  # A metadata (STAR) file with particle movie-frames (e.g. particles_movie.star or run1_ct27_data.star)
NODE_2DREFS = 5  # A STAR file with one or multiple 2D references, e.g. autopick_references.star
NODE_3DREF = 6  # A single 3D-reference, e.g. map.mrc
NODE_MASK = 7  # 3D mask, e.g. mask.mrc or masks.star
NODE_MODEL = 8  # A model STAR-file for class selection
NODE_OPTIMISER = 9  # An optimiser STAR-file for job continuation
NODE_HALFMAP = 10  # Unfiltered half-maps from 3D auto-refine, e.g. run1_half?_class001_unfil.mrc
NODE_FINALMAP = 11  # Sharpened final map from post-processing (cannot be used as input)
NODE_RESMAP = 12  # Resmap with local resolution (cannot be used as input)
NODE_PDF_LOGFILE = 13  # PDF logfile
NODE_POST = 14  # Postprocess STAR file (with FSC curve, unfil half-maps, masks etc in it: used by Jasenko's programs
NODE_POLISH_PARAMS = 15  # Txt file with optimal parameters for Bayesian polishing
