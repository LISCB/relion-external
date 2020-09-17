#!/usr/bin/env bash

###############################################################################
#  This file is part of the relion-external suite that allows integration of   #
#  arbitrary software into Relion 3.1.                                        #
#                                                                             #
#  Copyright (C) 2020 Univeristy of Leicester                                 #
#                                                                             #
#  This program is free software: you can redistribute it and/or modify       #
#  it under the terms of the GNU General Public License as published by       #
#  the Free Software Foundation, either version 3 of the License, or          #
#  (at your option) any later version.                                        #
#                                                                             #
#  This program is distributed in the hope that it will be useful,            #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of             #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              #
#  GNU General Public License for more details.                               #
#                                                                             #
#  You should have received a copy of the GNU General Public License          #
#  along with this program.  If not, see www.gnu.org/licenses/gpl-3.0.html.   #
#                                                                             #
#  Written by TJ Ragan (tj.ragan@leicester.ac.uk),                            #
#  Leicester Institute of Structural and Chemical Biology (LISCB)             #
###############################################################################


if [ ! -z "${JANNI_PYTHON}" ]; then
    PYTHON=${JANNI_PYTHON}
elif [ ! -z "${JANNI_GPU_PYTHON}" ]; then
    PYTHON=${JANNI_GPU_PYTHON}
elif [ ! -z "${JANNI_CPU_PYTHON}" ]; then
    PYTHON=${JANNI_CPU_PYTHON}
else
    PYTHON='python'
fi

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
SCRIPT_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"

PYTHONPATH=$SCRIPT_DIR:$PYTHONPATH $PYTHON $SCRIPT_DIR/JANNI/denoise.py "$@"
