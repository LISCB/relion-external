# relion-external
Scripts for running external jobs throught the Relion GUI


This requires several environment variables to be set.  In our facility, these are:  

CRYOLO_BIN_DIR=/net/prog/anaconda3/envs/cryolo/bin  
CRYOLO_CPU_PYTHON=/net/prog/anaconda3/envs/cryolo/bin/python  
CRYOLO_GPU_PYTHON=/net/prog/anaconda3/envs/cryolo-gpu/bin/python  
CRYOLO_GENERAL_MODEL=/net/common/cryolo/gmodel_phosnet_202005_N63_c17.h5  
CRYOLO_GENERAL_NN_MODEL=/net/common/cryolo/gmodel_phosnet_202005_nn_N63_c17.h5  
JANNI_GPU_PYTHON=$CRYOLO_GPU_PYTHON  
JANNI_CPU_PYTHON=$CRYOLO_CPU_PYTHON  
JANNI_DEFAULT_MODEL=/net/common/janni/gmodel_janni_20190703.h5  
TOPAZ_GPU_PYTHON=/net/prog/anaconda3/envs/topaz/bin/python  
TOPAZ_CPU_PYTHON=/net/prog/anaconda3/envs/topaz/bin/python  
TOPAZ_EXECUTABLE=/net/prog/anaconda3/envs/topaz/bin/topaz  

Some of these are optional, but it's best if you're explicit.  
