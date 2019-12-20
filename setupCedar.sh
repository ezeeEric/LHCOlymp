#export ALRB_CONT_POSTSETUP="lsetup 'views LCG_95apython3 x86_64-centos7-gcc7-opt' && source venv_lhcolymp/bin/activate  && export PYTHONPATH=$PYTHONPATH:$PWD && echo 'SETUP DONE'"
export ALRB_CONT_POSTSETUP="lsetup 'views LCG_95apython3 x86_64-centos7-gcc7-opt' && source venv_lhcolymp/bin/activate  && echo 'SETUP DONE'"
setupATLAS -c centos7
#pip install pyjet,tables
