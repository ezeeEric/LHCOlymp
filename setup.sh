
# ROOT version should be set manually 
# export ALRB_rootVersion=6.14.08-x86_64-centos7-gcc8-opt


# determine path to this script
# http://stackoverflow.com/questions/59895/can-a-bash-script-tell-what-directory-its-stored-in
SOURCE_SETUP="${BASH_SOURCE[0]:-$0}"

DIR_SETUP="$( dirname "$SOURCE_SETUP" )"
while [ -h "$SOURCE_SETUP" ]
do 
  SOURCE_SETUP="$(readlink "$SOURCE_SETUP")"
  [[ $SOURCE_SETUP != /* ]] && SOURCE_SETUP="$DIR_SETUP/$SOURCE_SETUP"
  DIR_SETUP="$( cd -P "$( dirname "$SOURCE_SETUP"  )" && pwd )"
done
DIR_SETUP="$( cd -P "$( dirname "$SOURCE_SETUP" )" && pwd )"


# lsetup root 

export PYTHONPATH=${DIR_SETUP}${PYTHONPATH:+:$PYTHONPATH}
export PATH=${DIR_SETUP}/bin${PATH:+:$PATH}
