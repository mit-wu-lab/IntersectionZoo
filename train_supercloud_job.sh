export F=$PWD

export WANDB_API_KEY="<YOUR_KEY>"
export WANDB_MODE="offline"
export LIBSUMO=false

if [ -n "$1" ]
then
    export EXP_NAME=$1
else
    export EXP_NAME="new_untitled_exp"
fi

if [ -n "$2" ]
then
    export KWARGS=$2
else
    export KWARGS="{}"
fi

if [ -n "$3" ]
then
    export TASK_KWARGS=$3
else
    export TASK_KWARGS="{}"
fi

mkdir wd/$EXP_NAME -p

LLsub train_helper_script.sh -s 20 -o wd/$EXP_NAME/output.log