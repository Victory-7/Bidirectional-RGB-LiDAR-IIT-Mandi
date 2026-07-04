conda deactivate 

eval "$(conda shell.bash hook)"

conda activate /home/teaching/Suhani/envs/suhani

export PROJECT_ROOT="/home/teaching/Suhani/Projects/suhani"

export HF_HOME="/home/teaching/Suhani/hf_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$HF_HOME/torch"

mkdir -p "$HF_HOME"/{hub,datasets,transformers,torch}

cd "$PROJECT_ROOT"

cd /home/teaching/Suhani/project 

echo "=================================="
echo "SUHANI Environment Activated"
echo "=================================="
echo "Env: $CONDA_PREFIX"
echo "Python: $(which python)"
echo "Project: $(pwd)"
echo "HF Cache: $HF_HOME"
echo "=================================="

