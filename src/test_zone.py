##========== Importing the necessary modules ==========#
import tensorflow, torch, jax
import mlx.core as mx
import random, numpy, matplotlib
from safetensors import safe_open
from accelerate import infer_auto_device_map, init_empty_weights
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoConfig, AutoModelForCausalLM, pipeline, LlamaTokenizer, LlamaForCausalLM
# additional module : sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

##========== Pre-trained models ==========#
# model_path = "facebook/opt-13b"
## OpenLM V2 models
model_path = "openlm-research/open_llama_3b_v2"
# model_path = 'openlm-research/open_llama_7b_v2'
## OpenLM V1 models
# model_path = 'openlm-research/open_llama_3b'
# model_path = 'openlm-research/open_llama_7b'
# model_path = 'openlm-research/open_llama_13b'

##========== Sub-setting of the modules ==========#
## PyTorch
torch.set_default_device("mps")
batch_size = 32

config = AutoConfig.from_pretrained("bigscience/bloom")
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

device_map = infer_auto_device_map(model)


tokenizer = LlamaTokenizer.from_pretrained(model_path, legacy = True)
base_model = LlamaForCausalLM.from_pretrained(
    model_path
)
model.tie_weights()

##========== Data set ==========#
# large_tensor = torch.randn(100000, 100000, device = "meta")




# if __name__ == "__main__":
#     print("runable")
#     
