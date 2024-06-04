# LLaMA model with KIVI
import warnings
warnings.filterwarnings("ignore")
import torch
import random
from models.chatglm_kivi import ChatGLMForConditionalGeneration
# from models.chatglm_original import ChatGLMForConditionalGeneration
from transformers import LlamaConfig, AutoTokenizer
from datasets import load_dataset
from models.chatglm_config import ChatGLMConfig

from torch.profiler import profile, record_function, ProfilerActivity
# For reproducibility
random.seed(0)
torch.manual_seed(0)

pretrained_model_name_or_path = "/mnt/f/models/chatglm3-6b-128k"
config = ChatGLMConfig.from_pretrained(pretrained_model_name_or_path)

config.k_bits = 2 # KiVi currently support 2/4 K/V bits
config.v_bits = 2
config.group_size = 32 
config.residual_length = 32 # corresponding to the number of recent fp16 tokens
CACHE_DIR = "./"

model = ChatGLMForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    config=config,
    cache_dir=CACHE_DIR,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
).cuda()
device = torch.device("cuda")
torch.cuda.reset_peak_memory_stats(device=device)
init_memory = torch.cuda.max_memory_allocated(device=device)
print(torch.cuda.max_memory_allocated(device=torch.device("cuda"))/ 1024 / 1024)

enc = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path, 
    use_fast=False, 
    trust_remote_code=True)

dataset = load_dataset('gsm8k', 'main')

prompt = ''
for i in range(5):
    prompt += 'Question: ' + dataset['train'][i]['question'] + '\nAnswer: ' + dataset['train'][i]['answer'] + '\n'
prompt += "Question: John takes care of 10 dogs. Each dog takes .5 hours a day to walk and take care of their business. How many hours a week does he spend taking care of dogs?"
inputs = enc(prompt, return_tensors="pt").input_ids.cuda()

output = model.generate(inputs, max_new_tokens=96)

# print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

config_str = f"# prompt tokens: {inputs.shape[1]}, K bit: {config.k_bits}, v_bits: {config.v_bits}, group_size: {config.group_size}, residual_length: {config.residual_length}"

peak_generation_memory = torch.cuda.max_memory_allocated(device=device) 

print(prompt + "\n" + "=" * 10 + f'\n{config_str}\n' + "=" * 10 + "\nKiVi Output:")
print(enc.decode(output[0].tolist()[inputs.shape[1]:], skip_special_tokens=True))
