import os
import time
import json
import torch
import importlib
from tqdm import tqdm
from typing import Any, Dict
import torch.distributed as dist
import torch.multiprocessing as mp
from dataclasses import dataclass, field
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import GenerationConfig, HfArgumentParser


os.environ["NCCL_DEBUG"] = ""
precision_dict = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def setup(DDP_port, rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = DDP_port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def gather_result(data_args, world_size, model_class):
    num_worker, result = world_size, []
    for i in range(num_worker):
        with open(f"{data_args.out_path}/{data_args.mode}_{model_class}_worker_{str(i)}.json", "r") as tf: result.extend(json.load(tf))
        os.remove(f"{data_args.out_path}/{data_args.mode}_{model_class}_worker_{str(i)}.json")
    temp = {each['id']: each for each in result}
    result = list(temp.values())
    with open(f"{data_args.out_path}/{data_args.mode}_{model_class}_result.json", "w") as f: json.dump(result, f, ensure_ascii=False, indent=4)


def get_data(data_path: str):
    with open(data_path, 'r') as f: return json.load(f)


class VLLMDataset(Dataset):
    def __init__(self, data_args):
        super(VLLMDataset, self).__init__()
        self.data_args = data_args
        self.data = get_data(data_args.data_path)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        item = self.data[index]
        return item


def inference_left_pad(input_ids, batch_first, padding_value):
    max_len = max([id.shape[0] for id in input_ids])
    pad_tensor = []
    for id in input_ids:
        pad_number = max_len - id.shape[0]
        id = torch.cat([torch.tensor([padding_value] * pad_number).long(), id])
        pad_tensor.append(id)
    return torch.stack(pad_tensor, dim=0)


def process_in(inputs, model_class, tokenizer):
    input_ids, prompt_lengths = [], []
    for i in inputs:
        if model_class.count('llava') > 0:
            message =  [{"role": "user", "content": [{"type": "text", "text": i}]}]
            prompt = tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", padding=True)
            if model_class.count('llama') > 0: input_id = torch.cat([torch.tensor([tokenizer.tokenizer.bos_token_id]), prompt['input_ids'][0]])
            else: input_id = prompt['input_ids'][0]
            prompt_length = len(tokenizer.decode(input_id, skip_special_tokens=True))
        elif model_class.count('glm') > 0:
            message = [{"role": "user", "content": i}]
            prompt = tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", padding=True)
            input_id = prompt['input_ids'][0]
            prompt_length = len(tokenizer.decode(input_id, skip_special_tokens=True))
        elif model_class.count('idefics') > 0:
            message = [{"role": "user", "content": [{"type": "text", "text": i}]}]
            input_id = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            prompt_length = 0
        elif model_class.count('MiniCPM') > 0:
            message = [{"role": "user", "content": i}]
            prompt = tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", padding=True)
            input_id = prompt['input_ids'][0]
            prompt_length = len(tokenizer.decode(input_id, skip_special_tokens=True))
        elif model_class.count('Phi') > 0:
            message = [{"role": "user", "content": i}]
            prompt = tokenizer.tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", padding=True)
            input_id = prompt['input_ids'][0]
            prompt_length = len(tokenizer.tokenizer.decode(input_id, skip_special_tokens=True))
        elif model_class.count('Qwen') > 0:
            message = [{"role": "user", "content": [{"type": "text", "text": i}]}]
            prompt = tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", padding=True)
            input_id = prompt['input_ids'][0]
            prompt_length = len(tokenizer.decode(input_id, skip_special_tokens=True))
        input_ids.append(input_id)
        prompt_lengths.append(prompt_length)
    if model_class.count('llava') > 0: real_tokenizer = tokenizer.tokenizer
    elif model_class.count('glm') > 0: real_tokenizer = tokenizer
    elif model_class.count('idefics') > 0:
        real_tokenizer = tokenizer.tokenizer
        imgfo = tokenizer(input_ids, None, return_tensors="pt", padding=True)
        input_ids = [imgfo['input_ids'][i] for i in range(imgfo['input_ids'].shape[0])]
        prompt_lengths = [len(tokenizer.decode(each, skip_special_tokens=True)) for each in input_ids]
    elif model_class.count('MiniCPM') > 0: real_tokenizer = tokenizer
    elif model_class.count('Phi') > 0: real_tokenizer = tokenizer.tokenizer
    elif model_class.count('Qwen') > 0: real_tokenizer = tokenizer.tokenizer
    return real_tokenizer, prompt_lengths, input_ids


@dataclass
class VLLMCollactor(object):
    tokenizer: Any
    model_class: str
    mode: str
    def __call__(self, dataset) -> Dict[str, torch.Tensor]:
        labels, inputs = [], []
        for item in dataset:
            if self.mode == 'q': input_text = item['vqa_question']
            elif self.mode == 'o': input_text = item['original_question']
            inputs.append(input_text)
            labels.append(item)
        tokenizer, prompt_length, input_ids = process_in(inputs, self.model_class, self.tokenizer)
        if tokenizer.padding_side == "left": input_ids = inference_left_pad(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        else: input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        return dict(input_ids=input_ids, labels=labels, prompt_length=prompt_length, attention_mask=input_ids.ne(tokenizer.pad_token_id))


@dataclass
class ModelArguments:
    model_path: str = field(default=None)
    trust_remote_code: bool = field(default=False)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    model_class: str = field(default=None)
    mode: str = field(default=None)
    tokenizer_class: str = field(default=None)
    special_args: str = field(default=None)


@dataclass
class InferenceArguments:
    batch_size: int = field(default=1)
    precision: str = field(default="fp16")


@dataclass
class TopArguments:
    DDP: bool = field(default=False)
    DDP_port: str = field(default="12345")
    world_size: int = field(default=1)


def main(name):
    parser = HfArgumentParser((ModelArguments, DataArguments, InferenceArguments, TopArguments))
    file = f'configs/backbone/{name}.yaml'
    model_args, data_args, infer_args, top_args = parser.parse_yaml_file(file)
    datatype = data_args.data_path
    data_args.data_path = f'data/{datatype}/test.json'
    data_args.out_path = f'result/{datatype}/backbone'
    model_class = model_args.model_path.split('/')[-1]
    if not os.path.exists(data_args.out_path): os.makedirs(data_args.out_path)
    world_size = top_args.world_size
    start = time.time()
    if top_args.DDP:
        mp.spawn(infer, args=(world_size, model_args, data_args, infer_args, top_args), nprocs=world_size)
        gather_result(data_args, world_size, model_class)
    else: infer(0, world_size, model_args, data_args, infer_args, top_args)
    end = time.time()
    print(f"Inference took {end - start:8.4f} seconds.")


def infer(rank, world_size, model_args, data_args, infer_args, top_args):
    this_rank_gpu_index = rank

    if top_args.DDP:
        torch.cuda.set_device(this_rank_gpu_index)
        setup(top_args.DDP_port, rank, world_size)

    device = torch.device("cuda:" + str(this_rank_gpu_index) if torch.cuda.is_available() else "cpu")
    torch_dtype = precision_dict[infer_args.precision]
    transformers = importlib.import_module('transformers')
    tokenizer_module = getattr(transformers, data_args.tokenizer_class)
    model_module = getattr(transformers, data_args.model_class)
    
    model_class = model_args.model_path.split('/')[-1] 
    trust_remote_code = True if model_args.trust_remote_code is True else None
    if model_class.count('MiniCPM') > 0:
        tokenizer = tokenizer_module.from_pretrained(model_args.model_path, padding_side="left", trust_remote_code=trust_remote_code)
        model = model_module.from_pretrained(model_args.model_path, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code, attn_implementation='sdpa')
    elif model_class.count('Phi') > 0:
        tokenizer = tokenizer_module.from_pretrained(model_args.model_path, trust_remote_code=trust_remote_code, num_crops=16)
        model = model_module.from_pretrained(model_args.model_path, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code, attn_implementation='flash_attention_2')
    elif model_class.count('Qwen') > 0:
        tokenizer = tokenizer_module.from_pretrained(model_args.model_path, trust_remote_code=trust_remote_code)
        model = model_module.from_pretrained(model_args.model_path, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code, attn_implementation='flash_attention_2')
    else:
        tokenizer = tokenizer_module.from_pretrained(model_args.model_path, padding_side="left", trust_remote_code=trust_remote_code)
        model = model_module.from_pretrained(model_args.model_path, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code)

    if model_class == 'glm-4v-9b': model.config.vision_config['image_size'] = 2 * model.config.vision_config['patch_size']
    elif model_class == 'MiniCPM-V-2_6': model = model.llm
    
    try:
        if tokenizer.tokenizer.pad_token_id is None: tokenizer.tokenizer.pad_token_id = tokenizer.tokenizer.eos_token_id
        generation_config = GenerationConfig(do_sample=False, max_new_tokens=512, pad_token_id=tokenizer.tokenizer.pad_token_id, eos_token_id=tokenizer.tokenizer.eos_token_id)
    except:
        if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
        generation_config = GenerationConfig(do_sample=False, max_new_tokens=512, pad_token_id = tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)

    model.eval()
    if top_args.DDP: model.to(device)
    dataset = VLLMDataset(data_args)
    collate_fn = VLLMCollactor(tokenizer=tokenizer, model_class=model_class, mode=data_args.mode)
    kwargs = dict(batch_size=infer_args.batch_size, collate_fn=collate_fn, pin_memory=True)
   
    if top_args.DDP: 
        kwargs['sampler'] = DistributedSampler(dataset, world_size, rank, False, drop_last=False)
        file = f"{data_args.out_path}/{data_args.mode}_{model_class}_worker_{str(rank)}.json"
    else: kwargs['num_workers'], file = 1, f"{data_args.out_path}/{data_args.mode}_{model_class}_result.json"
    dataloader, rf = DataLoader(dataset, **kwargs), open(file, "w")
    prog_bar = tqdm(dataloader, total=len(dataloader), desc="worker_" + str(rank)) if rank == 0 else dataloader
    result = []
    for test_batch in prog_bar:   
        with torch.inference_mode():
            outputs = model.generate(
                input_ids = test_batch['input_ids'].to(device),
                attention_mask = test_batch['attention_mask'].to(device),
                generation_config = generation_config
            )
        labels, prompt_length, predict = test_batch['labels'], test_batch['prompt_length'], tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for l, plen, p in zip(labels, prompt_length, predict): 
            l['pred'] = p[plen:].strip()
            result.append(l)
    json.dump(result, rf, ensure_ascii=False, indent=4)
     
        
if __name__ == "__main__":
    main("qwen")
