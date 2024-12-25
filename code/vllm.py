import os
import time
import json
import torch
import importlib
from PIL import Image
from tqdm import tqdm
from typing import Any, Dict
import torch.distributed as dist
import torch.multiprocessing as mp
from dataclasses import dataclass, field
from qwen_vl_utils import process_vision_info
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


def process_in(inputs, urls, model_class, tokenizer, device):
    input_ids, prompt_lengths, images = [], [], []
    for i, u in zip(inputs, urls):
        if model_class.count('llava') > 0:
            message =  [{"role": "user", "content": [{"type": "text", "text": i}, {"type": "image"}]}]
            prompt = tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", padding=True)
            if model_class.count('llama') > 0: input_id = torch.cat([torch.tensor([tokenizer.tokenizer.bos_token_id]), prompt['input_ids'][0]])
            else: input_id = prompt['input_ids'][0]
            prompt_length = len(tokenizer.decode(input_id, skip_special_tokens=True))
            image = u
        elif model_class.count('glm') > 0:
            message = [{"role": "user", "image": u.convert("RGB"), "content": i}]
            prompt = tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", padding=True)
            input_id = prompt['input_ids'][0]
            image = prompt['images'][0]
            prompt_length = len(tokenizer.decode(input_id, skip_special_tokens=True))
        elif model_class.count('idefics') > 0:
            message = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": i}]}]
            input_id = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            image, prompt_length = [u], 0
        elif model_class.count('MiniCPM') > 0:
            message = [{"role": "user", "content": f'(<image>./</image>)\n{i}'}]
            input_id = tokenizer.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            image, prompt_length = [u.convert("RGB")], 0
        elif model_class.count('Phi') > 0:
            message = [{"role": "user", "content": f'<|image_1|>\n{i}'}]
            input_id = tokenizer.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            image, prompt_length = [u], 0
        elif model_class.count('Qwen') > 0:
            message = [{"role": "user", "content": [{"type": "text", "text": i}, {"type": "image", "image": u}]}]
            input_id = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            image, prompt_length = message, 0
        input_ids.append(input_id)
        prompt_lengths.append(prompt_length)
        images.append(image)
    
    if model_class.count('llava') > 0:
        imgfo = tokenizer.image_processor(images, return_tensors="pt")
        pic_kwargs = {'pixel_values': imgfo['pixel_values'], 'image_sizes': imgfo['image_sizes']}
        real_tokenizer = tokenizer.tokenizer
    elif model_class.count('glm') > 0: 
        real_tokenizer = tokenizer
        pic_kwargs = {'images': torch.stack(images)}
    elif model_class.count('idefics') > 0:
        real_tokenizer = tokenizer.tokenizer
        imgfo = tokenizer(input_ids, images, return_tensors="pt", padding=True)
        input_ids = [imgfo['input_ids'][i] for i in range(imgfo['input_ids'].shape[0])]
        prompt_lengths = [len(tokenizer.decode(each, skip_special_tokens=True)) for each in input_ids]
        pic_kwargs = {'pixel_values': imgfo['pixel_values'], 'pixel_attention_mask': imgfo['pixel_attention_mask']}
    elif model_class.count('MiniCPM') > 0:
        real_tokenizer = None
        inputs = tokenizer(input_ids, images, return_tensors="pt", max_inp_length=8192).to(device)
        input_ids = inputs['input_ids']
        pic_kwargs = {'pixel_values': inputs['pixel_values'], 'tgt_sizes': inputs['tgt_sizes'], 'image_bound': inputs['image_bound']}
        pic_kwargs['attention_mask'] = inputs['attention_mask']
    elif model_class.count('Phi') > 0:
        real_tokenizer = None
        inputs = tokenizer(input_ids[0], images[0], return_tensors="pt")
        input_ids, pixel_values, image_sizes = inputs['input_ids'], inputs['pixel_values'], inputs['image_sizes']
        pic_kwargs = {'image_sizes': image_sizes, 'pixel_values': pixel_values, 'attention_mask': inputs['attention_mask']}
    elif model_class.count('Qwen') > 0:
        real_tokenizer = None
        image_inputs, video_inputs = process_vision_info(images)
        inputs = tokenizer(text=input_ids, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        input_ids, pixel_values, image_grid_thw, attention_mask = inputs['input_ids'], inputs['pixel_values'], inputs['image_grid_thw'], inputs['attention_mask']
        prompt_lengths = [len(tokenizer.decode(each, skip_special_tokens=True, clean_up_tokenization_spaces=False)) for each in input_ids]
        pic_kwargs = {'image_grid_thw': image_grid_thw, 'pixel_values': pixel_values, 'attention_mask': attention_mask}
    return real_tokenizer, prompt_lengths, input_ids, pic_kwargs


@dataclass
class VLLMCollactor(object):
    tokenizer: Any
    model_class: str
    device: Any
    mode: str
    def __call__(self, dataset) -> Dict[str, torch.Tensor]:
        labels, inputs, urls = [], [], []
        for item in dataset:
            labels.append(item)
            if self.model_class.count('qwen') > 0: urls.append(item['url'])
            else: urls.append(Image.open(item['url']))
            if self.mode == 'q': inputs.append(item['vqa_question'])
            elif self.mode == 'e': inputs.append(item['entity_question'])
            elif self.mode.startswith('q'):
                information = ""
                if self.mode == 'qk1': knowledge = [item['one_knowledge']]
                elif self.mode == 'qk3': knowledge = item['three_knowledge']
                elif self.mode == 'qkr': knowledge = item['rag_knowledge']
                elif self.mode == 'qr1': knowledge = item['advice'][:1]
                elif self.mode == 'qr3': knowledge = item['advice'][:3]
                elif self.mode == 'qr5': knowledge = item['advice'][:5]
                elif self.mode == 'qrx': knowledge = item['advice'][:10]
                for i, k in enumerate(knowledge): information += f' {i + 1}. {k}\n'
                prompt = f"Here are some pieces of information that might be helpful:\n{information}Now answer the question below:\n {item['vqa_question']}"
                inputs.append(prompt)
        tokenizer, prompt_length, input_ids, pic_kwargs = process_in(inputs, urls, self.model_class, self.tokenizer, self.device)
        if tokenizer is None: 
            attention_mask = pic_kwargs.pop('attention_mask')
        else:
            if tokenizer.padding_side == "left": 
                input_ids = inference_left_pad(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
            elif tokenizer.padding_side == "right":
                input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
            attention_mask = input_ids.ne(tokenizer.pad_token_id)
        return dict(input_ids=input_ids, labels=labels, pic_kwargs=pic_kwargs, prompt_length=prompt_length, attention_mask=attention_mask)

   

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
    file = f'configs/vllm/{name}.yaml'
    model_args, data_args, infer_args, top_args = parser.parse_yaml_file(file)
    datatype = data_args.data_path
    data_args.data_path = f'data/{datatype}/test.json'
    data_args.out_path = f'result/{datatype}/vllm'
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
    
    trust_remote_code = True if model_args.trust_remote_code is True else None
    model_class = model_args.model_path.split('/')[-1]
    if model_class.count('MiniCPM') > 0:
        tokenizer = tokenizer_module.from_pretrained(model_args.model_path, trust_remote_code=trust_remote_code)
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
    try:
        if tokenizer.tokenizer.pad_token_id is None: tokenizer.tokenizer.pad_token_id = tokenizer.tokenizer.eos_token_id
        generation_config = GenerationConfig(do_sample=False, max_new_tokens=256, pad_token_id=tokenizer.tokenizer.pad_token_id, eos_token_id=tokenizer.tokenizer.eos_token_id)
    except:
        if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
        generation_config = GenerationConfig(do_sample=False, max_new_tokens=256, pad_token_id = tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)

    model.eval()
    if top_args.DDP: model.to(device)
    dataset = VLLMDataset(data_args)
    collate_fn = VLLMCollactor(tokenizer=tokenizer, model_class=model_class, device=device, mode=data_args.mode)
    pin_memory = False if model_class.count('MiniCPM') > 0 else True
    kwargs = dict(batch_size=infer_args.batch_size, collate_fn=collate_fn, pin_memory=pin_memory)
   
    if top_args.DDP: 
        kwargs['sampler'] = DistributedSampler(dataset, world_size, rank, False, drop_last=False)
        file = f"{data_args.out_path}/{data_args.mode}_{model_class}_worker_{str(rank)}.json"
    else: kwargs['num_workers'], file = 1, f"{data_args.out_path}/{data_args.mode}_{model_class}_result.json"
    dataloader, rf = DataLoader(dataset, **kwargs), open(file, "w")
    prog_bar = tqdm(dataloader, total=len(dataloader), desc="worker_" + str(rank)) if rank == 0 else dataloader
    result = []
    for test_batch in prog_bar:   
        with torch.inference_mode():
            if model_class.count('MiniCPM') == 0:
                pic_kwargs = {k: v.to(device) for k, v in test_batch['pic_kwargs'].items()}
            else:
                pic_kwargs = test_batch['pic_kwargs']
                pic_kwargs['tokenizer'] = tokenizer.tokenizer
            outputs = model.generate(
                input_ids = test_batch['input_ids'].to(device),
                attention_mask = test_batch['attention_mask'].to(device),
                generation_config = generation_config,
                **pic_kwargs
            )
        labels, prompt_length = test_batch['labels'], test_batch['prompt_length']
        if model_class.count('Phi') > 0: predict = tokenizer.batch_decode(outputs[:, test_batch['input_ids'].shape[1] :])
        elif model_class.count('Qwen') > 0: predict = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        else: predict = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for l, plen, p in zip(labels, prompt_length, predict): 
            l['pred'] = p[plen:].strip()
            result.append(l)
    json.dump(result, rf, ensure_ascii=False, indent=4)
     
        
if __name__ == "__main__":
    main("qwen")
