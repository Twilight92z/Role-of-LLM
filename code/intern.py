import math
import json
import torch
from tqdm import tqdm
from PIL import Image
from typing import Dict, Any
from dataclasses import dataclass
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from torchvision.transforms.functional import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
torch_dtype = torch.float16


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def split_model():
    device_map = {}
    num_layers = 80
    world_size = torch.cuda.device_count()
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for _ in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map


def get_data(data_path: str):
    with open(f'data/{data_path}/test.json', 'r') as f: return json.load(f)


class VLLMDataset(Dataset):
    def __init__(self, data_path):
        super(VLLMDataset, self).__init__()
        self.data = get_data(data_path)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        item = self.data[index]
        return item


@dataclass
class VLLMCollactor(object):
    tokenizer: Any
    mode: str
    def __call__(self, dataset) -> Dict[str, torch.Tensor]:
        labels, inputs, urls = [], [], []
        for item in dataset:
            labels.append(item)
            urls.append(item['url'])
            if self.mode.startswith('qr'):
                information = ""
                if self.mode == 'qr1': knowledge = item['advice'][:1]
                elif self.mode == 'qr3': knowledge = item['advice'][:3]
                elif self.mode == 'qr5': knowledge = item['advice'][:5]
                elif self.mode == 'qrx': knowledge = item['advice'][:10]
                for i, k in enumerate(knowledge): information += f' {i + 1}. {k}\n'
                prompt = f"Here are some pieces of information that might be helpful:\n{information}Now answer the question below:\n {item['vqa_question']}"
                inputs.append(prompt)
            elif self.mode.startswith('q'): inputs.append(item['vqa_question'])
            elif self.mode == 'e': inputs.append(item['entity_question'])
            elif self.mode == 'o': inputs.append(item['original_question'])
        return dict(labels=labels, urls=urls, inputs=inputs)

    
def make_dataloader(tokenizer, mode, data_path, batch_size):
    dataset = VLLMDataset(data_path)
    collate_fn = VLLMCollactor(tokenizer=tokenizer, mode=mode)
    kwargs = dict(batch_size=batch_size, collate_fn=collate_fn, pin_memory=True)
    dataloader = DataLoader(dataset, **kwargs)
    return dataloader


def main(tokenizer, model, mode, data_path, batch_size):
    model.eval()
    dataloader = make_dataloader(tokenizer, mode, data_path, batch_size)
    generation_config = dict(max_new_tokens=1024, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    result = []
    mod = 'backbone' if mode in ['o', 'qb'] else 'vllm'
    out_path = f'result/{data_path}/{mod}/{mode}_InternVL2-Llama3-76B_result.json'
    rf = open(out_path, 'w')
    for test_batch in tqdm(dataloader, total=len(dataloader)):   
        with torch.inference_mode():
            if mode in ['o', 'qb']:
                inputs = test_batch['inputs']
                num_patches_list = [0 for _ in range(len(inputs))]
                responses = model.batch_chat(tokenizer, None, num_patches_list=num_patches_list, questions=inputs, generation_config=generation_config)
                for l, r in zip(test_batch['labels'], responses): 
                    l['pred'] = r.strip()
                    result.append(l)
            elif mode in ['e', 'qv'] or mode.startswith('qr'):
                inputs = [f'<image>\n{i}' for i in test_batch['inputs']]
                pixel_values = [load_image(url).to(torch_dtype).cuda() for url in test_batch['urls']]
                num_patches_list = [p.size(0) for p in pixel_values]
                pixel_values = torch.cat(pixel_values, dim=0)
                responses = model.batch_chat(tokenizer, pixel_values, num_patches_list=num_patches_list, questions=inputs, generation_config=generation_config)
                for l, r in zip(test_batch['labels'], responses): 
                    l['pred'] = r.strip()
                    result.append(l)
    json.dump(result, rf, ensure_ascii=False, indent=4)    


if __name__ == "__main__":
    path = 'OpenGVLab/InternVL2-Llama3-76B'
    device_map = split_model()
    model = AutoModel.from_pretrained(path, torch_dtype=torch_dtype, use_flash_attn=True, trust_remote_code=True, device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    mode = "qrx"
    data_path = "infoseek"
    batch_size = 32
    
    main(tokenizer, model, mode, data_path, 4)