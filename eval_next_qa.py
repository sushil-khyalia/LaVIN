# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from lavin.eval_model import ModelArgs, Transformer
from lavin.tokenizer import Tokenizer
from lavin.generator import LaVIN_Generator
from lavin.mm_adapter import set_MMAdapter,set_Vivit_Adapter, set_Whisper_Adapter
import re
import random

from util.misc import sample_frame_indices, read_video_pyav
from vivit import VivitImageProcessor
from whisper import WhisperFeatureExtractor

import av
import soundfile

import warnings
import pandas as pd

from pathlib import Path
import fairscale.nn.model_parallel.initialize as fs_init
import torch.distributed as dist
from util.apply_delta import apply_model_delta_online

from pathlib import Path
import gc

warnings.filterwarnings('ignore')



def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size

def _load_and_redistribute_checkpoint(llama_model_path, model_name):

    with open(Path(llama_model_path) / model_name / 'params.json') as f:
        params = json.load(f)
    tokenizer = Tokenizer(model_path=str(Path(llama_model_path) / '8B/tokenizer.model'))
    print('Using model path: %s, model_name: %s' % (llama_model_path, model_name))
    if model_name=='7B' or model_name =='8B':
        checkpoint = torch.load(llama_model_path + model_name + '/consolidated.00.pth', map_location="cpu")
        return checkpoint, tokenizer, params

    checkpoints = (Path(llama_model_path) / model_name).glob('*.pth')
    checkpoints = sorted(checkpoints)

    mp_world_size = fs_init.get_model_parallel_world_size()
    mp_rank = fs_init.get_model_parallel_rank()
    if mp_world_size == len(checkpoints):
        print('same number of shards of checkpoints and training, loading directly...')
        dist.barrier()
        print('[rank=%d, mp_rank=%d] loading from %s' % (dist.get_rank(), mp_rank, checkpoints[mp_rank]))
        checkpoint = torch.load(checkpoints[mp_rank], map_location='cpu')
    else:
        print('different number of shards of checkpoints and training, redistributing...')
        if dist.get_rank() == 0:
            loaded = []
            for x in checkpoints:
                print('loading from', x)
                loaded.append(torch.load(x, map_location='cpu'))

            full_state_dict = {}
            split_dims = {}

            def add_weight_with_split_dim(name, dim):
                if dim < 0:  # bcast without split
                    full_state_dict[name] = loaded[0][name].clone()
                else:
                    full_state_dict[name] = torch.cat([x[name] for x in loaded], dim=dim)
                for x in loaded:
                    del x[name]
                split_dims[name] = dim

            add_weight_with_split_dim('tok_embeddings.weight', 1)
            add_weight_with_split_dim('norm.weight', -1)
            add_weight_with_split_dim('output.weight', 0)
            for i in range(params['n_layers']):
                print('gathering layer %d of %d' % (i, params['n_layers']))
                layer_prefix = f'layers.{i}.'
                bcast_names = [
                    'attention_norm.weight',
                    'ffn_norm.weight',
                ]
                column_parallel_names = [
                    'attention.wq.weight',
                    'attention.wk.weight',
                    'attention.wv.weight',
                    'feed_forward.w1.weight',
                    'feed_forward.w3.weight',
                ]
                row_parallel_names = [
                    'attention.wo.weight',
                    'feed_forward.w2.weight',
                ]
                for key in bcast_names:
                    add_weight_with_split_dim(layer_prefix + key, -1)
                for key in column_parallel_names:
                    add_weight_with_split_dim(layer_prefix + key, 0)
                for key in row_parallel_names:
                    add_weight_with_split_dim(layer_prefix + key, 1)

            full_state_dict_meta = dict((k, v.shape) for k, v in full_state_dict.items())
            dist.broadcast_object_list([full_state_dict_meta, split_dims], src=0)

        else:  # dist.get_rank() != 0
            recv_objs = [None, None]
            dist.broadcast_object_list(recv_objs, src=0)
            full_state_dict_meta, split_dims = recv_objs

        local_state_dict = {}
        for k in sorted(full_state_dict_meta.keys()):
            print('redistributing weights: %s' % k)
            if dist.get_rank() == 0:
                value = full_state_dict[k].cuda().half()
                del full_state_dict[k]
            else:
                value = torch.empty(full_state_dict_meta[k], device='cuda', dtype=torch.half)
            dist.broadcast(value, src=0)
            value = value.cpu()
            if split_dims[k] < 0:
                local_state_dict[k] = value
            else:
                dim = split_dims[k]
                assert dim >= 0 and dim < value.ndim and value.size(dim) % mp_world_size == 0
                shard_size = value.size(dim) // mp_world_size
                shard_st, shard_ed = shard_size * mp_rank, shard_size * (mp_rank + 1)
                # TODO: make more general
                if dim == 0:
                    value = value[shard_st: shard_ed]
                elif dim == 1:
                    value = value[:, shard_st: shard_ed]
                else:
                    raise NotImplementedError()
                local_state_dict[k] = value.clone()

        checkpoint = local_state_dict

    return checkpoint, tokenizer, params


def load(
    ckpt_dir: str,
    llm_model:str,
    adapter_path: str,
    max_seq_len: int,
    max_batch_size: int,
    adapter_type: str,
    adapter_dim:int,
    adapter_scale:float,
    hidden_proj:int,
    visual_adapter_type: str,
    temperature: float,
    use_vicuna: bool,
    bits: str='16bits',
    cpu_load:bool=False,
) -> LaVIN_Generator:
    start_time = time.time()
    checkpoint, tokenizer, params = _load_and_redistribute_checkpoint(ckpt_dir, llm_model)

    print("Loading")
    adapter_checkpoint = torch.load(adapter_path, map_location="cpu")


    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size,hidden_proj=hidden_proj, **params
    )
    model_args.vocab_size = tokenizer.n_words

    if cpu_load:
        #cpu load is slow, but is freindly for GPU with limited memory.
        torch.set_default_tensor_type(torch.HalfTensor)
    else:
        torch.set_default_tensor_type(torch.cuda.HalfTensor)

    model = Transformer(model_args)
    #delete language encoder

    torch.set_default_tensor_type(torch.FloatTensor)

    if bits in ['4bit','8bit']:
        from util.quantization import quant_model_bnb
        model.layers = quant_model_bnb(model.layers, quant_bit='4bit')

    set_MMAdapter(model, adapter_type, dim=adapter_dim, s=adapter_scale,t=temperature)
    set_Vivit_Adapter(model.video_backbone,visual_adapter_type,dim=adapter_dim,s=adapter_scale,t=temperature)
    set_Whisper_Adapter(model.audio_backbone,visual_adapter_type,dim=adapter_dim,s=adapter_scale,t=temperature)

    model.load_state_dict(checkpoint, strict=False)

    if use_vicuna:
        apply_model_delta_online(model,'../data/weights/vicuna_'+llm_model)


    state_dict={}
    for key in adapter_checkpoint['model']:
        state_dict[key.replace('module.','')]=adapter_checkpoint['model'][key]

    model.load_state_dict(state_dict, strict=False)
    model.to(torch.device('cuda'))

    for name, param in model.named_parameters():
        print(name,param.dtype)
    generator = LaVIN_Generator(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator

def get_pred_idx(prediction, choices, options):
    """
    Get the index (e.g. 2) from the prediction (e.g. 'C')
    """
    if prediction in options[:len(choices)]:
        return options.index(prediction)
    else:
        return random.choice(range(len(choices)))

def main(
    ckpt_dir: str,
    adapter_path: str,
    max_seq_len: int,
    max_batch_size: int,
    llm_model:str='7B',
    generation_temperature: float = 0.1,
    top_p: float = 0.75,
    split='valid',
    adapter_type='repattn',
    adapter_dim=8,
    adapter_scale=1,
    hidden_proj=128,
    visual_adapter_type='normal',
    temperature=10.,
    use_vicuna=False,
    bits: str='16bits',
    cpu_load:bool=False,
):
    print(max_batch_size,max_seq_len)
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(
        ckpt_dir,llm_model, adapter_path, max_seq_len, max_batch_size,
        adapter_type,adapter_dim,adapter_scale,hidden_proj,visual_adapter_type,
        temperature,use_vicuna,bits=bits,cpu_load=cpu_load)
    
    print('split: ', split)

    image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
    video_path = Path('/ocean/projects/cis240055p/skhyalia/NExTVideo')
    data_path = Path('/ocean/projects/cis240055p/skhyalia/NExT-QA/dataset/nextqa/test.csv')
    raw_data = pd.read_csv(data_path)
    map = json.load(open('/ocean/projects/cis240055p/skhyalia/NExT-QA/dataset/nextqa/map_vid_vidorID.json','r'))
    feature_directory = video_path/"processed_features"
    feature_directory.mkdir(exist_ok=True)
    for _, row in raw_data.iterrows():
        id = str(row['video'])
        if not (feature_directory/(id+"_video.pt")).exists():
            video_file_path = video_path/(map[id] + ".mp4")
            audio_file_path = video_path/(map[id] + ".wav")
            command = f'/ocean/projects/cis240055p/skhyalia/NExT-QA/extract_or_generate_audio.sh {video_file_path} {audio_file_path}'
            os.system(command)
            container = av.open(video_file_path)
            indices = sample_frame_indices(clip_len=32, frame_sample_rate=4, seg_len=container.streams.video[0].frames)
            video = read_video_pyav(container=container, indices=indices)
            video = image_processor(list(video), return_tensors="pt").pixel_values.squeeze(0)
            waveform, sampling_rate = soundfile.read(audio_file_path)
            audio = feature_extractor(waveform, sampling_rate=sampling_rate, return_tensors="pt").input_features.squeeze(0)
            torch.save(video, feature_directory/(id+"_video.pt"))
            torch.save(audio, feature_directory/(id+"_audio.pt"))
            gc.collect()

    total_items=len(raw_data)
    print('total_items: ',total_items)

    choices = ["Final Answer: (1)", "Final Answer: (2)", "Final Answer: (3)", "Final Answer: (4)", "Final Answer: (5)"]
    pattern = re.compile('Final Answer: \([1-5]\)')

    answers = []
    preds=[]
    for i in range(total_items//max_batch_size+1):
        print('progresses: ',i,' / ', total_items//max_batch_size+1)
        data_points = raw_data.iloc[i*max_batch_size:(i+1)*max_batch_size]
        if len(data_points)==0:
            break

        prompts=[]
        videos = []
        audios = []
        for _, data_point in data_points.iterrows():
            id = str(data_point['video'])
            video = torch.load(video_path/"processed_features"/(id+"_video.pt"))
            audio = torch.load(video_path/"processed_features"/(id+"_audio.pt"))
            prompt_text = f"You will be given a question about a video and five possible answer options.\nQuestion: {data_point['question']}?\nPossible Answer Choices:\n(1) {data_point['a0']}\n(2) {data_point['a1']}\n(3) {data_point['a2']}\n(4) {data_point['a3']}\n(5) {data_point['a4']}\nOutput the final answer in the format \"Final Answer: (X) Y.\" where X is the correct digit choice and Y is the text of the choice. Never say \"unknown\" or \"unsure\", or \"None\", instead provide your most likely guess.\n"
            prompt_text+="Final Answer: "
            prompt_text='\n'+prompt_text
            prompt_text = prompt_text.replace("  ", " ").strip()
            
            label = f"Final Answer: ({data_point['answer']+1})"
            prompts.append(prompt_text)
            answers.append(label)
            videos.append(video.unsqueeze(0))
            audios.append(audio.unsqueeze(0))
        videos=torch.cat(videos,0)
        audios=torch.cat(audios,0)


        results = generator.generate(
            prompts,videos=videos,audios=audios, max_gen_len=64, temperature=generation_temperature, top_p=top_p
        )

        for result in results:
            pred = pattern.findall(result)

            if len(pred) >= 1:
                pred = pred[0]  # 'A', 'B', ...
            else:
                # print(result)
                pred = "FAILED"
            preds.append(pred)

    #evaluations
    results={}
    correct=0

    for i, prediction in enumerate(preds):
        pred_idx = get_pred_idx(prediction, choices, choices)  # 0, 1, ..., 4
        if pred_idx == choices.index(answers[i]):
            correct += 1
        results[f"{raw_data.iloc[i]['video']}_{raw_data.iloc[i]['qid']}"] = prediction
    acc = correct / len(results) * 100
    print('overall accuracy: ', acc)

    with open('./preds.json', 'w') as f:
        json.dump(results,f)

if __name__ == "__main__":
    fire.Fire(main)
