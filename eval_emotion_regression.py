import os
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

import util.misc as misc

from util.datasets import EmotionDatasetForRegression
from lavin.mm_adaptation import LaVINForRegression
import random
from util.misc import correlation, concordance_correlation_coefficient

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--bits', default='16bit', type=str,choices=['4bit','8bit','16bit'],
                        help='Quantization bits for training, fp16 by default')
    
    # Model parameters
    parser.add_argument('--llama_model_path', default='./llama', type=str,
                        help='path of llama model')

    parser.add_argument('--llm_model', default='7B', type=str, metavar='MODEL',
                        help='Name of llm model to train')

    parser.add_argument('--use_vicuna',  action='store_true',   help='use vicuna weights')

    parser.add_argument('--cpu_load',  action='store_true',   help='load the model on cpu and avoid OOM on gpu')

    #block is not supported now.
    parser.add_argument('--adapter_type', type=str, default='attn', metavar='LENGTH',choices=['block','attn'],
                        help='the insert position  of adapter layer')


    parser.add_argument('--visual_adapter_type', type=str, default='normal', metavar='LENGTH',choices=['normal','router','router_block'],
                        help='the type of adapter layer')

    parser.add_argument('--adapter_dim', type=int, default=8, metavar='LENGTH', help='the dims of adapter layer')

    parser.add_argument('--hidden_proj', type=int, default=128, metavar='LENGTH',
                        help='the visual adapter dim')

    parser.add_argument('--temperature', type=float, default=10., metavar='LENGTH',
                        help='the temperature of router')

    parser.add_argument('--adapter_scale', type=float, default=1., metavar='LENGTH', help='the scales of adapter layer')
    parser.add_argument('--drop_path', type=float, default=0., metavar='LENGTH', help='drop path')

    parser.add_argument('--max_seq_len', type=int, default=512, metavar='LENGTH',
                        help='the maximum sequence length')

    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='saving memory costs via gradient_checkpointing')
    
    # Dataset parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--adapter_path', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    #datasets
    return parser


def main(args):

    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    random.seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

    dataset_test = EmotionDatasetForRegression(args,'/ocean/projects/cis240055p/skhyalia/dataset_original/iemocap_valence_test.csv', 'test', 'valence', args.llama_model_path, args.max_seq_len)
    g = torch.Generator()
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False
    )
    print(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        generator=g,
    )
    # define the model
    model = LaVINForRegression(args)

    adapter_checkpoint = torch.load(args.adapter_path, map_location="cpu")
    state_dict={}
    for key in adapter_checkpoint['model']:
        state_dict[key.replace('module.','')]=adapter_checkpoint['model'][key]

    model.load_state_dict(state_dict, strict=False)
    model.to(torch.device('cuda'))

    model.eval()
    with torch.no_grad():
        total_items=len(dataset_test)
        print('total_items: ',total_items)
        answers = []
        preds=[]
        prefix_audio = torch.tensor(dataset_test.tokenizer.encode("Audio: ", bos=False, eos=False), dtype=torch.int64)
        prefix_video = torch.tensor(dataset_test.tokenizer.encode("Video: ", bos=False, eos=False), dtype=torch.int64)
        for data_iter_step, (examples, labels, values, example_mask, videos , audios) in enumerate(data_loader_test):
            print("Progress: ",data_iter_step,"/",len(data_loader_test))
            pred = model.predict(examples, labels, values, videos = videos, prefix_video = prefix_video, audios = audios, prefix_audio =  prefix_audio)
            preds.append(pred)
            answers.append(values)
        
        preds = torch.cat(preds, 0)
        answers = torch.cat(answers, 0).cuda()
        corr = concordance_correlation_coefficient(preds, answers)
        mae_loss = torch.nn.L1Loss()(preds, answers)
        print("Correlation: ",corr.item())
        print("MAE: ", mae_loss.item())

if __name__ == '__main__':

    args = get_args_parser()
    args = args.parse_args()
    main(args)
