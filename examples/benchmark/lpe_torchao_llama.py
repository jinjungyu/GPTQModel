import torch
import sys
sys.path.append('/NAS/JG/QAS4SD/torchao')
from torchao._models.llama.model import Transformer, ModelArgs
# from torchao._models.llama.model_mpe import Transformer as Transformer_mpe
from pathlib import Path
from argparse import ArgumentParser
import os
import time
import copy
import random
import json

from accelerate import infer_auto_device_map, dispatch_model
from collections import OrderedDict

from hqq.core.quantize import BaseQuantizeConfig
from hqq.utils.patching import prepare_for_inference
from hqq.models.hf.base import AutoHQQHFModel
from hqq.backends.bitblas import HQQLinearBitBlas
from hqq.backends.autogptq import AutoGPTQLinear

def recursive_setattr(root_attr, attr_name, newattr):
    attr_names = attr_name.split('.')
    if len(attr_names) == 1:
        setattr(root_attr, attr_name, newattr)
    else:
        recursive_setattr(getattr(root_attr, attr_names[0]), '.'.join(attr_names[1:]), newattr)

def measure_time(model, input_ids, input_pos, num_iter, compile=False, warm_up=10):
    if compile:
        torch.compiler.reset()
        model_mpe = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    else:
        model_mpe = model
        
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # warm up and compile
    with torch.no_grad():
        for _ in range(warm_up):
            model_mpe(input_ids, input_pos)

    start.record()
    with torch.no_grad():
        for _ in range(num_iter):
            model_mpe(input_ids, input_pos)
    end.record()
    torch.cuda.synchronize()
    
    return start.elapsed_time(end) / num_iter

def generate_bit_combinations(n_block, numel_dict, average_bit, num_combinations=1, tolerance=0.05):
    numel_list = list(numel_dict.values()) * n_block
    n_layers = len(numel_list)
    total_weighted_sum = average_bit * sum(numel_list)
    results = []
    attempts = 0
    
    if average_bit == 2.0:
        result = [2] * n_layers
        results.append(result)
    elif average_bit == 4.0:
        result = [4] * n_layers
        results.append(result)
    else:
        while len(results) < num_combinations and attempts < 100000:
            result = [2] * n_layers
            current_weighted_sum = sum(w * v for w, v in zip(numel_list, result)) 
            required_weighted_sum = total_weighted_sum - current_weighted_sum   
            indices = list(range(n_layers))
            random.shuffle(indices)
            for i in indices:
                if required_weighted_sum <= 0:
                    break
                addition_per_change = numel_list[i] * (4 - 2)  # weight × 2
                if addition_per_change <= required_weighted_sum:
                    result[i] = 4
                    required_weighted_sum -= addition_per_change      
            weighted_avg = sum(w * v for w, v in zip(numel_list, result)) / sum(numel_list)  
            if result not in results and abs(weighted_avg - average_bit) <= tolerance:
                results.append(result)    
            attempts += 1
    bit_config = {}
    if len(results) > 0:
        for layer_idx, name in enumerate(numel_dict.keys()):
            bit_config[name] = []
            for block_idx in range(n_block):
                bit_config[name].append(results[0][block_idx*len(numel_dict)+layer_idx])
        return bit_config
    else:
        return {}

def main(args):
    device = torch.device('cuda:0')
    config = ModelArgs.from_name(args.model_name_or_path)
    
    # for fast model loading
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    
    is_only_test = os.path.exists(args.output_path) and args.test
    L = config.n_layer
    config.n_layer = 1 # for lpe
    
    t1 = time.perf_counter()
    model = Transformer(config).half().to(device)
    model.dtype = torch.half
    model.device = device
    t2 = time.perf_counter()
    print(f"LPE Model loading time : {t2-t1} second")
    
    batch_size = 1
    T = 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size,), device=device).unsqueeze(0)
    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    with torch.device(device):
        model.setup_caches(max_batch_size=1, max_seq_length=T+8, kv_cache_quantization=False, linear_causal_mask=False, prompt_length=T)
        
    layers_dict = {}
    layers_dict[16] = {n:m for n,m in model.layers[0].named_modules() if isinstance(m, torch.nn.Linear)}
    
    # 3bit model test
    t1 = time.perf_counter()
    # 4bit model
    model_3bit = copy.deepcopy(model)
    quant_config = BaseQuantizeConfig(nbits=3, group_size=128)
    AutoHQQHFModel.quantize_model(model_3bit, quant_config=quant_config, compute_dtype=torch.half, device='cuda')
    prepare_for_inference(model_3bit, backend="gptq")
    layers_dict[3] = {n:m for n,m in model_3bit.layers[0].named_modules() if isinstance(m, AutoGPTQLinear)}
    t2 = time.perf_counter()
    print(f"3bit Model loading time : {t2-t1} second")

    name_map_dict = {
        'attention.wq': 'self_attn.q_proj',
        'attention.wk': 'self_attn.k_proj',
        'attention.wv': 'self_attn.v_proj',
        'attention.wo': 'self_attn.o_proj',
        'feed_forward.w1': 'mlp.gate_proj',
        'feed_forward.w3': 'mlp.up_proj',
        'feed_forward.w2': 'mlp.down_proj',
    }
    times_dict = {}
    times_dict['fp16_block'] = measure_time(model, input_ids, input_pos, args.num_iter, args.compile)
    block = model.layers[0]
    bit_candidate = [3]
    for bit in bit_candidate:
        times_dict[bit] = {}
        for name, module in block.named_modules():
            if isinstance(module, torch.nn.Linear):
                newname = name_map_dict[name]
                times_dict[bit][newname] = {}
                recursive_setattr(block, name, layers_dict[bit][name])
                times_dict[bit][newname] = measure_time(model, input_ids, input_pos, args.num_iter, args.compile) - times_dict['fp16_block']
                recursive_setattr(block, name, module)
                
    import code; code.interact('3bit check', local=dict(globals(), **locals()))
    t1 = time.perf_counter()
    # 4bit model
    model_4bit = copy.deepcopy(model)
    quant_config = BaseQuantizeConfig(nbits=4, group_size=128)
    AutoHQQHFModel.quantize_model(model_4bit, quant_config=quant_config, compute_dtype=torch.half, device='cuda')
    prepare_for_inference(model_4bit, backend="bitblas")
    layers_dict[4] = {n:m for n,m in model_4bit.layers[0].named_modules() if isinstance(m, HQQLinearBitBlas)}
    t2 = time.perf_counter()
    print(f"4bit Model loading time : {t2-t1} second")
    
    # 2bit model
    t1 = time.perf_counter()
    model_2bit = copy.deepcopy(model)
    quant_config = BaseQuantizeConfig(nbits=2, group_size=64)
    AutoHQQHFModel.quantize_model(model_2bit, quant_config=quant_config, compute_dtype=torch.half, device='cuda')
    prepare_for_inference(model_2bit, backend="bitblas")
    layers_dict[2] = {n:m for n,m in model_2bit.layers[0].named_modules() if isinstance(m, HQQLinearBitBlas)}
    del model_2bit
    t2 = time.perf_counter()
    print(f"2bit Model loading time : {t2-t1} second")

    name_map_dict = {
        'attention.wq': 'self_attn.q_proj',
        'attention.wk': 'self_attn.k_proj',
        'attention.wv': 'self_attn.v_proj',
        'attention.wo': 'self_attn.o_proj',
        'feed_forward.w1': 'mlp.gate_proj',
        'feed_forward.w3': 'mlp.up_proj',
        'feed_forward.w2': 'mlp.down_proj',
    }
    
    numel_dict = {name_map_dict[n]:m.weight.numel() for n, m in layers_dict[16].items()}
    
    config.n_layer = L
    
    if not is_only_test:
        # LPE
        st = time.perf_counter()
        times_dict = {}
        times_dict['fp16_block'] = measure_time(model, input_ids, input_pos, args.num_iter, args.compile)
        block = model.layers[0]
        bit_candidate = [2, 4]
        for bit in bit_candidate:
            times_dict[bit] = {}
            for name, module in block.named_modules():
                if isinstance(module, torch.nn.Linear):
                    newname = name_map_dict[name]
                    times_dict[bit][newname] = {}
                    recursive_setattr(block, name, layers_dict[bit][name])
                    times_dict[bit][newname] = measure_time(model, input_ids, input_pos, args.num_iter, args.compile) - times_dict['fp16_block']
                    recursive_setattr(block, name, module)
        
        end = time.perf_counter()
        print(f"LPE running time : {end-st} second")
        # import code; code.interact('end mpe', local=locals())
        
        # actual forward
        del model
        
        st = time.perf_counter()
        model = Transformer(config).half()
        # with torch.device(device):
        model.setup_caches(max_batch_size=1, max_seq_length=T+8, 
                            kv_cache_quantization=False, linear_causal_mask=False, prompt_length=T)
        num_gpu = len(os.environ.get('CUDA_VISIBLE_DEVICES').split(','))

        if num_gpu > 1: # multi gpu
            device_map = OrderedDict()
            device_map['causal_mask'] = 0
            device_map['freqs_cis'] = 0
            device_map['tok_embeddings'] = 0
            for i in range(L):
                device_map[f'layers.{i}'] = i // (L // num_gpu)
            device_map['norm'] = num_gpu - 1
            device_map['output'] = num_gpu - 1
        else:
            device_map = OrderedDict()
            device_map[''] = 0 
        
        if not is_only_test:
            model = dispatch_model(model, device_map=device_map)
            times_dict['fp16_model'] = measure_time(model, input_ids, input_pos, args.num_iter // 100, args.compile)
        end = time.perf_counter()
        print(f"FP16 model loading and forward running time : {end-st} second")
        model = model.cpu()
    
        if args.output_path is not None:
            with open(args.output_path, 'w') as f:
                json.dump(times_dict, f, indent=2)
        else:
            print("output_path is not given")
    else:
        with open(args.output_path, 'r') as f:
            times_dict = json.load(f)
        for str_key in list(times_dict.keys()):
            if str_key.isdigit():
                times_dict[int(str_key)] = times_dict[str_key]
                del times_dict[str_key]
        model = Transformer(config).half()
        model.setup_caches(max_batch_size=1, max_seq_length=T+8, 
                            kv_cache_quantization=False, linear_causal_mask=False, prompt_length=T)
    
    if args.test:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # load 1 block to gpu
        st = time.perf_counter()
        for name in layers_dict[4]:
            layers_dict[4][name] = layers_dict[4][name].to(device)
        for name in layers_dict[2]:
            layers_dict[2][name] = layers_dict[2][name].to(device)
        
        device_map = OrderedDict()
        device_map[''] = 0
        
        random.seed(42)
        # validate 2.0 ~ 4.0 bit
        average_bits_range = np.arange(2.0, 4.1, 0.1).tolist()
        bit_configs = {}
        for i, average_bit in enumerate(average_bits_range):
            t1 = time.perf_counter()
            bit_config = generate_bit_combinations(config.n_layer, numel_dict, average_bit, num_combinations=1)
            bit_configs[i] = {}
            bit_configs[i]['linear'] = bit_config
            t_lpe = times_dict['fp16_model']
            # import code; code.interact('bit combinations', local=dict(globals(), **locals()))
            
            for block_idx, block in enumerate(model.layers):
                for name, module in block.named_modules():
                    if isinstance(module, (torch.nn.Linear, HQQLinearBitBlas)):
                        newname = name_map_dict[name]
                        bit = bit_config[newname][block_idx]
                        recursive_setattr(block, name, layers_dict[bit][name])
                        if bit < 16:
                            t_lpe += times_dict[bit][newname]
            model = dispatch_model(model, device_map=device_map)
            t2 = time.perf_counter()
            t_real = measure_time(model, input_ids, input_pos, args.num_iter // 100, args.compile)
            bit_configs[i]['t_lpe'] = t_lpe
            bit_configs[i]['t_real'] = t_real
            
            bit_configs[i]['average_bit'] = average_bit
            t3 = time.perf_counter()
            print(f"sample {i} | average_bit : {average_bit} | measure time: {t3-t2} sec | t_lpe : {t_lpe} | t_real : {t_real}")
            # import code; code.interact(f'sample {i}', local=dict(globals(), **locals()))
            
        lpe_times = [bit_configs[i]['t_lpe'] for i in range(len(bit_configs))]
        real_times = [bit_configs[i]['t_real'] for i in range(len(bit_configs))]
        average_bits = [bit_configs[i]['average_bit'] for i in range(len(bit_configs))]
        
        plt.figure(figsize=(8, 6))
        plt.scatter(average_bits, real_times, color='blue', label='Real time', alpha=0.7)
        plt.scatter(average_bits, lpe_times, color='red', label='LPE estimate time', alpha=0.7)

        # 제목과 레이블 추가
        plt.title("Scatter Plot Example", fontsize=14)
        plt.xlabel("Average bits", fontsize=12)
        plt.ylabel("GeMV forward times (ms)", fontsize=12)
        plt.legend()
        plt.grid(True)

        end = time.perf_counter()
        print(f"Elapsed time to generate {len(bit_configs)} samples : {end-st} seconds")
        
        result_path = args.output_path.replace('.json','_test.json')
        png_path = result_path.replace('.json', '.png')
        i = 0
        while True:
            if os.path.exists(png_path):
                png_path = png_path.replace('.png', '{i}.png')
                result_path = result_path.replace('.json', '{i}.json')
                i += 1
            else:
                break
        plt.savefig(png_path)
        with open(result_path, 'w') as f:
            json.dump(bit_configs, f, indent=2)
        print(f"result json is saved to {result_path} and png is saved to {png_path}")
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--num_iter", type=int, default=100)
    parser.add_argument("--quantized_4bit_path", type=str)
    parser.add_argument("--quantized_2bit_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--compile", action='store_true')
    args = parser.parse_args()
    main(args)