import torch
from torchao._models.llama.model import Transformer, ModelArgs
from torchao._models.llama.model_mpe import Transformer as Transformer_mpe
from pathlib import Path
from argparse import ArgumentParser
import os
import time
import copy
import random
import json
from functools import partial

from accelerate import infer_auto_device_map, dispatch_model
from collections import OrderedDict

from hqq.core.quantize import BaseQuantizeConfig
from hqq.utils.patching import prepare_for_inference
from hqq.models.hf.base import AutoHQQHFModel
from hqq.core.quantize import HQQLinear
from hqq.backends.bitblas import HQQLinearBitBlas

from torchao.utils import TORCH_VERSION_AT_LEAST_2_5
from torchao.utils import unwrap_tensor_subclass
torch.profiler._utils._init_for_cuda_graphs()

def recursive_setattr(root_attr, attr_name, newattr):
    attr_names = attr_name.split('.')
    if len(attr_names) == 1:
        setattr(root_attr, attr_name, newattr)
    else:
        recursive_setattr(getattr(root_attr, attr_names[0]), '.'.join(attr_names[1:]), newattr)

def register_input_caching_hook(module, input_args_list):
    def pre_hook(module, input, input_args_list):
        input_args_list.extend(list(input))

    hook_function = partial(pre_hook, input_args_list=input_args_list)
    
    return module.register_forward_pre_hook(hook_function)

def measure_time(model, input_args, num_iter, compile=False, prof_name='', warm_up=10):
    if compile:
        torch.compiler.reset()
        model_mpe = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    else:
        model_mpe = model
    
    torch._dynamo.reset()
    if not TORCH_VERSION_AT_LEAST_2_5:
        unwrap_tensor_subclass(model_mpe)
        
    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)
    # # warm up and compile
    with torch.no_grad():
        for _ in range(warm_up):
            model_mpe(*input_args)
    torch.cuda.synchronize()
    
    # start.record()
    with torch.profiler.profile(with_stack=False, 
                                schedule=torch.profiler.schedule(
                                    wait=0,
                                    warmup=num_iter-10,
                                    active=10,
                                    repeat=1),
                                ) as prof:
        with torch.profiler.record_function(f"warmup"):
            with torch.no_grad():
                model_mpe(*input_args)
        for i in range(num_iter):
            with torch.profiler.record_function(f"model forward iter {i}"):
                    with torch.no_grad():
                        model_mpe(*input_args)
                        torch.cuda.synchronize()
                        prof.step()
    # end.record()
    # torch.cuda.synchronize()
    prof.export_chrome_trace(f"241211_compile_iter{num_iter}_{prof_name}.json")
    # import code; code.interact('measure_time profile', local=dict(globals(), **locals()))
    
    return 0
    # return start.elapsed_time(end) / num_iter

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
    
    L = config.n_layer
    config.n_layer = 1 # for lpe
    
    t1 = time.perf_counter()
    model = Transformer(config).half().to(device)
    model.dtype = torch.half
    model.device = device
    t2 = time.perf_counter()
    print(f"MPE Model loading time : {t2-t1} second")
    
    batch_size = 1
    T = 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size,), device=device).unsqueeze(0)
    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    with torch.device(device):
        model.setup_caches(max_batch_size=1, max_seq_length=T+8, kv_cache_quantization=False, linear_causal_mask=False, prompt_length=T)
        
    layers_dict = {}
    layers_dict[16] = {n:m for n,m in model.layers[0].named_modules() if isinstance(m, torch.nn.Linear)}
    t1 = time.perf_counter()
    # 4bit model
    model_4bit = copy.deepcopy(model)
    quant_config = BaseQuantizeConfig(nbits=4, group_size=128)
    AutoHQQHFModel.quantize_model(model_4bit, quant_config=quant_config, compute_dtype=torch.half, device='cuda')
    prepare_for_inference(model_4bit, backend="bitblas")
    layers_dict[4] = {n:m for n,m in model_4bit.layers[0].named_modules() if isinstance(m, (HQQLinear, HQQLinearBitBlas))}
    del model_4bit
    t2 = time.perf_counter()
    print(f"4bit Model loading time : {t2-t1} second")
    
    # 2bit model
    t1 = time.perf_counter()
    model_2bit = copy.deepcopy(model)
    quant_config = BaseQuantizeConfig(nbits=2, group_size=64)
    AutoHQQHFModel.quantize_model(model_2bit, quant_config=quant_config, compute_dtype=torch.half, device='cuda')
    prepare_for_inference(model_2bit, backend="bitblas")
    layers_dict[2] = {n:m for n,m in model_2bit.layers[0].named_modules() if isinstance(m, (HQQLinear, HQQLinearBitBlas))}
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
    
    def convert_layername(name, bit):
        return name_map_dict[name]+f'_{bit}bit'
    numel_dict = {name_map_dict[n]:m.weight.numel() for n, m in layers_dict[16].items()}
    
    # block input args caching
    input_args = []
    block2 = copy.deepcopy(model.layers[0])
    hook = register_input_caching_hook(model.layers[0], input_args)
    model(input_ids, input_pos)
    hook.remove()
    del hook
    b, s, d = input_args[0].shape
    
    # stage 1 : single layer
    st = time.perf_counter()
    times_dict = {}
    bit_candidate = [2, 4, 16]
    times_dict['linear'] = {}
    for name in name_map_dict:
        for bit in bit_candidate:
            newname = convert_layername(name, bit)
            prof_name = 'linear_'+newname.split('.')[-1][0]+f'{bit}'
            linear = layers_dict[bit][name]
            input_x = torch.randn((b, s, layers_dict[16][name].in_features), dtype=input_args[0].dtype, device=input_args[0].device)
            times_dict['linear'][newname] = measure_time(linear, [input_x], args.num_iter, args.compile, prof_name)
            
    # import code; code.interact('after stage 1', local=dict(globals(), **locals()))
    # stage 2 : single block
    combination_names = [
        ['attention.wq', 'attention.wk', 'qk'], 
        ['attention.wk', 'attention.wv', 'kv'], 
        ['attention.wv', 'attention.wo', 'vo'], 
        ['attention.wo', 'feed_forward.w1', 'og'], 
        ['feed_forward.w1', 'feed_forward.w3', 'gu'], 
        ['feed_forward.w3', 'feed_forward.w2', 'ud'],
    ]
    
    block = model.layers[0]
    times_dict['block'] = {}
    times_dict['edge'] = {}
    times_dict['block']['fp16'] = measure_time(block, input_args, args.num_iter, args.compile, 'fp16')
    # low bit only
    for bit in [2, 4]:
        for name in name_map_dict:
            newname = convert_layername(name, bit)
            recursive_setattr(block, name, layers_dict[bit][name])
        times_dict['block'][f'{bit}bit'] = measure_time(block, input_args, args.num_iter, args.compile, f'{bit}bit')
    import code; code.interact('after stage 2, 4, 16bit only', local=dict(globals(), **locals()))
    
    # single layer in block
    bit_candidate = [2, 4]
    for name in name_map_dict:
        for bit in bit_candidate:
            newname = convert_layername(name, bit)
            prof_name = 'block_'+newname.split('.')[-1][0]+f'{bit}'
            recursive_setattr(block, name, layers_dict[bit][name])
            times_dict['block'][newname] = measure_time(block, input_args, args.num_iter, args.compile, prof_name)
            recursive_setattr(block, name, layers_dict[16][name])
        
    # import code; code.interact('after stage 2-1', local=dict(globals(), **locals()))
    
    for lname1, lname2, prof_name_ in combination_names:
        for bit1, bit2 in [[2, 2], [2, 4], [4, 2], [4, 4]]:
            newname1 = convert_layername(lname1, bit1)
            newname2 = convert_layername(lname2, bit2)
            prof_name = 'block_'+prof_name_+f'{bit1}{bit2}'
            edge_name = newname1+'-'+newname2
            recursive_setattr(block, lname1, layers_dict[bit1][lname1])
            recursive_setattr(block, lname2, layers_dict[bit2][lname2])
            times_dict['block'][edge_name] = measure_time(block, input_args, args.num_iter, args.compile, prof_name)
            recursive_setattr(block, lname1, layers_dict[16][lname1])
            recursive_setattr(block, lname2, layers_dict[16][lname2])
            # times_dict['edge'][edge_name] = (times_dict['block']['fp16'] - times_dict['block'][edge_name])  \
            #                                         - (times_dict['linear'][convert_layername(lname1, 16)] + times_dict['linear'][convert_layername(lname2, 16)]) \
            #                                         + (times_dict['linear'][newname1] + times_dict['linear'][newname2])
    import code; code.interact('after stage 2', local=dict(globals(), **locals()))
    
    # d-q edge
    model.layers.append(block2) # construct 2 block model
    
    times_dict['block']['fp16_2block'] = measure_time(model, [input_ids, input_pos], args.num_iter, args.compile)
    lname1, lname2 = 'feed_forward.w2', 'attention.wq'
    for bit1, bit2 in [[2, 2], [2, 4], [4, 2], [4, 4]]:
        newname1 = convert_layername(lname1, bit1)
        newname2 = convert_layername(lname2, bit2)
        edge_name = newname1+'-'+newname2
        recursive_setattr(model.layers[0], lname1, layers_dict[bit1][lname1])
        recursive_setattr(model.layers[1], lname2, layers_dict[bit2][lname2])
        times_dict['block'][edge_name] = measure_time(model, [input_ids, input_pos], args.num_iter, args.compile)
        recursive_setattr(model.layers[0], lname1, layers_dict[16][lname1])
        recursive_setattr(model.layers[1], lname2, layers_dict[16][lname2])
        times_dict['edge'][edge_name] = (times_dict['block']['fp16_2block'] - times_dict['block'][edge_name]) \
                                        - (times_dict['linear'][convert_layername(lname1, 16)] + times_dict['linear'][convert_layername(lname2, 16)]) \
                                        + (times_dict['linear'][newname1] + times_dict['linear'][newname2])
    
    combination_names.append([lname1, lname2])
        
    # import code; code.interact('after stage 2', local=dict(globals(), **locals()))
    
    end = time.perf_counter()
    print(f"MPE running time : {end-st} second")
        
    # actual forward
    del model
    config.n_layer = L
    
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
    
    model = dispatch_model(model, device_map=device_map)
    model_num_iter = 100
    # import code; code.interact('line 247', local=dict(globals(), **locals()))
    times_dict['model'] = measure_time(model, [input_ids, input_pos], model_num_iter, args.compile)
    model = model.cpu()
   
    end = time.perf_counter()
    print(f"FP16 model loading and forward running time : {end-st} second")
    
    if args.output_path is not None:
        with open(args.output_path, 'w') as f:
            json.dump(times_dict, f, indent=2)
    else:
        print("output_path is not given")
    
    if args.test:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # load 1 block to gpu
        st = time.perf_counter()
        # layers_dict[16] = {n:m.to(device) for n,m in model.layers[0].named_modules() if isinstance(m, torch.nn.Linear)}
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
            t_lpe = times_dict['model']
            # import code; code.interact('bit combinations', local=dict(globals(), **locals()))
            
            # layer
            for block_idx, block in enumerate(model.layers):
                for name, module in block.named_modules():
                    if isinstance(module, (torch.nn.Linear, HQQLinearBitBlas)):
                        newname = name_map_dict[name]
                        bit = bit_config[newname][block_idx]
                        recursive_setattr(block, name, layers_dict[bit][name])
                        t_16bit = times_dict['linear'][newname+f'_16bit']
                        t_quant = times_dict['linear'][newname+f'_{bit}bit']
                        t_lpe = t_lpe - t_16bit + t_quant
            
            # edge
            for block_idx, block in enumerate(model.layers):
                for lname1, lname2 in combination_names:
                    newname1 = name_map_dict[lname1]
                    newname2 = name_map_dict[lname2]
                    bit1 = bit_config[newname1][block_idx]
                    bit2 = bit_config[newname2][block_idx]
                    edge_name = convert_layername(lname1, bit1)+'-'+convert_layername(lname2, bit2)
                    edge = times_dict['edge'][edge_name]
                    t_lpe = t_lpe - edge
                    print(f"{edge_name} gain : {edge} | t_lpe : {t_lpe}")
            
            t_lpe = t_lpe + edge # conpensate last d-q
                
            model = dispatch_model(model, device_map=device_map)
            t2 = time.perf_counter()
            t_real = measure_time(model, [input_ids, input_pos], model_num_iter, args.compile)
            bit_configs[i]['t_lpe'] = t_lpe
            bit_configs[i]['t_real'] = t_real
            
            bit_configs[i]['average_bit'] = average_bit
            # import code; code.interact('average_bit', local=dict(globals(), **locals()))
            t3 = time.perf_counter()
            print(f"sample {i} | average_bit : {average_bit} | measure time: {t3-t2} sec | t_lpe : {t_lpe} | t_real : {t_real}")
            import code; code.interact('line 254', local=dict(globals(), **locals()))
            
        lpe_times = [bit_configs[i]['t_lpe'] for i in range(len(bit_configs))]
        real_times = [bit_configs[i]['t_real'] for i in range(len(bit_configs))]
        average_bits = [bit_configs[i]['average_bit'] for i in range(len(bit_configs))]
        
        plt.figure(figsize=(8, 6))
        plt.scatter(average_bits, real_times, color='blue', label='Real time', alpha=0.7)
        plt.scatter(average_bits, lpe_times, color='red', label='MPE estimate time', alpha=0.7)

        # 제목과 레이블 추가
        plt.title("Scatter Plot Example", fontsize=14)
        plt.xlabel("Average bits", fontsize=12)
        plt.ylabel("GeMV forward times (ms)", fontsize=12)
        plt.legend()
        plt.grid(True)

        # 그래프 표시
        plt.savefig(args.output_path.replace('.json', f'_{len(bit_configs)}samples.png'))
        end = time.perf_counter()
        print(f"Elapsed time to generate {len(bit_configs)} samples : {end-st} seconds")
    
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