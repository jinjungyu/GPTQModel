import logging
import random
import time
from argparse import ArgumentParser
from itertools import chain
from typing import Dict, List, Optional

import torch
from datasets import Dataset, load_dataset
from gptqmodel import BACKEND, GPTQModel, QuantizeConfig, get_backend
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig
from transformers.generation.logits_process import LogitsProcessor

logger = logging.getLogger(__name__)
profile = False

random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime
import torch
import torchao
import torch._dynamo.config
import torch._inductor.config
from torchao.utils import get_model_size_in_bytes
from torchao.utils import TORCH_VERSION_AT_LEAST_2_5
import contextlib
import copy
import accelerate
from glob import glob

from fastchat.llm_judge.common import load_questions
from fastchat.model import get_conversation_template

from hqq.core.quantize import BaseQuantizeConfig
from hqq.utils.patching import prepare_for_inference
from hqq.models.hf.base import AutoHQQHFModel
from hqq.backends.bitblas import HQQLinearBitBlas
from hqq.backends.autogptq import GPTQLinear

def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif ("cpu" in device) or ("mps" in device):
        pass
    else:
        print(f"device={device} is not yet suppported")

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from torchao._models.llama.model import Transformer, prepare_inputs_for_model, ModelArgs

def multinomial_sample_one_no_sync(probs_sort): # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort.float()).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    dtype = logits.dtype
    logits = logits.float() / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1).to(dtype)
    return probs

def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[0, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs

@torch.no_grad()
def prefill(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> torch.Tensor:
    # input_pos: [B, S]
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)[0]

@torch.no_grad()
def decode_one_token(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    with torch.profiler.record_function(f"decode_one_token"):
        logits = model(x, input_pos)
    # import code; code.interact('decode_one_token', local=dict(globals(), **locals()))
    return sample(logits, **sampling_kwargs)

@torch.no_grad()
def decode_n_tokens(model: Transformer, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, callback=lambda _: _, **sampling_kwargs):
    new_tokens, new_probs = [], []
    for i in range(num_new_tokens):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True): # Actually better for Inductor to codegen attention here
            next_token, next_prob = decode_one_token(
                model, cur_token, input_pos, **sampling_kwargs
            )
            next_token, next_prob = next_token.clone(), next_prob.clone()
            input_pos += 1
            new_tokens.append(next_token)
            callback(new_tokens[-1])
            new_probs.append(next_prob)
            cur_token = next_token.view(1, -1)

    return new_tokens, new_probs

@torch.no_grad()
def model_forward(model, x, input_pos):
    return model(x, input_pos)

def speculative_decode(
    model: Transformer,
    draft_model: Transformer,
    cur_token: torch.Tensor,
    input_pos: int,
    speculate_k: int,
    **sampling_kwargs
) -> torch.Tensor:
    # draft model inference sequentially
    device = cur_token.device
    orig_input_pos = torch.tensor([input_pos], dtype=torch.int64, device=cur_token.device)
    draft_tokens, draft_probs = decode_n_tokens(draft_model, cur_token.view(1, -1), orig_input_pos.clone(), speculate_k, **sampling_kwargs)

    draft_tokens = torch.cat(draft_tokens)
    # parallel inference on target model using draft tokens
    target_logits = model_forward(
        model,
        torch.cat([cur_token.view(1), draft_tokens]).view(1, -1),
        torch.arange(input_pos, input_pos + speculate_k + 1, device=cur_token.device)
    )
    
    temperature = sampling_kwargs['temperature']

    if temperature > 1e-5:
        target_probs = logits_to_probs(target_logits[0], **sampling_kwargs)
        draft_probs = torch.stack(draft_probs)
        # q: target prob, p: draft prob
        # q >= p: always accept draft token
        # q < p: q/p prob to accept draft token
        p = draft_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
        q = target_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
        # q = target_probs[torch.arange(1, speculate_k+1, device=device), draft_tokens]
        accept_draft_prob = torch.minimum(torch.ones(()), q[:speculate_k]/ p)
        rejected_locations = (torch.rand_like(accept_draft_prob) > accept_draft_prob).nonzero()
        
        if rejected_locations.shape[0] == 0: # All draft tokens have been accepted
            accept_length = speculate_k + 1
            last_token = multinomial_sample_one_no_sync(target_probs[-1])
            # fill last token into draft model
            model_forward(
                draft_model,
                draft_tokens[-1].view(1, -1),
                orig_input_pos + speculate_k,
            )
            return torch.cat([draft_tokens, last_token])
        else:
            accept_length = rejected_locations[0].item()
            p = draft_probs[accept_length]
            q = target_probs[accept_length]
            new = q - p
            new = torch.where(new > 0, new, 0.0)
            new = new / new.sum()
            next_token = multinomial_sample_one_no_sync(new)
            return torch.cat([draft_tokens[:accept_length], next_token])
    else: # Greedy
        pass
        # selected_tokens = target_logits[0].argmax(dim=-1)
        # candidate_new_tokens = 
        
@torch.no_grad()
def generate(
    model: Transformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    *,
    interactive: bool,
    draft_model: Transformer = None,
    speculate_k: Optional[int] = 5,
    callback = lambda x: x,
    kv_cache_quantization: bool = False,
    cache_size: Optional[int] = None,
    linear_causal_mask: bool=False,
    **sampling_kwargs
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    is_speculative = draft_model is not None
    # create an empty tensor of the expected final shape and fill in the current tokens
    device = prompt.device
    T = prompt.numel()
    T_new = T + max_new_tokens

    # calculate how many tokens to generate based on max_new_tokens and model's upper bound (block_size)
    max_seq_length = min(T + max_new_tokens, model.config.block_size) if not interactive else 350
    max_seq_length = max_seq_length + speculate_k + 1 if is_speculative else max_seq_length
    
    new_tokens = max_seq_length - T

    # full prompt+output will be stored in seq
    seq = torch.empty(T_new, dtype=prompt.dtype, device=device)
    seq[:T] = prompt.view(-1)

    
    # setup model caches
    with torch.device(device):
        if cache_size is None:
            cache_size = max_seq_length
        assert cache_size >= max_seq_length, "need cache_size to be greater than max_new_tokens + size-of-prompt"
        model.setup_caches(max_batch_size=1, max_seq_length=cache_size, kv_cache_quantization=kv_cache_quantization, linear_causal_mask=linear_causal_mask, prompt_length=T)
        if is_speculative and draft_model is not model:
            draft_model.setup_caches(max_batch_size=1, max_seq_length=min(cache_size, model.config.block_size), kv_cache_quantization=kv_cache_quantization, linear_causal_mask=linear_causal_mask, prompt_length=T)

    # format model input
    x, input_pos = prepare_inputs_for_model(prompt, max_new_tokens)

    # execute prefill
    with torch.profiler.record_function(f"Prefill"):
        next_token = prefill(model, x, input_pos, **sampling_kwargs)

    if is_speculative:
        prefill(draft_model, x, input_pos, **sampling_kwargs)
        
    seq[T] = next_token
    # execute token generation
    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    accept_counts = [0] * (speculate_k + 1)
    gamma = 0
    # import code; code.interact('generate', local=dict(globals(), **locals()))
    with torch.profiler.record_function(f"Generate"):
        if is_speculative:
            input_pos = input_pos.item()  # for speculative decoding easier to keep on host
            # print(f"input_pos : {input_pos} max_seq_length : {max_seq_length}")
            while input_pos < T_new - 1:
                cur_token = next_token.view(())

                next_tokens = speculative_decode(
                    model, draft_model, cur_token, input_pos, speculate_k, **sampling_kwargs
                )
                
                accept_counts[len(next_tokens) - 1] += 1
                num_added = min(T_new - input_pos - 1, len(next_tokens))
                seq[input_pos + 1 : input_pos + num_added + 1] = next_tokens[: num_added]
                for i in next_tokens[: num_added,]:
                    callback(i)
                input_pos = input_pos + num_added
                next_token = next_tokens[-1]
                # print(f"cur input_pos : {input_pos} num_added : {num_added}")
            gamma = sum([length * count for length, count in enumerate(accept_counts, start=1)]) / sum(accept_counts)
        else:
            generated_tokens, _ = decode_n_tokens(model, next_token.view(1, -1), input_pos, new_tokens-1, callback=callback, **sampling_kwargs)
            seq = torch.cat((seq[:T+1], *generated_tokens))
    
    generate_stats = {
        'accept_counts': accept_counts,
        'gamma': gamma,
    }

    return seq, generate_stats

def encode_tokens(tokenizer, string, bos=True, device=default_device):
    tokens = tokenizer.encode(string)
    if bos:
        tokens = [tokenizer.bos_id()] + tokens
    return torch.tensor(tokens, dtype=torch.int, device=device)

def benchmark_generation_speed(model, tokenizer, examples, generation_config, draft_model, compile):
    output_tokens_list = []
    generation_time_list = []
    num_generated_tokens_list = []
    start = -1 if compile else 0
    num_samples = len(examples)
    progress_bar = tqdm(range(start, num_samples))

    total_generate_stats = []
    if profile:
        torch.profiler._utils._init_for_cuda_graphs()
        prof = torch.profiler.profile(with_stack=False)
    else:
        prof = contextlib.nullcontext()
    
    with prof:
        for i in progress_bar:
            with torch.profiler.record_function(f"generate iter {i}"):
                random.seed(42)
                torch.manual_seed(42)
                conv = get_conversation_template('llama-2')
                qs = examples[0]["turns"][0] # 0 : single turn, same input
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = torch.as_tensor(tokenizer([prompt]).input_ids).to(default_device)
                # input_ids = examples[idx]["input_ids"].to(default_device)
                if i==0:
                    torch.cuda.reset_peak_memory_stats()
                device_sync(default_device) # MKG
                start = time.perf_counter()
                # if (i != num_samples - 1 or not profile):
                #     prof = contextlib.nullcontext()
                # else:
                output_ids, generate_stats = generate(
                    model,
                    input_ids,
                    generation_config.max_new_tokens,
                    interactive=False,
                    draft_model=draft_model,
                    speculate_k=generation_config.num_assistant_tokens,
                    callback=lambda x:x,
                    temperature=generation_config.temperature,
                    top_k=generation_config.top_k,
                    kv_cache_quantization=False,
                    cache_size=None,
                    linear_causal_mask=False,
                )
                total_generate_stats.append(generate_stats)
                # print(f"output_ids.shape : {output_ids.shape}")
            if i == -1:
                print(f"Compilation time: {time.perf_counter() - start:.2f} seconds")
                continue
            device_sync(default_device) # MKG
            end = time.perf_counter()

            output_tokens_list.append(output_ids)
            generation_time_list.append(end - start)
            num_generated_tokens = 0
            num_generated_tokens += len(
                [token_id for token_id in output_ids[input_ids.numel() :] if token_id != tokenizer.pad_token_id]
                )
            num_generated_tokens_list.append(num_generated_tokens)

            output_text = tokenizer.decode(output_ids)
            # import code; code.interact('check output', local=dict(globals(), **locals()))
            progress_bar.set_postfix(
                num_tokens=num_generated_tokens_list[-1],
                time=generation_time_list[-1],
                speed=f"{num_generated_tokens_list[-1] / generation_time_list[-1]:.3f} tokens/s",
            )

    if profile:
        prof.export_chrome_trace(f"241226_profile_3bit.json")
    total_tokens = sum(num_generated_tokens_list)
    total_seconds = sum(generation_time_list)
    logger.info(
        f"generated {total_tokens} tokens using {total_seconds:.3f} seconds, "
        f"generation speed: {total_tokens / total_seconds:.3f} tokens/s"
    )
    # import code; code.interact('end',local=dict(globals(), **locals()))

def main():
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--quantized_path", type=str, default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--backend", choices=['AUTO', 'TRITON', 'EXLLAMA', 'EXLLAMA_V2', 'MARLIN', 'BITBLAS'])
    parser.add_argument("--use_fast_tokenizer", action="store_true")
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--per_gpu_max_memory", type=int, default=None)
    parser.add_argument("--cpu_max_memory", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--num_beams", type=int, default=1)
    # added
    parser.add_argument("--bit", type=int, default=16)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')
    parser.add_argument('--compile_prefill', action='store_true', help='Whether to compile the prefill (improves prefill perf, but higher compile times)')
    parser.add_argument("--draft_model_name_or_path", type=str)
    parser.add_argument("--speculate_k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument('--profile', action='store_true', help='Whether to profile model forward.')
    args = parser.parse_args()
    
    max_memory = {}
    if args.per_gpu_max_memory is not None and args.per_gpu_max_memory > 0:
        if torch.cuda.is_available():
            max_memory.update({i: f"{args.per_gpu_max_memory}GIB" for i in range(torch.cuda.device_count())})
    if args.cpu_max_memory is not None and args.cpu_max_memory > 0 and max_memory:
        max_memory["cpu"] = f"{args.cpu_max_memory}GIB"
    if not max_memory:
        max_memory = None

    global profile
    profile = args.profile
    
    logger.info(f"max_memory: {max_memory}")
    logger.info("loading model and tokenizer")
    start = time.perf_counter()
    
    if os.environ.get('FUSED'):
        pth_name = 'model_fused.pth'
    else:
        pth_name = 'model.pth'
        
    # for fast model loading
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    
    torchao.quantization.utils.recommended_inductor_config_setter()
    config = ModelArgs.from_name(args.model_name_or_path)
    # config.n_layer = 1
    model = Transformer(config)
    model.dtype = torch.half
    model.device = device
    model.load_state_dict(torch.load(Path(args.model_name_or_path) / pth_name, mmap=True, weights_only=True), assign=True, strict=False)
    
    tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=args.model_name_or_path,
    use_fast=args.use_fast_tokenizer,
    trust_remote_code=args.trust_remote_code,
    )
    
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = model.half().to(device)
    # HQQ
    if args.bit < 16:
        quant_config = BaseQuantizeConfig(nbits=args.bit, group_size=args.group_size)
        AutoHQQHFModel.quantize_model(model, quant_config=quant_config, compute_dtype=torch.half, device='cuda')
        prepare_for_inference(model, backend="bitblas" if not args.bit == 3 else "gptq")

    model.eval()
    # mtbench questions
    examples = load_questions(glob('**/mtbench_question.jsonl', recursive=True)[0], 0, args.num_samples) # single question
    
    end = time.perf_counter()
    logger.info(f"model and tokenizer loading time: {end - start:.4f}s")
        
    def get_memory_footprint(module, return_buffers=True):
        mem = sum([param.nelement() * param.element_size() for param in module.parameters()])
        if return_buffers:
            mem_bufs = sum([buf.nelement() * buf.element_size() for buf in module.buffers()])
            mem = mem + mem_bufs
        return mem
    print(f"Model memory : {get_memory_footprint(model) / 1024 / 1024 / 1024} GB")
    # import code; code.interact(f'model', local=dict(globals(), **locals()))
    
    is_speculative = args.draft_model_name_or_path is not None
    if is_speculative:
        draft_model_torchao = Transformer.from_name(Path(args.draft_model_name_or_path))
        draft_model_torchao.load_state_dict(torch.load(Path(args.draft_model_name_or_path) / pth_name, mmap=True, weights_only=True), assign=True)
        draft_model = draft_model_torchao.half().to(device)
    else:
        draft_model = None
        
    if args.compile:
        print("Compiling Model")
        
        if is_speculative:
            global model_forward, logits_to_prob
            model_forward = torch.compile(model_forward, mode="reduce-overhead", fullgraph=True)
            
        global decode_one_token, prefill
        decode_one_token = torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=True)

        if args.compile_prefill:
            prefill = torch.compile(prefill, fullgraph=True, dynamic=True)

    generation_config = GenerationConfig(
        num_beams=args.num_beams,
        num_return_sequences=args.num_beams,
        do_sample=args.do_sample,
        min_new_tokens=args.max_new_tokens,
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        num_assistant_tokens=args.speculate_k,
        temperature=args.temperature,
    )
    
    if args.temperature < 1e-5:
        generation_config.top_k = None
    logger.info(f"generation config: {generation_config.to_dict()}")

    logger.info("benchmark generation speed")
    benchmark_generation_speed(model, tokenizer, examples, generation_config, draft_model, args.compile)

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    main()
