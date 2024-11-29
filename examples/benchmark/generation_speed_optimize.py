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

from fastchat.llm_judge.common import load_questions
from fastchat.model import get_conversation_template

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

from torchao._models.llama.model import Transformer, prepare_inputs_for_model

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
    logits = model(x, input_pos)
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
        
        # import code; code.interact('speculative_decode', local=dict(globals(), **locals()))
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
    next_token = prefill(model, x, input_pos, **sampling_kwargs).clone()
    if is_speculative:
        prefill(draft_model, x, input_pos, **sampling_kwargs)
        
    seq[T] = next_token
    # execute token generation
    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    accept_counts = [0] * (speculate_k + 1)
    gamma = 0
    # import code; code.interact('generate', local=dict(globals(), **locals()))
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

def load_data(tokenizer, n_samples, max_new_tokens):
    data_dict = load_dataset("ModelCloud/alpaca-data-cleaned", data_files="alpaca_data_cleaned.json", split="train")

    datas = [
        {
            'input': item['input'],
            'output': item['output'],
            'instruction': item['instruction']
        }
        for item in data_dict
    ]

    raw_data = random.sample(datas, k=min(n_samples, len(datas)))

    def dummy_gen():
        return raw_data

    def tokenize(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]

        prompts = []
        texts = []
        input_ids = []
        attention_mask = []
        for istr, inp, opt in zip(instructions, inputs, outputs):
            if inp:
                prompt = f"Instruction:\n{istr}\nInput:\n{inp}\nOutput:\n"
                text = prompt + opt
            else:
                prompt = f"Instruction:\n{istr}\nOutput:\n"
                text = prompt + opt
            if len(tokenizer(prompt)["input_ids"]) >= tokenizer.model_max_length - max_new_tokens:
                continue

            tokenized_data = tokenizer(text)

            input_ids.append(tokenized_data["input_ids"][: tokenizer.model_max_length])
            attention_mask.append(tokenized_data["attention_mask"][: tokenizer.model_max_length])
            prompts.append(prompt)
            texts.append(text)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt": prompts,
        }

    dataset = Dataset.from_generator(dummy_gen)

    dataset = dataset.map(
        tokenize,
        batched=True,
        batch_size=len(dataset),
        num_proc=1,
        keep_in_memory=True,
        load_from_cache_file=False,
        remove_columns=["instruction", "input"],
    )

    dataset = dataset.to_list()

    for sample in dataset:
        sample["input_ids"] = torch.LongTensor(sample["input_ids"])
        sample["attention_mask"] = torch.LongTensor(sample["attention_mask"])

    return dataset

def load_model_tokenizer(
    model_name_or_path: str,
    backend: BACKEND,
    tokenizer_name_or_path: Optional[str] = None,
    from_pretrained: bool = False,
    max_memory: Optional[dict] = None,
    model_basename: Optional[str] = None,
    quantize_config: Optional[str] = None,
    trust_remote_code: bool = False,
    use_safetensors: bool = True,
    use_fast_tokenizer: bool = False,
    autogptq: bool = False,
):
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=tokenizer_name_or_path or model_name_or_path,
        use_fast=use_fast_tokenizer,
        trust_remote_code=trust_remote_code,
    )
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if autogptq:
        from auto_gptq import AutoGPTQForCausalLM
        AUTO_MODEL = AutoGPTQForCausalLM
    else:
        AUTO_MODEL = GPTQModel
        
    if from_pretrained:
        model = AUTO_MODEL.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path,
            quantize_config=QuantizeConfig(),
            max_memory=max_memory,
            trust_remote_code=trust_remote_code,
        )
    else:
        model = AUTO_MODEL.from_quantized(
            model_name_or_path,
            max_memory=max_memory,
            quantize_config=quantize_config,
            model_basename=model_basename,
            use_safetensors=use_safetensors,
            trust_remote_code=trust_remote_code,
            backend=backend,
        )

    return model, tokenizer


def benchmark_generation_speed(model, tokenizer, examples, generation_config, draft_model, compile):
    profile = False
    output_tokens_list = []
    generation_time_list = []
    num_generated_tokens_list = []
    start = -1 if compile else 0
    num_samples = len(examples)
    progress_bar = tqdm(range(start, num_samples))

    total_generate_stats = []

    for i in progress_bar:
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
        if (i != num_samples - 1 or not profile):
            prof = contextlib.nullcontext()
        else:
            torch.profiler._utils._init_for_cuda_graphs()
            prof = torch.profiler.profile()
        with prof:
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
        if hasattr(prof, "export_chrome_trace"):
            prof.export_chrome_trace(f"{profile}.json")
        device_sync(default_device) # MKG
        end = time.perf_counter()

        output_tokens_list.append(output_ids)
        generation_time_list.append(end - start)
        num_generated_tokens = 0
        num_generated_tokens += len(
            [token_id for token_id in output_ids[input_ids.numel() :] if token_id != tokenizer.pad_token_id]
            )
        num_generated_tokens_list.append(num_generated_tokens)

        progress_bar.set_postfix(
            num_tokens=num_generated_tokens_list[-1],
            time=generation_time_list[-1],
            speed=f"{num_generated_tokens_list[-1] / generation_time_list[-1]:.3f} tokens/s",
        )

    total_tokens = sum(num_generated_tokens_list)
    total_seconds = sum(generation_time_list)
    logger.info(
        f"generated {total_tokens} tokens using {total_seconds:.3f} seconds, "
        f"generation speed: {total_tokens / total_seconds:.3f} tokens/s"
    )
    # import code; code.interact('end',local=dict(globals(), **locals()))

def _load_model(checkpoint_path, device, precision):
    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    if "model" in checkpoint and "stories" in str(checkpoint_path):
        checkpoint = checkpoint["model"]

    model = Transformer.from_name(checkpoint_path.parent.name)
    model.load_state_dict(checkpoint, assign=True)
    model = model.to(device=device, dtype=precision)

    return model.eval()

def main():
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--from_pretrained", action="store_true")
    parser.add_argument("--model_basename", type=str, default=None)
    parser.add_argument("--quantize_config_save_dir", type=str, default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--backend", choices=['AUTO', 'TRITON', 'EXLLAMA', 'EXLLAMA_V2', 'MARLIN', 'BITBLAS'])
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--use_fast_tokenizer", action="store_true")
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--per_gpu_max_memory", type=int, default=None)
    parser.add_argument("--cpu_max_memory", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--num_beams", type=int, default=1)
    # added
    parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')
    parser.add_argument('--compile_prefill', action='store_true', help='Whether to compile the prefill (improves prefill perf, but higher compile times)')
    parser.add_argument("--draft_checkpoint_path", type=str, default=None)
    parser.add_argument("--speculate_k", type=int, default=5)
    parser.add_argument('--autogptq', action='store_true', help='use AutoGPTQ, not GPTQModel')
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()
    
    max_memory = {}
    if args.per_gpu_max_memory is not None and args.per_gpu_max_memory > 0:
        if torch.cuda.is_available():
            max_memory.update({i: f"{args.per_gpu_max_memory}GIB" for i in range(torch.cuda.device_count())})
    if args.cpu_max_memory is not None and args.cpu_max_memory > 0 and max_memory:
        max_memory["cpu"] = f"{args.cpu_max_memory}GIB"
    if not max_memory:
        max_memory = None

    logger.info(f"max_memory: {max_memory}")
    
    quantize_config = None
    if args.quantize_config_save_dir:
        quantize_config = QuantizeConfig.from_pretrained(args.quantize_config_save_dir)

    if args.use_safetensors:
        logger.warning(
            "The command --use_safetensors is deprecated and will be removed in the next release. It is now by default activated."
        )

    logger.info("loading model and tokenizer")
    start = time.perf_counter()
    
    if os.environ.get('FUSED'):
        pth_name = 'model_fused.pth'
    else:
        pth_name = 'model.pth'
        
    # fast model
    torchao.quantization.utils.recommended_inductor_config_setter()
    model_torchao = Transformer.from_name(Path('/SSD/JG/checkpoints') / args.tokenizer_name_or_path)
    model_torchao.load_state_dict(torch.load(Path('/SSD/JG/checkpoints') / args.tokenizer_name_or_path / pth_name, mmap=True, weights_only=True), assign=True)
    
    if not args.from_pretrained:
        model, tokenizer = load_model_tokenizer(
            model_name_or_path=args.model_name_or_path,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            from_pretrained=args.from_pretrained,
            max_memory=max_memory,
            model_basename=args.model_basename,
            quantize_config=quantize_config,
            trust_remote_code=args.trust_remote_code,
            use_safetensors=True,
            use_fast_tokenizer=args.use_fast_tokenizer,
            backend=get_backend(args.backend),
            autogptq=args.autogptq,
        )
        logger.info(f"model quantized: {model.quantized}")
        logger.info(f"quantize config: {model.quantize_config.to_dict()}")
        logger.info(f"model device map: {model.hf_device_map}")
        logger.info("loading data")

        for i in range(len(model.model.model.layers)):
            layer = model.model.model.layers[i]
            torchao_layer = model_torchao.layers[i]
            # replace linear layer
            # import copy
            setattr(torchao_layer.attention, 'wq', layer.self_attn.q_proj)
            setattr(torchao_layer.attention, 'wk', layer.self_attn.k_proj)
            setattr(torchao_layer.attention, 'wv', layer.self_attn.v_proj)
            setattr(torchao_layer.attention, 'wo', layer.self_attn.o_proj)
            setattr(torchao_layer.feed_forward, 'w1', layer.mlp.gate_proj)
            setattr(torchao_layer.feed_forward, 'w3', layer.mlp.up_proj)
            setattr(torchao_layer.feed_forward, 'w2', layer.mlp.down_proj)
            
        del model
    else:
        tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.tokenizer_name_or_path or args.model_name_or_path,
        use_fast=args.use_fast_tokenizer,
        trust_remote_code=args.trust_remote_code,
        )
        
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        
    model = model_torchao
    model.eval()
    model = model.half().to(device)
        
    if args.autogptq:
        for layer in model.layers:
            layer.attention.wq.post_init()
            layer.attention.wk.post_init()
            layer.attention.wv.post_init()
            layer.attention.wo.post_init()
            layer.feed_forward.w1.post_init()
            layer.feed_forward.w3.post_init()
            layer.feed_forward.w2.post_init()
    
    # mtbench questions
    examples = load_questions('/NAS/JG/QAS4SD/GPTQModel/examples/benchmark/mtbench_question.jsonl', 0, args.num_samples) # single question
    # examples = load_data(
    #     tokenizer,
    #     args.num_samples,
    #     args.max_new_tokens,
    # )
    
    end = time.perf_counter()
    logger.info(f"model and tokenizer loading time: {end - start:.4f}s")
        
    # from torchao.quantization.quant_api import (
    #         quantize_,
    #         int4_weight_only,
    #     )
    # model = model.to(device=device, dtype=torch.bfloat16)
    # quantize_(model, int4_weight_only(group_size=128))
    # import code; code.interact(f'model', local=dict(globals(), **locals()))
    
    is_speculative = args.draft_checkpoint_path is not None
    if is_speculative:
        if '68m' in args.draft_checkpoint_path:
            draft_model = _load_model(Path(args.draft_checkpoint_path) / pth_name, device, torch.half)
        else:
            draft_model_torchao = Transformer.from_name(Path('/SSD/JG/checkpoints') / args.tokenizer_name_or_path)
            draft_model_torchao.load_state_dict(torch.load(Path('/SSD/JG/checkpoints') / args.tokenizer_name_or_path / pth_name, mmap=True, weights_only=True), assign=True)
            
            draft_model, _ = load_model_tokenizer(
                model_name_or_path=args.draft_checkpoint_path,
                tokenizer_name_or_path=args.tokenizer_name_or_path,
                from_pretrained=False,
                max_memory=max_memory,
                model_basename=args.model_basename,
                quantize_config=quantize_config,
                trust_remote_code=args.trust_remote_code,
                use_safetensors=True,
                use_fast_tokenizer=args.use_fast_tokenizer,
                backend=get_backend(args.backend),
                autogptq=args.autogptq,
            )

            for i in range(len(draft_model.model.model.layers)):
                layer = draft_model.model.model.layers[i]
                torchao_layer = draft_model_torchao.layers[i]
                # replace linear layer
                # import copy
                setattr(torchao_layer.attention, 'wq', layer.self_attn.q_proj)
                setattr(torchao_layer.attention, 'wk', layer.self_attn.k_proj)
                setattr(torchao_layer.attention, 'wv', layer.self_attn.v_proj)
                setattr(torchao_layer.attention, 'wo', layer.self_attn.o_proj)
                setattr(torchao_layer.feed_forward, 'w1', layer.mlp.gate_proj)
                setattr(torchao_layer.feed_forward, 'w3', layer.mlp.up_proj)
                setattr(torchao_layer.feed_forward, 'w2', layer.mlp.down_proj)
                
            del draft_model
                
            draft_model = draft_model_torchao
            draft_model.eval()
            draft_model = draft_model.half().to(device)
                
            if args.autogptq:
                for layer in draft_model.layers:
                    layer.attention.wq.post_init()
                    layer.attention.wk.post_init()
                    layer.attention.wv.post_init()
                    layer.attention.wo.post_init()
                    layer.feed_forward.w1.post_init()
                    layer.feed_forward.w3.post_init()
                    layer.feed_forward.w2.post_init()
    else:
        draft_model = None
        
    if args.compile:
        print("Compiling Model")
        
        if is_speculative:
            global model_forward, logits_to_prob
            model_forward = torch.compile(model_forward, mode="reduce-overhead", fullgraph=True)
            
        global decode_one_token, prefill
        # model = torch.compile(model)
        decode_one_token = torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=True)
        # decode_one_token = torch.compile(decode_one_token, mode="reduce-overhead")

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
