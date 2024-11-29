from gptqmodel import GPTQModel, QuantizeConfig
from transformers import AutoTokenizer
from argparse import ArgumentParser

from gptqmodel.quantization.config import FORMAT
from gptqmodel.utils.backend import BACKEND
import torch

def get_c4(nsamples, seed, seqlen, tokenizer):
    from datasets import load_dataset
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    
    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'])
            if len(trainenc.input_ids) >= seqlen:
                break
        i = random.randint(0, len(trainenc.input_ids) - seqlen - 1)
        j = i + seqlen
        trainenc.input_ids = trainenc.input_ids[i:j]
        trainenc.attention_mask = trainenc.attention_mask[i:j]
        trainloader.append(trainenc)

    return trainloader

@torch.no_grad()
def calculate_avg_ppl(model, tokenizer):
    from gptqmodel.utils import Perplexity

    ppl = Perplexity(
        model=model,
        tokenizer=tokenizer,
        dataset_path="wikitext",
        dataset_name="wikitext-2-raw-v1",
        split="train",
        text_column="text",
    )

    all = ppl.calculate(n_ctx=512, n_batch=512)

    # average ppl
    avg = sum(all) / len(all)

    return avg

def main():
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--bits", type=int)
    parser.add_argument("--group_size", type=int)
    parser.add_argument("--asym", action='store_true')
    parser.add_argument("--backend", choices=['AUTO', 'TRITON', 'EXLLAMA', 'EXLLAMA_V2', 'MARLIN', 'BITBLAS'])
    parser.add_argument("--save_quantized_path", type=str)
    parser.add_argument("--test", action='store_true')
    args = parser.parse_args()
    
    pretrained_model_id = args.model_name_or_path
    quantized_model_id = args.save_quantized_path
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_id, use_fast=True)
    if not args.test:
        # calibration_dataset = [
        #     tokenizer(
        #         "gptqmodel is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
        #     )
        # ]
        calibration_dataset = get_c4(nsamples=128, seed=42, seqlen=2048, tokenizer=tokenizer)

        quantize_config = QuantizeConfig(
            bits=args.bits,  # quantize model to 4-bit
            group_size=args.group_size,  # it is recommended to set the value to 128
            desc_act=False,
            true_sequential=True,
            sym=not args.asym,
            format=getattr(FORMAT, args.backend),
            runtime_format=getattr(FORMAT, args.backend)
            # parallel_packing=True,
        )

        # load un-quantized model, by default, the model will always be loaded into CPU memory
        model = GPTQModel.from_pretrained(pretrained_model_id, quantize_config)

        # quantize model, the calibration_dataset should be list of dict whose keys can only be "input_ids" and "attention_mask"
        model.quantize(calibration_dataset)

        # save quantized model
        model.save_quantized(quantized_model_id)
    else:
        model = GPTQModel.from_quantized(quantized_model_id, device="cuda:0")
        # model = GPTQModel.from_pretrained(pretrained_model_id, quantize_config=None).to("cuda")
        print(tokenizer.decode(model.generate(**tokenizer("gptqmodel is", return_tensors="pt").to(model.device))[0]))
    print(f"Quantized Model {quantized_model_id} avg PPL is {calculate_avg_ppl(model, tokenizer)}")
    # push quantized model to Hugging Face Hub.
    # to use use_auth_token=True, Login first via huggingface-cli login.
    # or pass explcit token with: use_auth_token="hf_xxxxxxx"
    # (uncomment the following three lines to enable this feature)
    # repo_id = f"YourUserName/{quantized_model_dir}"
    # commit_message = f"GPTQModel model for {pretrained_model_dir}: {quantize_config.bits}bits, gr{quantize_config.group_size}, desc_act={quantize_config.desc_act}"
    # model.push_to_hub(repo_id, commit_message=commit_message, use_auth_token=True)

    # alternatively you can save and push at the same time
    # (uncomment the following three lines to enable this feature)
    # repo_id = f"YourUserName/{quantized_model_dir}"
    # commit_message = f"GPTQModel model for {pretrained_model_dir}: {quantize_config.bits}bits, gr{quantize_config.group_size}, desc_act={quantize_config.desc_act}"
    # model.push_to_hub(repo_id, save_dir=quantized_model_dir, use_safetensors=True, commit_message=commit_message, use_auth_token=True)

    # # save quantized model using safetensors
    # model.save_quantized(quantized_model_id, use_safetensors=True)

    # # load quantized model to the first GPU
    # model = GPTQModel.from_quantized(quantized_model_id, device="cuda:0")

    # # load quantized model to CPU with QBits kernel linear.
    # # model = GPTQModel.from_quantized(quantized_model_dir, device="cpu")

    # # download quantized model from Hugging Face Hub and load to the first GPU
    # # model = GPTQModel.from_quantized(repo_id, device="cuda:0", use_safetensors=True,)

    # # inference with model.generate
    # print(tokenizer.decode(model.generate(**tokenizer("gptqmodel is", return_tensors="pt").to(model.device))[0]))


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    main()
