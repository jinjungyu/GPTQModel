from gptqmodel import GPTQModel, QuantizeConfig
from transformers import AutoTokenizer
from argparse import ArgumentParser

from gptqmodel.quantization.config import FORMAT
from gptqmodel.utils.backend import BACKEND


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--bits", type=int)
    parser.add_argument("--group_size", type=int)
    parser.add_argument("--asym", action='store_true')
    parser.add_argument("--save_quantized_path", type=str)
    args = parser.parse_args()
    
    pretrained_model_id = args.model_name_or_path
    quantized_model_id = args.save_quantized_path
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_id, use_fast=True)
    calibration_dataset = [
        tokenizer(
            "gptqmodel is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
        )
    ]

    quantize_config = QuantizeConfig(
        bits=args.bits,  # quantize model to 4-bit
        group_size=args.group_size,  # it is recommended to set the value to 128
        desc_act=False,
        sym=not args.asym,
        format=FORMAT.GPTQ,
        runtime_format=FORMAT.GPTQ,
        # parallel_packing=True,
    )

    # load un-quantized model, by default, the model will always be loaded into CPU memory
    model = GPTQModel.from_pretrained(pretrained_model_id, quantize_config)

    # quantize model, the calibration_dataset should be list of dict whose keys can only be "input_ids" and "attention_mask"
    model.quantize(calibration_dataset)

    # save quantized model
    model.save_quantized(quantized_model_id)

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
