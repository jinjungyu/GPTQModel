# License: GPTQModel/licenses/LICENSE.apache

import ctypes
import operator
import os
from functools import reduce
from logging import getLogger
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from torch import Tensor

logger = getLogger(__name__)

BITBLAS_TARGET = None
BITBLAS_DATABASE_PATH = None
BITBLAS_PROPAGATE_WEIGHTS = False

try:
    import bitblas  # noqa: F401
    BITBLAS_AVAILABLE = True
except Exception:
    BITBLAS_AVAILABLE = False

BITBLAS_INSTALL_HINT = "bitblas not installed. Please install via `pip install bitblas`."


def import_bitblas():
    # print("import_bitblas() called")
    global BITBLAS_DATABASE_PATH, BITBLAS_TARGET

    # guard against bitblas pip whl incompatible env`
    import bitblas

    bitblas.set_log_level("INFO")

    if BITBLAS_TARGET is None:
        from .bitblas_target_detector import patched_auto_detect_nvidia_target

        bitblas.auto_detect_nvidia_target = patched_auto_detect_nvidia_target
        BITBLAS_TARGET = patched_auto_detect_nvidia_target(int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")))
        os.environ["TVM_TARGET"] = f"{BITBLAS_TARGET}"
        print(f"BITBLAS_TARGET {BITBLAS_TARGET}")

    if BITBLAS_DATABASE_PATH is None:
        # from importlib.metadata import version

        from bitblas.cache import get_database_path

        # bitblas_version = version(distribution_name="bitblas")
        # gptqmodel_version = version(distribution_name="gptqmodel")

        BITBLAS_DATABASE_PATH = get_database_path()


def unpack_qzeros(qzeros, bits):
    qzeros = qzeros.view(torch.int32)
    elems_per_int32 = 32 // bits
    unpacked_zeros = torch.zeros(
        (qzeros.shape[0], qzeros.shape[1] * elems_per_int32),
        dtype=torch.int8,
        device=qzeros.device,
        requires_grad=False,
    )

    for col in range(unpacked_zeros.shape[1]):
        i = col % elems_per_int32
        unpacked_zeros[:, col] = (qzeros[:, col // elems_per_int32] >> (bits * i)) & 0xF

    return unpacked_zeros

def unpack_qweight(qweight, bits):
    qweight = qweight.view(torch.int8)
    elems_per_int8 = 8 // bits
    unpacked_weight = torch.zeros(
        (qweight.shape[0], qweight.shape[1] * elems_per_int8),
        dtype=torch.int8,
        device=qweight.device,
        requires_grad=False,
    )
    for col in range(unpacked_weight.shape[1]):
        i = col % elems_per_int8
        unpacked_weight[:, col] = (qweight[:, col // elems_per_int8] >> (bits * i))

    return torch.bitwise_and(unpacked_weight, 2**bits - 1)
    
@torch.library.custom_op("bitblas::matmul_bitblas", mutates_args=())
def matmul_bitblas(
    x: Tensor, W_q: Tensor, scale: Tensor, zero: Tensor, out_features: int, eng_tag: str,
) -> Tensor:

    origin_x_size = x.size()
    c = BitBLASQuantLinear.ENG_CACHE[eng_tag](x, W_q, scale=scale, zeros=zero)
    new_shape = origin_x_size[:-1] + (out_features,)
    c = c.reshape(new_shape)
    return c


@torch.library.register_fake("bitblas::matmul_bitblas")
def matmul_bitblas_fake(
    x: Tensor, W_q: Tensor, scale: Tensor, zero: Tensor, out_features: int, eng_tag: str,
) -> Tensor:
    return torch.empty(
        [x.shape[0], x.shape[1], out_features], device=W_q.device, dtype=scale.dtype
    )

class BitBLASQuantLinear(BaseQuantLinear):
    SUPPORTS_BITS = [1, 2, 4]
    SUPPORTS_DESC_ACT = [False]
    SUPPORTS_SHARDS = True
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [16]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [16]

    OPT_FEATURES = [1, 4, 5, 6, 16, 32, 64, 128, 256, 512]
    zeros_mode = "quantized"  # "original" or "rescale" or "quantized"
    TORCH_DTYPE = torch.float16
    STORAGE_DTYPE = "int8"  # assume int8 storage
    TORCH_STORAGE_DTYPE = getattr(torch, STORAGE_DTYPE)
    BITBLAS_DTYPES = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.half: "float16",
        torch.int8: "int8",
    }

    ENG_CACHE = {}
    
    def __init__(
        self,
        bits: int,
        group_size: int,
        desc_act: bool,
        sym: bool,
        infeatures: int,
        outfeatures: int,
        bias: bool,
        enable_tuning: bool = True,
        fast_decoding: bool = True,
        propagate_b: bool = BITBLAS_PROPAGATE_WEIGHTS,
        opt_features: Union[int, List[int]] = OPT_FEATURES,
        layout: str = "nt",
        **kwargs,
    ):
        super().__init__(bits=bits, group_size=group_size, sym=sym, desc_act=desc_act, infeatures=infeatures,outfeatures=outfeatures,**kwargs)

        import_bitblas()

        self._validate_parameters(group_size, infeatures, outfeatures)

        self.bits = bits
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.group_size = self._set_group_size(group_size, infeatures)
        self.opt_features = opt_features
        self.target = BITBLAS_TARGET
        # added
        self.eng_tag = (
            f"{infeatures}x{outfeatures}" + "_" + str(self.bits) + "_" + str(self.group_size)
        )
        self._configure_bitblas_matmul(
            enable_tuning, fast_decoding, bias, propagate_b, layout, bits
        )
        self._initialize_buffers(infeatures, outfeatures, bias)
        self.reset_parameters()
        

    def _validate_parameters(
        self, group_size: int, infeatures: int, outfeatures: int
    ):
        if infeatures % group_size != 0:
            raise ValueError("`infeatures` must be divisible by `group_size`.")

    def _set_group_size(self, group_size: int, infeatures: int):
        return infeatures if group_size == -1 else group_size

    def _initialize_buffers(self, infeatures: int, outfeatures: int, bias: bool):
        self.register_buffer(
            "qweight",
            torch.zeros(
                self.bitblas_matmul.retrieve_weight_shape(),
                dtype=self.TORCH_STORAGE_DTYPE,
            ),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (outfeatures, infeatures // self.group_size), dtype=self.TORCH_DTYPE
            ),
        )
        if self.zeros_mode == "quantized":
            storage_nbit = int("".join(c for c in self.STORAGE_DTYPE if c.isdigit()))
            self.register_buffer(
                "zeros",
                torch.zeros(
                    (infeatures // self.group_size, outfeatures // storage_nbit * self.bits), dtype=self.TORCH_STORAGE_DTYPE
                ),
            )
        else:
            self.register_buffer(
                "zeros",
                torch.zeros(
                    (outfeatures, infeatures // self.group_size), dtype=self.TORCH_DTYPE
                ),
            )

        if bias:
            self.register_buffer(
                "bias", torch.zeros((outfeatures), dtype=self.TORCH_DTYPE)
            )
        else:
            self.bias = None

    def _configure_bitblas_matmul(
        self, enable_tuning: bool, fast_decoding: bool, bias: bool, propagate_b, layout, bits: int
    ):
        from bitblas import MatmulConfig

        # Assuming MatmulWeightOnlyDequantizeConfig and MatmulWeightOnlyDequantize are defined elsewhere
        bitblas_dtype = self.BITBLAS_DTYPES[self.TORCH_DTYPE]
        W_dtype = f"uint{bits}"
        matmul_config = MatmulConfig(
            M=self.opt_features,
            N=self.outfeatures,
            K=self.infeatures,
            A_dtype=bitblas_dtype,
            W_dtype=W_dtype,
            out_dtype=bitblas_dtype,
            accum_dtype="int32" if bitblas_dtype == "int8" else bitblas_dtype,
            storage_dtype=self.STORAGE_DTYPE,
            with_scaling=True,
            with_zeros=True,
            group_size=self.group_size,
            with_bias=bias,
            layout=layout,
            zeros_mode=self.zeros_mode,
        )
        
        if self.eng_tag not in BitBLASQuantLinear.ENG_CACHE:
            self.bitblas_matmul = self._get_or_create_bitblas_operator(
                matmul_config, enable_tuning
            )
            BitBLASQuantLinear.ENG_CACHE[self.eng_tag] = self.bitblas_matmul
        else:
            self.bitblas_matmul = BitBLASQuantLinear.ENG_CACHE[self.eng_tag]

    def _get_or_create_bitblas_operator(self, config, enable_tuning):
        from bitblas import Matmul
        from bitblas.cache import global_operator_cache

        if global_operator_cache.size() == 0:
            global_operator_cache.load_from_database(
                BITBLAS_DATABASE_PATH, BITBLAS_TARGET
            )

        bitblas_matmul = global_operator_cache.get(config)
        if bitblas_matmul is None:
            bitblas_matmul = Matmul(config, target=self.target)
            if enable_tuning:
                bitblas_matmul.hardware_aware_finetune(topk=20)
                global_operator_cache.add(config, bitblas_matmul)
                global_operator_cache.save_into_database(
                    BITBLAS_DATABASE_PATH, BITBLAS_TARGET
                )
                logger.info(
                    "BitBLAS Tuning done, appended operator to global_operator_cache."
                )
            else:
                logger.info("BitBLAS Operator created.")
        else:
            logger.info("BitBLAS Operator found in global_operator_cache.")
        return bitblas_matmul

    def reset_parameters(self):
        # init for char
        self.qweight = torch.randint_like(
            self.qweight,
            0,
            2 ** (self.bits - 1) - 1,
            dtype=torch.int8,
            device=self.qweight.device,
        )
        nn.init.normal_(self.scales)
        nn.init.zeros_(self.zeros)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        self.q_params = None

    def post_init(self):
        self.validate_device(self.qweight.device.type)
        # # eliminate runtime overhead like exllama state
        # param_list = [self.qweight, self.scales, self.zeros]
        # if self.bitblas_matmul.config.with_bias:
        #     param_list.append(self.bias)
        # self.q_params = [ctypes.c_void_p(arr.data_ptr()) for arr in param_list]

    def pack(self, linear, scales, zeros, g_idx=None):
        from bitblas.quantization.utils import general_compress

        W = linear.weight.data.clone()

        scales = scales.t().contiguous()
        zeros = zeros.t().contiguous()
        scale_zeros = zeros * scales
        self.scales = scales.clone().half()
        if linear.bias is not None:
            self.bias = linear.bias.clone().half()

        intweight = []
        for idx in range(self.infeatures):
            g_idx = idx // self.group_size
            intweight.append(
                torch.round((W[:, idx] + scale_zeros[g_idx]) / self.scales[g_idx]).to(torch.int)[
                    :, None
                ]
            )
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)

        i = 0
        row = 0
        qweight = np.zeros((intweight.shape[0] // 32 * self.bits, intweight.shape[1]), dtype=np.uint32)
        while row < qweight.shape[0]:
            for j in range(i, i + (32 // self.bits)):
                qweight[row] |= intweight[j] << (self.bits * (j - i))
            i += 32 // self.bits
            row += 1

        qweight = qweight.astype(np.int32)
        qweight = torch.from_numpy(qweight)
        qweight = qweight.T.contiguous().view(self.TORCH_STORAGE_DTYPE)
        intweight = unpack_qweight(qweight, self.bits).contiguous()
        if self.bitblas_matmul.weight_transform is not None:
            qweight = self.bitblas_matmul.weight_transform(intweight.cpu()).cuda()
        self.qweight = qweight

        scales = self.scales.T.contiguous().view(self.TORCH_DTYPE)
        self.scales = scales

        zeros = zeros.numpy().astype(np.uint32)
        qzeros = np.zeros((zeros.shape[0], zeros.shape[1] // 32 * self.bits), dtype=np.uint32)
        i = 0
        col = 0
        while col < qzeros.shape[1]:
            for j in range(i, i + (32 // self.bits)):
                qzeros[:, col] |= zeros[:, j] << (self.bits * (j - i))
            i += 32 // self.bits
            col += 1

        qzeros = qzeros.astype(np.int32)
        self.qzeros = torch.from_numpy(qzeros)

        intzeros = unpack_qzeros(self.qzeros, self.bits).T.contiguous()
        if self.bitblas_matmul.config.zeros_mode == "original":
            self.zeros = intzeros.to(torch.float16).contiguous()
        elif self.bitblas_matmul.config.zeros_mode == "rescale":
            self.zeros[:, :] = intzeros.to(torch.float16)[:, :] * self.scales[:, :]
        elif self.bitblas_matmul.config.zeros_mode == "quantized":
            self.zeros = (
                torch.Tensor(
                    general_compress(intzeros.T.contiguous().cpu().numpy(), self.bits)
                )
                .to(self.qweight.device)
                .to(self.zeros.dtype)
                .contiguous()
            )
        else:
            raise ValueError(
                f"Unsupported zeros type: {self.bitblas_matmul.config.zeros_mode}"
            )

        if self.bias is not None:
            self.bias = self.bias.data.to(torch.float16).contiguous()

    def repack_from_gptq(self, gptq_module):
        from bitblas.quantization.utils import general_compress

        # qweight in gptq old quant linear stored with (outfeatures, infeatures), should be transposed.
        qweight = gptq_module.qweight.T.contiguous().view(self.TORCH_STORAGE_DTYPE)
        intweight = unpack_qweight(qweight, self.bits).contiguous()
        if self.bitblas_matmul.weight_transform is not None:
            qweight = self.bitblas_matmul.weight_transform(intweight.cpu()).cuda()
        self.qweight = qweight
        # scales in gptq old quant linear stored with (infeatures // group_size, outfeatures), should be transposed.
        scales = gptq_module.scales.T.contiguous().view(self.TORCH_DTYPE)
        self.scales = scales
        # qzeros should be de-quantized to int zeros.
        intzeros = unpack_qzeros(gptq_module.qzeros, self.bits).T.contiguous()
        if self.bitblas_matmul.config.zeros_mode == "original":
            self.zeros = intzeros.to(torch.float16).contiguous()
        elif self.bitblas_matmul.config.zeros_mode == "rescale":
            self.zeros[:, :] = intzeros.to(torch.float16)[:, :] * self.scales[:, :]
        elif self.bitblas_matmul.config.zeros_mode == "quantized":
            self.zeros = (
                torch.Tensor(
                    general_compress(intzeros.T.contiguous().cpu().numpy(), self.bits)
                )
                .to(self.qweight.device)
                .to(self.zeros.dtype)
                .contiguous()
            )
        else:
            raise ValueError(
                f"Unsupported zeros type: {self.bitblas_matmul.config.zeros_mode}"
            )
        if self.bias is not None:
            self.bias = gptq_module.bias.data.to(torch.float16).contiguous()
    
    def matmul(self, x: Tensor) -> Tensor:
        return matmul_bitblas(
            x.to(self.qweight.device), self.qweight, self.scales, self.zeros, self.outfeatures, self.eng_tag
        )
        
    def forward(self, x: Tensor) -> Tensor:
        out = self.matmul(x)
        if self.bias is not None:
            out += self.bias
        return out
    
    # def forward(self, A):
    #     if A.dtype != torch.float16:
    #         A = A.half()

    #     C = torch.empty(
    #         A.shape[:-1] + (self.scales.shape[0],), dtype=A.dtype, device=A.device
    #     )
    #     A_void = ctypes.c_void_p(A.data_ptr())
    #     # m is the product of the last n - 1 dimensions of A
    #     m = ctypes.c_int32(reduce(operator.mul, A.shape[:-1], 1))
    #     self.bitblas_matmul.call_lib(
    #         A_void , *self.q_params, ctypes.c_void_p(C.data_ptr()), m
    #     )
    #     return C


__all__ = ["BitBLASQuantLinear"]
