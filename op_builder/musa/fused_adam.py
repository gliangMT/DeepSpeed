# Copyright (c) Microsoft Corporation.
# Copyright (c) Moore Threads Technology  Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
from .builder import MUSAOpBuilder


class FusedAdamBuilder(MUSAOpBuilder):
    BUILD_VAR = "DS_BUILD_FUSED_ADAM"
    NAME = "fused_adam"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.adam.{self.NAME}_op'

    def sources(self):
        return ['csrc/musa/adam/fused_adam_frontend.cpp', 'csrc/musa/adam/multi_tensor_adam.mu']

    def include_paths(self):
        return ['csrc/musa/includes', 'csrc/musa/adam']

    def cxx_args(self):
        args = super().cxx_args()
        return args

    def mcc_args(self):
        mcc_args = super().mcc_args()
        return mcc_args
