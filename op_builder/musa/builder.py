# Copyright (c) Microsoft Corporation.
# Copyright (c) Moore Threads Technology  Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os

try:
    import torch_musa  # noqa: F401
except ImportError as e:
    pass

try:
    # is op_builder from deepspeed or a 3p version? this should only succeed if it's deepspeed
    # if successful this also means we're doing a local install and not JIT compile path
    from op_builder import __deepspeed__  # noqa: F401 # type: ignore
    from op_builder.builder import OpBuilder
except ImportError:
    from deepspeed.ops.op_builder.builder import OpBuilder


class MUSAOpBuilder(OpBuilder):

    def builder(self):
        from torch_musa.utils.musa_extension import MUSAExtension as ExtensionBuilder
        include_dirs = [os.path.abspath(x) for x in self.strip_empty_entries(self.include_paths())]
        compile_args = {'cxx': self.strip_empty_entries(self.cxx_args()), \
                        'mcc': self.strip_empty_entries(self.mcc_args())}
        musa_ext = ExtensionBuilder(name=self.absolute_name(),
                                    sources=self.strip_empty_entries(self.sources()),
                                    include_dirs=include_dirs,
                                    libraries=self.strip_empty_entries(self.libraries_args()),
                                    extra_compile_args=compile_args)

        return musa_ext

    def cxx_args(self):
        return ['-O3', '-std=c++17', '-g', '-Wno-reorder']

    def mcc_args(self):
        args = ['-O2']
        std_lib = '-std=c++17'
        args += ['', std_lib]

        return args

    def libraries_args(self):
        return []
