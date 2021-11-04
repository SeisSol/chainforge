from math import ceil
from chainforge.common.vm import VM, vm_factory
from chainforge.common.basic_types import FloatingPointType


class Options:
  def __init__(self,
               exact_contraction_length=False,
               prefetch_gemm=False,
               align_shr_mem=True,
               enable_sync_threads_opt=True):
    self.exact_contraction_length: bool = exact_contraction_length
    self.prefetch_gemm: bool = prefetch_gemm
    self.align_shr_mem: bool = align_shr_mem
    self.enable_sync_threads_opt = enable_sync_threads_opt


class Context:
  def __init__(self,
               arch: str,
               backend: str,
               fp_type: FloatingPointType,
               options: Options = Options()):
    self._vm: VM = vm_factory(arch, backend)
    self.fp_type = fp_type
    self._options = options

  def set_fp_type(self, fp_type: FloatingPointType):
    self.fp_type = fp_type

  def fp_as_str(self):
    return FloatingPointType.as_str(self.fp_type)

  def get_vm(self):
    return self._vm

  def get_user_options(self):
    return self._options

  def align(self, num):
    fp_size = 4 if self.fp_type == FloatingPointType.FLOAT else 8
    hw_fp_word_size = self._vm.hw_descr.hw_fp_word_size
    vec_unit_length = self._vm.hw_descr.vec_unit_length

    align_length = (vec_unit_length * hw_fp_word_size) / fp_size
    return int(ceil(num / align_length) * align_length)
