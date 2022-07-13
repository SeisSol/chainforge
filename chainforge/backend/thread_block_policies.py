from abc import ABC, abstractmethod
from chainforge.common import Context


class AbstractThreadBlockPolicy(ABC):
  def __init__(self, context: Context, mem_per_mult: int, num_threads: int):
    self._context: Context = context
    self._mem_per_mult: int = mem_per_mult
    self._num_threads: int = num_threads

    vm = self._context.get_vm()
    self._max_blocks = vm.hw_descr.max_block_per_sm
    self._max_allowed_mem = vm.hw_descr.max_local_mem_size_per_block

  @abstractmethod
  def get_num_mults_per_block(self):
    pass


class SimpleThreadBlockPolicy(AbstractThreadBlockPolicy):
  def __init__(self, context, mem_size_per_mult, num_threads):
    super().__init__(context, mem_size_per_mult, num_threads)

  def get_num_mults_per_block(self):
    if self._num_threads <= 32:
      return 2
    else:
      return 1


class NvidiaAmdThreadBlockPolicy(AbstractThreadBlockPolicy):
  def __init__(self, context, mem_size_per_mult, num_threads):
    super().__init__(context, mem_size_per_mult, num_threads)

  def _estimate_num_registers_per_mult(self, num_active_threads):
    # Note: derived experimentally
    factor = self._context.bytes_per_real() / 4
    return factor * (32 + num_active_threads)

  def get_num_mults_per_block(self):

    hw_descr = self._context.get_vm().hw_descr

    max_num_regs_per_thread = self._estimate_num_registers_per_mult(self._num_threads)
    shr_mem_bytes = self._mem_per_mult * self._context.bytes_per_real()
    mults_wrt_shr_mem = hw_descr.max_local_mem_size_per_block / shr_mem_bytes
    mults_wrt_num_regs = hw_descr.max_reg_per_block / (self._num_threads * max_num_regs_per_thread)
    mults_per_sm = int(min(mults_wrt_shr_mem, mults_wrt_num_regs))
    return max(int(mults_per_sm / hw_descr.max_block_per_sm), 1)


def get_thread_policy(context: Context):
  manufacturer = context.get_vm().hw_descr.manufacturer
  if (manufacturer == 'nvidia' or manufacturer == 'amd'):
    return NvidiaAmdThreadBlockPolicy
  else:
    return SimpleThreadBlockPolicy