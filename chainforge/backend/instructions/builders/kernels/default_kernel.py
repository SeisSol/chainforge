from .base_kernel import BaseKernelBuilder
from .kernel_types import KernelType
from chainforge.backend.instructions import RegistersAllocBuilder
from chainforge.backend.instructions import ShrMemAllocBuilder
from chainforge.backend.instructions import GemmBuilder
from chainforge.common import Context, VM
from chainforge.backend.scopes import Scopes


class DefaultKernelBuilder(BaseKernelBuilder):
  """ This is a class for building shared-memory-based gemm kernels.
  This type of gemm kernels perform well on Nvidia and AMD GPUs"""

  def __init__(self, context: Context, scopes: Scopes, gemm_list):
    super(DefaultKernelBuilder, self).__init__(context, scopes, gemm_list)

  def get_selected_kernel_type(self):
    return KernelType.DEFAULT

  def _build_kernel(self):
    # allocate registers
    builder = RegistersAllocBuilder(self._context, self._scopes)
    builder.build(self._accumulator_size, 0.0)
    self._register_array_obj = builder.get_resultant_obj()
    self._instructions.extend(builder.get_instructions())

    # allocate shared memory
    builder = ShrMemAllocBuilder(self._context, self._scopes)
    builder.build(size=None)
    self._shr_mem_obj = builder.get_resultant_obj()
    self._instructions.extend(builder.get_instructions())

    self._scopes.add_scope()
    # generate GEMM and store operations
    builder = GemmBuilder(self._context,
                          self._scopes,
                          self._scopes.get_symbol(self._register_array_obj),
                          self._scopes.get_symbol(self._shr_mem_obj),
                          self._num_threads)

    for gemm_descr in self._gemm_list:
      builder.build(op1=self._scopes.get_symbol(gemm_descr.mat_a),
                    op2=self._scopes.get_symbol(gemm_descr.mat_b),
                    dest_obj=gemm_descr.mat_c,
                    descr=gemm_descr)
      self._instructions.extend(builder.get_instructions())

  def _deduce_num_threads(self):
    for gemm in self._gemm_list:
      num_threads, _ = gemm.get_num_threads(self._context)
      self._num_threads = max(num_threads, self._num_threads)

  def _deduce_accumulator_size(self):
    accumulator_num_columns = 0
    for gemm in self._gemm_list:
      local_acc_size = gemm.get_accumulator_size()
      accumulator_num_columns = max(accumulator_num_columns, local_acc_size)

    self._accumulator_size = [1, accumulator_num_columns]
