from chainforge.common import Context, VM
from chainforge.backend.scopes import Scopes
from chainforge.backend.instructions import GetElementPtrBuilder
from chainforge.backend.instructions.builders.abstract_builder import AbstractBuilder
from abc import abstractmethod


class BaseKernelBuilder(AbstractBuilder):
  """ This is the base class for building complete gemm kernels."""

  def __init__(self, context: Context, scopes: Scopes, gemm_list):
    super(BaseKernelBuilder, self).__init__(context, scopes)
    self._gemm_list = gemm_list

    self._num_threads = 0
    self._accumulator_size = 0

    self._register_array_obj = None
    self._shr_mem_obj = None

  @abstractmethod
  def _deduce_num_threads(self):
    pass

  @abstractmethod
  def _deduce_accumulator_size(self):
    pass

  def get_num_threads(self):
    return self._num_threads

  def get_accumulator_size(self):
    return self._accumulator_size

  def get_reg_array_obj(self):
    return self._register_array_obj

  def get_shr_mem_obj(self):
    return self._shr_mem_obj

  def _build_prologue(self):
    self._deduce_num_threads()
    self._deduce_accumulator_size()

    # find local data from batches
    builder = GetElementPtrBuilder(self._context, self._scopes)
    self._scopes.add_scope()
    for symbol in self._scopes.get_global_scope().values():
      builder.build(symbol)
      self._instructions.extend(builder.get_instructions())

  @abstractmethod
  def _build_kernel(self):
    pass

  def build(self):
    self._build_prologue()
    self._build_kernel()
