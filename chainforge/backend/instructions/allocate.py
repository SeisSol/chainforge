from .abstract_instruction import AbstractInstruction
from chainforge.common import Context
from chainforge.backend.symbol import Symbol
from chainforge.backend.exceptions import InternalError
from chainforge.backend.writer import Writer
from typing import Union


class RegisterAlloc(AbstractInstruction):
  def __init__(self,
               context: Context,
               dest: Symbol,
               size: int,
               init_value: Union[float, None] = None):
    super(RegisterAlloc, self).__init__(context)
    self._size = size
    self._init_value = init_value
    self._dest = dest
    self._is_ready = True
    dest.add_user(self)

  def gen_code(self, writer: Writer):
    if self._dest.obj.size < 1:
      raise InternalError('size of reg. obj must be at least 1')

    if self._dest.obj.size == 1:
      init_value = ''
      if isinstance(self._init_value, float):
        init_value = f' = {self._init_value}'
      result = f'{self._context.fp_as_str()} {self._dest.obj.name}{init_value};'
    else:
      init_values_list = ''
      if isinstance(self._init_value, float):
        init_values = ', '.join([str(self._init_value)] * self._dest.obj.size)
        init_values_list = f' = {{{init_values}}}'
      result = f'{self._context.fp_as_str()} {self._dest.obj.name}[{self._dest.obj.size}]{init_values_list};'
    writer(result)

  def __str__(self) -> str:
    return f'{self._dest.obj.name} = alloc_regs {self._dest.obj.size};'


class ShrMemAlloc(AbstractInstruction):
  def __init__(self,
               context: Context,
               dest: Symbol,
               size: Union[int, None]):
    super(ShrMemAlloc, self).__init__(context)
    self._size = size
    self._dest = dest

    dest.add_user(self)

  def gen_code(self, writer: Writer):
    shrmem_obj = self._dest.obj
    common_shrmem = f'total_{shrmem_obj.name}'
    common_shrmem_size = shrmem_obj.get_total_size()

    alignment = 8
    type_as_str = f'{self._vm.lexic.shr_mem_kw} __align__({alignment}) {self._fp_as_str}'
    writer(f'{type_as_str} {common_shrmem}[{common_shrmem_size}];')

    address = f'{shrmem_obj.get_size_per_mult()} * {self._vm.lexic.threadIdx_y}'
    writer(f'{self._fp_as_str} * {shrmem_obj.name} = &{common_shrmem}[{address}];')

  def is_ready(self):
    shrmem_obj = self._dest.obj
    if shrmem_obj.get_total_size():
      return True
    else:
      return False

  def __str__(self):
    return f'{self._dest.name} = alloc_shr [{self._dest.obj.get_total_size_as_str()}];'
