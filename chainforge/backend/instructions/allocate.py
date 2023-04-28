from typing import Union, List
from chainforge.common.basic_types import FloatingPointType
from chainforge.common import Context
from chainforge.backend.symbol import Symbol
from chainforge.backend.writer import Writer
from chainforge.backend.symbol import DataView
from .abstract_instruction import AbstractInstruction


class RegisterAlloc(AbstractInstruction):
  def __init__(self,
               context: Context,
               dest: Symbol,
               size: List[int],
               init_value: Union[float, None]=None):
    super(RegisterAlloc, self).__init__(context)
    self._init_value = init_value

    assert type(size) == list
    self._size = size
    self._dest = dest
    self._dest.data_view = DataView(rows=size[0],
                                    columns=size[1],
                                    lead_dim=size[0],
                                    is_transposed=False)
    self._is_ready = True
    dest.add_user(self)

  def gen_code(self, writer: Writer):
    num_rows = self._dest.data_view.rows
    num_columns = self._dest.data_view.columns

    values = ''
    if isinstance(self._init_value, float):
      fp_prefix = 'f' if self._context.fp_type == FloatingPointType.FLOAT else ''
      value = f'{self._init_value}{fp_prefix}'

      column_values = ', '.join([value] * num_columns)
      column_values = f'{{{column_values}}}'
      values = ', '.join([column_values] * num_rows)
      values = f' = {{{values}}}'

    dims = f'[{num_rows}][{num_columns}]'
    result = f'{self._context.fp_as_str()} {self._dest.obj.name}{dims}{values};'
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

    address = f'{shrmem_obj.get_size_per_mult()} * {self._vm.lexic.thread_idx_y}'
    writer(f'{self._fp_as_str} * {shrmem_obj.name} = &{common_shrmem}[{address}];')

  def is_ready(self):
    shrmem_obj = self._dest.obj
    if shrmem_obj.get_total_size():
      return True
    else:
      return False

  def __str__(self):
    return f'{self._dest.name} = alloc_shr [{self._dest.obj.get_total_size_as_str()}];'
