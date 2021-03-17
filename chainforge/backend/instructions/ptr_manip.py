from .abstract_instruction import AbstractInstruction
from chainforge.common.vm import VM
from chainforge.common.matrix import Matrix
from chainforge.common.basic_types import Addressing
from chainforge.common.aux import get_2d_block_id, get_extra_offset_name
from chainforge.common.basic_types import DataFlowDirection
from chainforge.backend.symbol import Symbol, SymbolType, DataView
from chainforge.backend.writer import Writer
from chainforge.backend.exceptions import InternalError, GenerationError


class GetElementPtr(AbstractInstruction):
  def __init__(self,
               vm: VM,
               src: Symbol,
               dest: Symbol):
    super(GetElementPtr, self).__init__(vm)

    if src.stype != SymbolType.Batch:
      raise InternalError('ptr: operand `src` is not in a batch')

    if not isinstance(src.obj, Matrix):
      raise InternalError(f'ptr: operand `src` is not a matrix')

    if dest.stype != SymbolType.Global:
      raise InternalError('ptr: operand `dest` is not in global mem.')

    if not isinstance(dest.obj, Matrix):
      raise InternalError('ptr: operand `dest` is not a matrix')

    dest.data_view = DataView(rows=src.obj.get_actual_num_rows(),
                              columns=src.obj.get_actual_num_cols(),
                              lead_dim=src.obj.num_rows,
                              is_transposed=False)

    self._dest = dest
    self._src = src
    self._is_ready = True

    src.add_user(self)
    dest.add_user(self)

  def gen_code(self, writer: Writer):
    extra_offset = get_extra_offset_name(self._src)
    batch_id = get_2d_block_id(self._vm)
    matrix = self._src.obj
    address = ''
    if matrix.addressing == Addressing.STRIDED:
      main_offset = f'({batch_id}) * {matrix.get_real_volume()}'
      sub_offset = f'{matrix.get_offset_to_first_element()}'
      address = f'{main_offset} + {sub_offset} + {extra_offset}'
    elif matrix.addressing == Addressing.PTR_BASED:
      main_offset = f'{batch_id}'
      sub_offset = f'{matrix.get_offset_to_first_element()}'
      address = f'{main_offset}][{sub_offset} + {extra_offset}'
    elif matrix.addressing == Addressing.NONE:
      address = f'{matrix.get_offset_to_first_element()}'
    else:
      GenerationError(f'unknown addressing of `src` operand, given {matrix.addressing}')

    rhs = f'&{self._src.name}[{address}]'

    lhs = 'const ' if matrix.direction == DataFlowDirection.SOURCE else ''
    lhs += f'{self._vm.fp_as_str()} * const __restrict__ {self._dest.name}'
    writer(f'{lhs} = {rhs};')

  def __str__(self) -> str:
    return f'{self._dest.name} = getelementptr_b2g {self._src.name};'
