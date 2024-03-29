from chainforge.common import Context
from chainforge.common.matrix import Matrix
from chainforge.common.basic_types import Addressing
from chainforge.common.aux import get_2d_block_id, get_extra_offset_name
from chainforge.common.basic_types import DataFlowDirection
from chainforge.backend.symbol import Symbol, SymbolType, DataView
from chainforge.backend.writer import Writer
from chainforge.backend.exceptions import InternalError, GenerationError
from .abstract_instruction import AbstractInstruction


class GetElementPtr(AbstractInstruction):
  def __init__(self,
               context: Context,
               src: Symbol,
               dest: Symbol):
    super(GetElementPtr, self).__init__(context)

    if src.stype != SymbolType.Batch:
      raise InternalError('ptr: operand `src` is not in a batch')

    if not isinstance(src.obj, Matrix):
      raise InternalError(f'ptr: operand `src` is not a matrix')

    if dest.stype != SymbolType.Global:
      raise InternalError('ptr: operand `dest` is not in global mem.')

    if not isinstance(dest.obj, Matrix):
      raise InternalError('ptr: operand `dest` is not a matrix')

    dest.data_view = DataView(rows=src.obj.num_rows,
                              columns=src.obj.num_cols,
                              is_transposed=False,
                              bbox=src.obj.get_bbox())

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
      offset = f'({batch_id}) * {matrix.get_real_volume()}'
      address = f'{offset} + {extra_offset}'
    elif matrix.addressing == Addressing.PTR_BASED:
      address = f'{batch_id}][{extra_offset}'
    elif matrix.addressing == Addressing.NONE:
      address = f'0'
    else:
      GenerationError(f'unknown addressing of `src` operand, given {matrix.addressing}')

    rhs = f'&{self._src.name}[{address}]'

    lhs = 'const ' if matrix.direction == DataFlowDirection.SOURCE else ''
    lhs += f'{self._fp_as_str} * const {self._vm.lexic.restrict_kw} {self._dest.name}'
    writer(f'{lhs} = {rhs};')

  def __str__(self) -> str:
    return f'{self._dest.name} = getelementptr_b2g {self._src.name};'
