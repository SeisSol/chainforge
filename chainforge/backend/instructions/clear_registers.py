from chainforge.common import Context
from chainforge.common.basic_types import FloatingPointType
from chainforge.backend.symbol import Symbol, SymbolType
from chainforge.backend.writer import Writer
from chainforge.backend.exceptions import InternalError
from .abstract_instruction import AbstractInstruction


class ClearRegisters(AbstractInstruction):
  def __init__(self,
               context: Context,
               src: Symbol):
    super(ClearRegisters, self).__init__(context)

    if src.stype != SymbolType.Register:
      raise InternalError('ptr: operand `src` is not in registers')

    self._is_ready = True
    self._src = src
    src.add_user(self)

  def gen_code(self, writer: Writer):
    writer.new_line()
    writer(f'// clear registers')

    writer.insert_pragma_unroll()
    with writer.block(f'for (int i = 0; i < {self._src.data_view.rows}; ++i)'):
      writer.insert_pragma_unroll()
      with writer.block(f'for (int j = 0; j < {self._src.data_view.columns}; ++j)'):
        fp_prefix = 'f' if self._context.fp_type == FloatingPointType.FLOAT else ''
        writer(f'{self._src.name}[i][j] = 0.0{fp_prefix};')

  def __str__(self) -> str:
    return f'clear_regs {self._src.name}[{self._src.obj.size}];'
