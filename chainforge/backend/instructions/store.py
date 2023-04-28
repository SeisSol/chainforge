from typing import Union
from chainforge.common import Context
from chainforge.common.matrix import Matrix
from chainforge.backend.data_types import RegMemObject
from chainforge.backend.symbol import Symbol, SymbolType, DataView
from chainforge.backend.exceptions import InternalError
from chainforge.backend.writer import Writer
from .abstract_instruction import AbstractInstruction, AbstractShrMemWrite


class StoreRegToShr(AbstractShrMemWrite):
  def __init__(self,
               context: Context,
               src: Symbol,
               dest: Symbol,
               shr_mem: Symbol,
               num_threads: int):
    super(StoreRegToShr, self).__init__(context)

    if src.stype != SymbolType.Register:
      raise InternalError('store: operand `src` is not in registers')

    if not isinstance(src.obj, RegMemObject):
      raise InternalError(f'store: operand `src` is not registers, instead: {type(src.obj)}')

    if dest.stype != SymbolType.SharedMem:
      raise InternalError('store: operand `dest` is not in shared mem.')

    if not isinstance(dest.obj, Matrix):
      raise InternalError(f'store: operand `dest` is not a matrix, instead: {type(src.obj)}')

    src.add_user(self)
    dest.add_user(self)
    shr_mem.add_user(self)

    dest.data_view = DataView(rows=dest.obj.get_actual_num_rows(),
                              columns=dest.obj.get_actual_num_cols(),
                              lead_dim=dest.obj.num_rows,
                              is_transposed=False)

    self._dest: Symbol = dest
    self._src: Symbol = src
    self._shr_mem: Symbol = shr_mem
    self._num_threads: int = num_threads
    self._shr_mem_offset: Union[int, None] = None
    view: DataView = self._dest.data_view
    self._shm_volume: int = view.rows * view.columns

  def _gen_body(self, writer: Writer, thread_var, reg_var):
    view = self._dest.data_view
    writer.insert_pragma_unroll()
    loop = f'for (int j = 0; j < {view.columns}; ++j)'
    with writer.block(loop):
      lhs = f'{self._dest.name}[{thread_var} + {view.lead_dim} * j]'
      rhs = f'{self._src.name}[{reg_var}][j]'
      writer(f'{lhs} = {rhs};')

  def gen_code(self, writer: Writer) -> None:
    writer.new_line()
    writer(f' // writing from reg. to shr. mem: from {self._src.name} to {self._dest.name}')

    lhs = f'{self._fp_as_str}* {self._vm.lexic.restrict_kw} {self._dest.name}'
    rhs = f'&{self._shr_mem.name}[{self._shr_mem_offset}]'
    writer(f'{lhs} = {rhs};')

    lexic = self._vm.lexic
    loop_header = f'int t = {lexic.thread_idx_x}, c = 0; '
    loop_header += f't < {self._dest.data_view.rows}; '
    loop_header += f't += {lexic.block_dim_x}, ++c'
    with writer.block(f'for({loop_header})'):
      self._gen_body(writer=writer,
                     thread_var='t',
                     reg_var='c')

  def get_dest(self) -> Symbol:
    return self._dest

  def __str__(self) -> str:
    return f'{self._dest.name} = store_r2s {self._shr_mem.name}, {self._src.name};'


class StoreRegToGlb(AbstractInstruction):
  def __init__(self,
               context: Context,
               src: Symbol,
               dest: Symbol,
               alpha: float,
               beta: float,
               num_threads: int):
    super(StoreRegToGlb, self).__init__(context)

    if src.stype != SymbolType.Register:
      raise InternalError('store: operand `src` is not in reg mem')

    if not isinstance(src.obj, RegMemObject):
      raise InternalError(f'store: operand `src` is registers, instead: {type(src.obj)}')

    if dest.stype != SymbolType.Global:
      raise InternalError('store: operand `dest` is not in global memory.')

    if not isinstance(dest.obj, Matrix):
      raise InternalError('store: operand `dest` is not a matrix')

    src.add_user(self)
    dest.add_user(self)

    dest.data_view = DataView(rows=dest.obj.get_actual_num_rows(),
                              columns=dest.obj.get_actual_num_cols(),
                              lead_dim=dest.obj.num_rows,
                              is_transposed=False)

    self._dest: Symbol = dest
    self._src: Symbol = src
    self._alpha = alpha
    self._beta = beta
    self._num_threads: int = num_threads
    self._is_ready: bool = True

  def _gen_body(self, writer: Writer, thread_var, reg_var):
    dest_view = self._dest.data_view
    src_view = self._src.data_view

    writer.insert_pragma_unroll()
    loop = f'for(int n = 0; n < {dest_view.columns}; ++n)'
    with writer.block(loop):
      lhs = f'{self._dest.name}[{thread_var} + {dest_view.lead_dim} * n]'

      src_address = f'[{reg_var}][n]'
      rhs = f'{self._alpha} * {self._src.name}{src_address}'

      if self._beta != 0.0:
        rhs += f' + {self._beta} * {lhs}'

      writer(f'{lhs} = {rhs};')

  def gen_code(self, writer: Writer) -> None:
    writer.new_line()
    writer(f' // writing from reg. to gdb. mem: from {self._src.name} to {self._dest.name}')

    lexic = self._vm.lexic
    loop_header = f'int t = {lexic.thread_idx_x}, c = 0; '
    loop_header += f't < {self._dest.data_view.rows}; '
    loop_header += f't += {lexic.block_dim_x}, ++c'
    with writer.block(f'for({loop_header})'):
      self._gen_body(writer=writer,
                     thread_var='t',
                     reg_var='c')

  def __str__(self) -> str:
    return f'{self._dest.name} = store_r2g {self._src.name};'
