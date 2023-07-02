from typing import Union
from chainforge.common import Context
from chainforge.common.matrix import Matrix
from chainforge.backend.data_types import RegMemObject
from chainforge.backend.symbol import Symbol, SymbolType, DataView
from chainforge.backend.exceptions import InternalError
from chainforge.backend.writer import Writer
from .abstract_instruction import AbstractInstruction, AbstractShrMemWrite
from copy import deepcopy


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

    bbox = dest.obj.get_bbox()
    bbox = [0, 0, bbox[2] - bbox[0], bbox[3] - bbox[1]]
    num_rows = context.align(bbox[2] - bbox[0])
    num_cols = dest.obj.get_actual_num_cols()
    dest.data_view = DataView(rows=num_rows,
                              columns=num_cols,
                              is_transposed=False,
                              bbox=bbox)

    self._dest: Symbol = dest
    self._src: Symbol = deepcopy(src)
    self._shr_mem: Symbol = shr_mem
    self._num_threads: int = num_threads
    self._shr_mem_offset: Union[int, None] = None
    view: DataView = self._dest.data_view
    self._shm_volume: int = view.get_volume()

  def gen_code(self, writer: Writer) -> None:
    writer.new_line()
    writer(f' // writing to shr mem: from {self._src.name} to {self._dest.name}')
    lhs = f'{self._fp_as_str}* {self._vm.lexic.restrict_kw} {self._dest.name}'
    rhs = f'&{self._shr_mem.name}[{self._shr_mem_offset}]'
    writer(f'{lhs} = {rhs};')

    dest_view = self._dest.data_view
    src_bbox = self._src.data_view.get_bbox()

    with writer.block(self.gen_range_mask_threads(begin=src_bbox[0], end=src_bbox[2])):
      writer.insert_pragma_unroll()
      loop = f'for (int i = 0; i < {dest_view.get_dim_size(1)}; ++i)'
      with writer.block(loop):
        dest_row_idx = f'{self._vm.lexic.thread_idx_x}'
        thread_id_displacement = self._src.data_view.get_offset()
        if thread_id_displacement:
          dest_row_idx += f' - {thread_id_displacement}'

        dest_addr = dest_view.get_address(row_idx=dest_row_idx, column_idx='i')
        lhs = f'{self._dest.name}[{dest_addr}]'

        rhs = f'{self._src.name}[i]'
        writer(f'{lhs} = {rhs};')

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

    if dest.data_view.get_dim_size(0) != src.data_view.get_dim_size(0):
      raise InternalError('store: `src` and `dest` do not match in size aling dim `0`')

    src.add_user(self)
    dest.add_user(self)

    dest.data_view = DataView(rows=dest.obj.num_rows,
                              columns=dest.obj.num_cols,
                              is_transposed=False,
                              bbox=dest.obj.get_bbox())

    self._dest: Symbol = dest
    self._src: Symbol = deepcopy(src)
    self._alpha = alpha
    self._beta = beta
    self._num_threads: int = num_threads
    self._is_ready: bool = True

  def gen_code(self, writer: Writer) -> None:
    writer.new_line()
    dest_view = self._dest.data_view

    writer('// write results back to glb. memory')
    src_bbox = self._src.data_view.get_bbox()
    with writer.block(self.gen_range_mask_threads(begin=src_bbox[0], end=src_bbox[2])):

      writer.insert_pragma_unroll()
      loop = f'for(int n = 0; n < {dest_view.get_dim_size(1)}; ++n)'
      with writer.block(loop):
        dest_row_idx = f'{self._vm.lexic.thread_idx_x}'
        thread_id_displacement = self._src.data_view.get_offset()
        if thread_id_displacement:
          dest_row_idx += f' - {thread_id_displacement}'

        dest_addr = dest_view.get_address(row_idx=dest_row_idx, column_idx='n')
        lhs = f'{self._dest.name}[{dest_addr}]'

        src_address = '' if self._src.obj.size == 1 else '[n]'
        rhs = f'{self._alpha} * {self._src.name}{src_address}'

        if self._beta != 0.0:
            rhs += f' + {self._beta} * {lhs}'

        writer(f'{lhs} = {rhs};')

  def __str__(self) -> str:
    return f'{self._dest.name} = store_r2g {self._src.name};'
