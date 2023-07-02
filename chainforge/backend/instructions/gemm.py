from chainforge.common import Context
from chainforge.common.matrix import Matrix
from chainforge.backend.symbol import Symbol, SymbolType, DataView
from chainforge.backend.exceptions import InternalError, GenerationError
from chainforge.backend.writer import Writer
from .abstract_instruction import AbstractInstruction
from copy import deepcopy


class Gemm(AbstractInstruction):
  def __init__(self,
               context: Context,
               trans_a: bool,
               trans_b: bool,
               op1: Symbol,
               op2: Symbol,
               dest: Symbol,
               prefer_align: bool):
    super(Gemm, self).__init__(context)
    self._trans_a = trans_a
    self._trans_b = trans_b
    self._op1 = op1
    self._op2 = op2
    self._prefer_align = prefer_align
    self._is_ready = True
    self._user_options = context.get_user_options()
    self._gemm_meta_data = None

    self.registers = None
    if dest.stype != SymbolType.Register:
      raise InternalError(f'gemm: accumulator-register array is not provided. Instead: {dest.stype}')
    else:
      self._dest = dest

    if not isinstance(self._op1.obj, Matrix):
       raise InternalError('gemm: op1 is not a matrix')

    if not isinstance(self._op2.obj, Matrix):
      raise InternalError('gemm: op2 is not a matrix')

    op1.add_user(self)
    op2.add_user(self)
    dest.add_user(self)

    self._analyze()

  def _analyze(self):
    self._op1_view = self._op1.data_view
    self._op2_view = self._op2.data_view

    self._is_layout_as_requested = self._op2_view.is_transposed == self._trans_b
    if self._is_layout_as_requested:
      self._n_range = self._op2_view.get_dim_size(1)
    else:
      self._n_range = self._op2_view.get_dim_size(0)

    self._m_range = self._op1_view.get_dim_size(0)
    num_dirty_rows = 0

    if self._prefer_align:
      op1_bbox = self._op1_view.get_bbox()
      self._op1_view = deepcopy(self._op1.data_view)
      aligned_begin, aligned_end = self._context.align_range(begin=op1_bbox[0], end=op1_bbox[2])

      if aligned_end > self._op1_view.get_lead_dim():
        aligned_end = self._op1_view.get_lead_dim()

      aligned_bbox = [aligned_begin, op1_bbox[1], aligned_end, op1_bbox[3]]

      self._op1_view.reset_bbox(aligned_bbox)
      num_dirty_rows = op1_bbox[0] - aligned_begin

      if not (aligned_begin == op1_bbox[0] and aligned_end == op1_bbox[2]):
        text = f'gemm aligned along `m` dim: '
        text += f'from [{op1_bbox[0]}, {op1_bbox[2]}] '
        text += f'to [{aligned_begin}, {aligned_end}]; '
        text += f'num. dirty rows in `result`: {num_dirty_rows}'
        self._gemm_meta_data = text

    dest_bbox = [0, 0, self._m_range, self._n_range]
    dest_bbox[0] += num_dirty_rows
    dest_bbox[2] += num_dirty_rows
    self._dest.data_view = DataView(rows=self._op1_view.get_dim_size(0),
                                    columns=self._n_range,
                                    is_transposed=False,
                                    bbox=dest_bbox)


  def gen_code(self, writer: Writer):
    self._check()
    writer.new_line()
    writer(f'// gemm: {self._op1.name} x {self._op2.name}')
    if self._gemm_meta_data:
      writer(f'// meta: {self._gemm_meta_data}')

    with writer.block(self.gen_mask_threads(self._op1_view.get_dim_size(0))):
      k_range = self._op1_view.get_dim_size(1)
      writer.insert_pragma_unroll()
      with writer.block(f'for (int k = 0; k < {k_range}; ++k)'):
        address = self._op1_view.get_address(row_idx=self._vm.lexic.thread_idx_x, column_idx='k')
        writer(f'{self._fp_as_str} value = {self._op1.name}[{address}];')

        writer.new_line()
        self._gen_inner_loop(writer, op1_element='value', k='k')

  def _gen_inner_loop(self, writer, op1_element, k):
    writer.insert_pragma_unroll()
    with writer.block(f'for (int n = 0; n < {self._n_range}; ++n)'):
      if self._is_layout_as_requested:
        address = self._op2.data_view.get_address(row_idx='k', column_idx='n')
      else:
        address = self._op2.data_view.get_address(row_idx='n', column_idx='k')

      dest_address = '' if self._dest.obj.size == 1 else '[n]'
      writer(f'{self._dest.name}{dest_address} += {op1_element} * {self._op2.name}[{address}];')

  def _check(self):
    view_op1 = self._op1.data_view
    view_op2 = self._op2.data_view
    if not view_op1:
      raise InternalError(f'symbol data view has not been assign to `op1`')

    if not view_op1.is_transposed == self._trans_a:
      raise GenerationError(f'`op1 layout does not match the layout request by gemm instr.`')

    if not view_op2:
      raise InternalError(f'gemm: symbol data view has not been assign to `op2`')

    is_requested_layout = view_op2.is_transposed == self._trans_b

    # layout op1 is transposed if necessary and layout has already been adjusted
    # Note: if a subsequent GEMM requires to change the current layout
    # the matrix is going to be reloaded to the shared memory
    k_range_op1 = view_op1.get_dim_size(1)

    # Note: we do not reload op2 to the shared memory if the current gemm op. requires
    # a different layout in contrast to the one that has already been loaded to the shared memory
    k_range_op2 = view_op2.get_dim_size(0) if is_requested_layout else view_op2.get_dim_size(1)

    if self._user_options.exact_contraction_length:
      if k_range_op1 != k_range_op2:
        print(view_op1)
        print(view_op2)
        raise GenerationError(f'gemm: mismatch of contraction length '
                              f'k_range_op1( {k_range_op1} ) != k_range_op2( {k_range_op2} )')

    op2_columns = view_op2.get_dim_size(1)
    if op2_columns > self._dest.obj.size:
      msg = f'{op2_columns} > {self._dest.obj.size}'
      raise InternalError(f'gemm: contraction length is bigger than reg. size i.e, {msg}')

  def get_op1(self):
    return self._op1

  def get_op2(self):
    return self._op2

  def __str__(self):
    return f'{self._dest.name} = gemm {self._op1.name}, {self._op2.name};'
