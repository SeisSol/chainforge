from chainforge.common import Context
from chainforge.common.matrix import Matrix
from chainforge.backend.symbol import Symbol, SymbolType
from chainforge.backend.exceptions import InternalError, GenerationError
from chainforge.backend.writer import Writer
from .abstract_instruction import AbstractInstruction


class Gemm(AbstractInstruction):
  def __init__(self,
               context: Context,
               trans_a: bool,
               trans_b: bool,
               op1: Symbol,
               op2: Symbol,
               dest: Symbol,
               num_threads: int):
    super(Gemm, self).__init__(context)
    self._trans_a = trans_a
    self._trans_b = trans_b
    self._op1 = op1
    self._op2 = op2
    self._is_ready = True
    self._user_options = context.get_user_options()

    self.registers = None
    if dest.stype != SymbolType.Register:
      raise InternalError(f'gemm: accumulator-register array is not provided. Instead: {dest.stype}')
    else:
      self._dest = dest

    self._num_threads = num_threads

    if not isinstance(self._op1.obj, Matrix):
       raise InternalError('gemm: op1 is not a matrix')

    if not isinstance(self._op2.obj, Matrix):
      raise InternalError('gemm: op2 is not a matrix')

    op1.add_user(self)
    op2.add_user(self)
    dest.add_user(self)

  def gen_code(self, writer: Writer):
    self._check()
    writer.new_line()
    writer(f'// gemm: {self._op1.name} x {self._op2.name}')

    op1_view = self._op1.data_view
    num_full_cycles = int(op1_view.rows / self._num_threads)
    tails_mask = op1_view.rows % self._num_threads

    gemm_variant = self.gen_code_without_prefetch
    if self._user_options.prefetch_gemm:
      if self._op1.stype == SymbolType.Global:
        gemm_variant = self.gen_code_with_prefetch

    lexic = self._vm.lexic
    loop_header = f'int t = {lexic.thread_idx_x}, c = 0; '
    loop_header += f't < {self._op1.data_view.rows}; '
    loop_header += f't += {lexic.block_dim_x}, ++c'
    with writer.block(f'for({loop_header})'):
      gemm_variant(writer,
                   self._op1.data_view,
                   self._op2.data_view,
                   thread_var='t',
                   reg_var='c')

  def _gen_inner_loop(self,
                      writer,
                      op1_element,
                      is_requested_layout,
                      n_range,
                      k,
                      lead_dim,
                      reg_var):
    writer.insert_pragma_unroll()
    with writer.block(f'for (int n = 0; n < {n_range}; ++n)'):
      if is_requested_layout:
        address = f'{k} + {lead_dim} * n'
      else:
        address = f'n + {lead_dim} * {k}'

      dest_address = f'[{reg_var}][n]'
      writer(f'{self._dest.name}{dest_address} += {op1_element} * {self._op2.name}[{address}];')

  def gen_code_without_prefetch(self,
                                writer,
                                view_op1,
                                view_op2,
                                thread_var,
                                reg_var):
    k_range = view_op1.columns
    writer.insert_pragma_unroll()
    with writer.block(f'for (int k = 0; k < {k_range}; ++k)'):
      address = f'{thread_var} + k * {view_op1.lead_dim}'
      writer(f'{self._fp_as_str} value = {self._op1.name}[{address}];')

      is_requested_layout = view_op2.is_transposed == self._trans_b
      n_range = view_op2.columns if is_requested_layout else view_op2.rows

      writer.new_line()
      self._gen_inner_loop(writer,
                           op1_element='value',
                           is_requested_layout=is_requested_layout,
                           n_range=n_range,
                           k=f'k',
                           lead_dim=view_op2.lead_dim,
                           reg_var=reg_var)

  def gen_code_with_prefetch(self,
                             writer,
                             view_op1,
                             view_op2,
                             iteration_variable):
    address = f'{self._vm.lexic.thread_idx_x}'
    writer(f'{self._fp_as_str} prefetch = {self._op1.name}[{address}];')

    if self._user_options.exact_contraction_length:
      k_range = view_op1.columns
    else:
      k_range = min(view_op1.columns, view_op2.rows)

    with writer.block(f'for (int k = 0; k < {k_range - 1}; ++k)'):

      writer(f'{self._fp_as_str} value = prefetch;')
      address = f'{self._vm.lexic.thread_idx_x} + (k + 1) * {view_op1.lead_dim}'
      writer(f'prefetch = {self._op1.name}[{address}];')

      is_requested_layout = view_op2.is_transposed == self._trans_b
      n_range = view_op2.columns if is_requested_layout else view_op2.rows

      writer.new_line()
      self._gen_inner_loop(writer,
                           op1_element='value',
                           is_requested_layout=is_requested_layout,
                           n_range=n_range,
                           k=f'k',
                           lead_dim=view_op2.lead_dim,
                           iteration_variable=iteration_variable)
    with writer.block():
      writer('// gemm tail i.e. last iteration')
      self._gen_inner_loop(writer,
                           op1_element='prefetch',
                           is_requested_layout=is_requested_layout,
                           n_range=n_range,
                           k=f'{k_range - 1}',
                           lead_dim=view_op2.lead_dim,
                           iteration_variable=iteration_variable)

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
    k_range_op1 = view_op1.columns

    # Note: we do not reload op2 to the shared memory if the current gemm op. requires
    # a different layout in contrast to the one that has already been loaded to the shared memory
    k_range_op2 = view_op2.rows if is_requested_layout else view_op2.columns

    if self._user_options.exact_contraction_length:
      if k_range_op1 != k_range_op2:
        print(view_op1)
        print(view_op2)
        raise GenerationError(f'gemm: mismatch of contraction length '
                              f'k_range_op1( {k_range_op1} ) != k_range_op2( {k_range_op2} )')


    if view_op2.columns > self._dest.data_view.columns:
      msg = f'{view_op2.columns} > {self._dest.data_view.columns}'
      raise InternalError(f'gemm: contraction length is bigger than reg. size i.e, {msg}')

  def get_op1(self):
    return self._op1

  def get_op2(self):
    return self._op2

  def __str__(self):
    return f'{self._dest.name} = gemm {self._op1.name}, {self._op2.name};'
