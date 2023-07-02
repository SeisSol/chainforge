from chainforge.backend.writer import Writer
from chainforge.backend.symbol import DataView
from .abstract_loader import AbstractShrMemLoader, ShrMemLoaderType


def _find_next_prime(number):
  factor = 2
  return _find_prime_in_range(number, factor * number)


def _find_prime_in_range(source, target):
  for number in range(source, target):
    for i in range(2, number):
      if number % i == 0:
        break
    else:
      return number


class ExtendedTransposePatchLoader(AbstractShrMemLoader):
  """A strategy which loads an entire matrix into shared memory and transposes it on the fly
  """

  def __init__(self, **kwargs):
    super(ExtendedTransposePatchLoader, self).__init__(**kwargs)

    optimal_num_cols = _find_next_prime(self._matrix.get_actual_num_cols())
    self._shm_volume = optimal_num_cols * self._matrix.num_rows

    src_bbox = self._matrix.get_bbox()
    self._src.data_view = DataView(rows=self._matrix.num_rows,
                                   columns=self._matrix.num_cols,
                                   is_transposed=False,
                                   bbox=src_bbox)

    dest_bbox = [0, 0, src_bbox[3] - src_bbox[1], src_bbox[2] - src_bbox[0]]
    self._dest.data_view = DataView(rows=optimal_num_cols,
                                    columns=self._matrix.get_actual_num_rows(),
                                    is_transposed=True,
                                    bbox=dest_bbox)

  def get_loader_type(self):
    return ShrMemLoaderType.TRANSPOSED

  def gen_code(self, writer: Writer):
    super(ExtendedTransposePatchLoader, self).gen_code(writer)
    writer(f'// loading {self._src.name} to {self._dest.name}: # trans, extended')

    num_hops = int(self._shm_volume / self._num_threads)
    tmp_var = 'index'

    src_lead_dim = self._src.data_view.get_lead_dim()
    dest_lead_dim = self._dest.data_view.get_lead_dim()

    src_offset = self._src.data_view.get_offset()
    src_offset = f'{src_offset} + ' if src_offset else ''

    with writer.block():
      writer(f'int {tmp_var};')
      writer.new_line()
      if num_hops > 0:
        # for-block: main part
        writer.insert_pragma_unroll()
        with writer.block(f'for (int i = 0; i < {num_hops}; ++i)'):
          writer(f'{tmp_var} = {self._vm.lexic.thread_idx_x} + i * {self._num_threads};')

          shr_mem_index = f'({tmp_var} % {src_lead_dim}) * {dest_lead_dim}'
          shr_mem_index += f' + {tmp_var} / {src_lead_dim}'
          lhs = f'{self._dest.name}[{shr_mem_index}]'

          glb_mem_index = f'{self._vm.lexic.thread_idx_x} + i * {self._num_threads}'
          rhs = f'{self._src.name}[{src_offset}{glb_mem_index}]'
          writer(f'{lhs} = {rhs};')

      # if-block: residual part
      if (self._shm_volume % self._num_threads) != 0:
        residual = self._shm_volume - num_hops * self._num_threads
        with writer.block(f'if ({self._vm.lexic.thread_idx_x} < {residual})'):
          writer(f'{tmp_var} = {self._vm.lexic.thread_idx_x} + {num_hops * self._num_threads};')

          shr_mem_index = f'({tmp_var} % {src_lead_dim}) * {dest_lead_dim}'
          shr_mem_index += f' + {tmp_var} / {src_lead_dim}'
          lhs = f'{self._dest.name}[{shr_mem_index}]'

          glb_mem_index = f'{self._vm.lexic.thread_idx_x} + {num_hops * self._num_threads}'
          rhs = f'{self._src.name}[{src_offset}{glb_mem_index}]'
          writer(f'{lhs} = {rhs};')

  def __str__(self):
    return f'{self._dest.name} = load_g2s_trans_ext {self._shr_mem.name}, {self._src.name};'


class ExactTransposePatchLoader(AbstractShrMemLoader):
  """A strategy which loads only a necessary part of a matrix into shared memory
  and transposes it on the fly
  """

  def __init__(self, **kwargs):
    super(ExactTransposePatchLoader, self).__init__(**kwargs)
    optimal_num_cols = _find_next_prime(self._matrix.get_actual_num_cols())
    self._shm_volume = optimal_num_cols * self._matrix.num_rows

    src_bbox = self._matrix.get_bbox()
    self._src.data_view = DataView(rows=self._matrix.num_rows,
                                   columns=self._matrix.num_cols,
                                   is_transposed=False,
                                   bbox=src_bbox)

    dest_bbox = [0, 0, src_bbox[3] - src_bbox[1], src_bbox[2] - src_bbox[0]]
    self._dest.data_view = DataView(rows=optimal_num_cols,
                                    columns=self._matrix.get_actual_num_rows(),
                                    is_transposed=True,
                                    bbox=dest_bbox)


  def get_loader_type(self):
    return ShrMemLoaderType.TRANSPOSED

  def gen_code(self, writer: Writer):
    super(ExactTransposePatchLoader, self).gen_code(writer)
    writer(f'// loading {self._src.name} to {self._dest.name}: # trans, exact')

    tmp_var = 'index'
    src_view = self._src.data_view
    dest_view = self._dest.data_view

    src_offset = self._src.data_view.get_offset()
    src_offset = f'{src_offset} + ' if src_offset else ''

    with writer.block(f'for (int i = 0; i < {src_view.get_dim_size(1)}; ++i)'):
      num_hops = int(src_view.get_dim_size(0) / self._num_threads)
      if num_hops > 0:

        # for-block: main part
        writer.insert_pragma_unroll()
        with writer.block(f'for (int counter = 0; counter < {num_hops}; ++counter)'):
          thread_idx = f'{self._vm.lexic.thread_idx_x} + counter * {self._num_threads}'
          writer(f'int {tmp_var} = {thread_idx} + i * {src_view.get_dim_size(0)};')

          shr_mem_index = f'({tmp_var} % {src_view.get_dim_size(0)}) * {dest_view.get_lead_dim()} + '
          shr_mem_index += f'{tmp_var} / {src_view.get_dim_size(0)}'
          lhs = f'{self._dest.name}[{shr_mem_index}]'

          glb_mem_index = f'{thread_idx} + i * {src_view.get_lead_dim()}'
          rhs = f'{self._src.name}[{src_offset}{glb_mem_index}]'
          writer(f'{lhs} = {rhs};')

      # if-block: residual part
      if (src_view.get_dim_size(0) % self._num_threads) != 0:
        residual = src_view.get_dim_size(0) - num_hops * self._num_threads

        with writer.block(f'if ({self._vm.lexic.thread_idx_x} < {residual})'):
          finial_offset = num_hops * self._num_threads
          thread_idx = f'{self._vm.lexic.thread_idx_x} + {finial_offset}'
          writer(f'int {tmp_var} = {thread_idx} + i * {src_view.get_dim_size(0)};')

          shr_mem_index = f'({tmp_var} % {src_view.get_dim_size(0)}) * {dest_view.get_lead_dim()} + '
          shr_mem_index += f'{tmp_var} / {src_view.get_dim_size(0)}'
          lhs = f'{self._dest.name}[{shr_mem_index}]'

          glb_mem_index = f'{thread_idx} + i * {src_view.get_lead_dim()}'
          rhs = f'{self._src.name}[{src_offset}{glb_mem_index}]'
          writer(f'{lhs} = {rhs};')

  def __str__(self):
    return f'{self._dest.name} = load_g2s_trans {self._shr_mem.name}, {self._src.name};'
