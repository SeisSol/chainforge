from .abstract_loader import AbstractShrMemLoader, ShrMemLoaderType
from chainforge.backend.writer import Writer
from chainforge.backend.symbol import DataView


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
    self._lid_dim = optimal_num_cols

    self._dest.data_view = DataView(rows=self._matrix.get_actual_num_cols(),
                                    columns=self._matrix.get_actual_num_rows(),
                                    lead_dim=self._lid_dim,
                                    is_transposed=True)

  def get_loader_type(self):
    return ShrMemLoaderType.TRANSPOSED

  def gen_code(self, writer: Writer):
    super(ExtendedTransposePatchLoader, self).gen_code(writer)
    writer(f'// loading {self._src.name} to {self._dest.name}: # trans, extended')

    matrix = self._src.obj
    num_hops = int(self._shm_volume / self._num_threads)
    tmp_var = 'index'
    with writer.block():
      if num_hops > 0:

        writer(f'int {tmp_var};')
        writer.new_line()

        # for-block: main part
        writer.insert_pragma_unroll()
        with writer.block(f'for (int i = 0; i < {num_hops}; ++i)'):
          writer(f'{tmp_var} = {self._vm.lexic.threadIdx_x} + i * {self._num_threads};')

          shr_mem_index = f'({tmp_var} % {matrix.num_rows}) * {self._lid_dim} + {tmp_var} / {matrix.num_rows}'
          glb_mem_index = f'{self._vm.lexic.threadIdx_x} + i * {self._num_threads}'
          writer(f'{self._dest.name}[{shr_mem_index}] = {self._src.name}[{glb_mem_index}];')

      # if-block: residual part
      if (self._shm_volume % self._num_threads) != 0:
        residual = self._shm_volume - num_hops * self._num_threads
        with writer.block(f'if ({self._vm.lexic.threadIdx_x} < {residual})'):
          writer(f'{tmp_var} = {self._vm.lexic.threadIdx_x} + {num_hops * self._num_threads};')

          shr_mem_index = f'({tmp_var} % {matrix.num_rows}) * {self._lid_dim} + {tmp_var} / {matrix.num_rows}'
          glb_mem_index = f'{self._vm.lexic.threadIdx_x} + {num_hops * self._num_threads}'
          writer(f'{self._dest.name}[{shr_mem_index}] = {self._src.name}[{glb_mem_index}];')

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
    self._lid_dim = optimal_num_cols

    self._dest.data_view = DataView(rows=self._matrix.get_actual_num_cols(),
                                    columns=self._matrix.get_actual_num_rows(),
                                    lead_dim=self._lid_dim,
                                    is_transposed=True)

  def get_loader_type(self):
    return ShrMemLoaderType.TRANSPOSED

  def gen_code(self, writer: Writer):
    super(ExactTransposePatchLoader, self).gen_code(writer)
    writer(f'// loading {self._src.name} to {self._dest.name}: # trans, exact')

    matrix = self._src.obj
    tmp_var = 'index'
    with writer.block(f'for (int i = 0; i < {matrix.get_actual_num_cols()}; ++i)'):
      num_hops = int(matrix.get_actual_num_rows() / self._num_threads)
      if num_hops > 0:

        # for-block: main part
        writer.insert_pragma_unroll()
        with writer.block(f'for (int counter = 0; counter < {num_hops}; ++counter)'):
          thread_idx = f'{self._vm.lexic.threadIdx_x} + counter * {self._num_threads}'
          writer(f'int {tmp_var} = {thread_idx} + i * {matrix.get_actual_num_rows()};')

          shr_mem_index = f'({tmp_var} % {matrix.get_actual_num_rows()}) * {self._lid_dim} + ' \
            f'{tmp_var} / {matrix.get_actual_num_rows()}'

          glb_mem_index = f'{thread_idx} + i * {matrix.num_rows}'
          writer(f'{self._dest.name}[{shr_mem_index}] = {self._src.name}[{glb_mem_index}];')

      # if-block: residual part
      if (matrix.get_actual_num_rows() % self._num_threads) != 0:
        residual = matrix.get_actual_num_rows() - num_hops * self._num_threads

        with writer.block(f'if ({self._vm.lexic.threadIdx_x} < {residual})'):
          finial_offset = num_hops * self._num_threads
          thread_idx = f'{self._vm.lexic.threadIdx_x} + {finial_offset}'
          writer(f'int {tmp_var} = {thread_idx} + i * {matrix.get_actual_num_rows()};')

          shr_mem_index = f'({tmp_var} % {matrix.get_actual_num_rows()}) * {self._lid_dim} + ' \
            f'{tmp_var} / {matrix.get_actual_num_rows()}'
          glb_mem_index = f'{thread_idx} + i * {matrix.num_rows}'
          writer(f'{self._dest.name}[{shr_mem_index}] = {self._src.name}[{glb_mem_index}];')

  def __str__(self):
    return f'{self._dest.name} = load_g2s_trans {self._shr_mem.name}, {self._src.name};'
