from .abstract_loader import AbstractShrMemLoader, ShrMemLoaderType
from chainforge.backend.writer import Writer
from chainforge.backend.symbol import DataView


class ExtendedPatchLoader(AbstractShrMemLoader):
  """A strategy which loads an entire matrix into shared memory.
  """

  def __init__(self, **kwargs):
    super(ExtendedPatchLoader, self).__init__(**kwargs)

    full_subvolume = (self._matrix.get_actual_num_cols() - 2) * self._matrix.num_rows
    cropped_subvolume = self._matrix.get_actual_num_rows() + self._matrix.num_rows
    self._shm_volume = cropped_subvolume + full_subvolume
    self._lid_dim = self._matrix.num_rows

    self._dest.data_view = DataView(rows=self._matrix.get_actual_num_rows(),
                                    columns=self._matrix.get_actual_num_cols(),
                                    lead_dim=self._lid_dim,
                                    is_transposed=False)

  def get_loader_type(self):
    return ShrMemLoaderType.NOT_TRANSPOSED

  def gen_code(self, writer: Writer):
    super(ExtendedPatchLoader, self).gen_code(writer)
    writer(f'// loading {self._src.name} to {self._dest.name}: # no trans, extended')

    num_hops = int(self._shm_volume / self._num_threads)
    if num_hops > 0:
      writer.insert_pragma_unroll()
      with writer.block(f'for (int i = 0; i < {num_hops}; ++i)'):
        index = f'{self._vm.lexic.threadIdx_x} + i * {self._num_threads}'
        lhs = f'{self._dest.name}[{index}]'
        rhs = f'{self._src.name}[{index}]'
        writer(f'{lhs} = {rhs};')

    # the last hop to fill shared mem with data
    if (self._shm_volume % self._num_threads) != 0:
      residue = self._shm_volume - num_hops * self._num_threads
      with writer.block(f'if ({self._vm.lexic.threadIdx_x} < {residue})'):
        index = f'{self._vm.lexic.threadIdx_x} + {num_hops * self._num_threads}'
        writer(f'{self._dest.name}[{index}] = {self._src.name}[{index}];')

  def __str__(self):
    return f'{self._dest.name} = load_g2s_ext {self._shr_mem.name}, {self._src.name};'


class ExactPatchLoader(AbstractShrMemLoader):
  """A strategy which loads only a necessary part of a matrix into shared memory.
  """

  def __init__(self, **kwargs):
    super(ExactPatchLoader, self).__init__(**kwargs)
    self._lid_dim = self._matrix.get_actual_num_rows()
    self._shm_volume = self._matrix.get_actual_volume()

    self._dest.data_view = DataView(rows=self._matrix.get_actual_num_rows(),
                                    columns=self._matrix.get_actual_num_cols(),
                                    lead_dim=self._lid_dim,
                                    is_transposed=False)

  def get_loader_type(self):
    return ShrMemLoaderType.NOT_TRANSPOSED

  def gen_code(self, writer: Writer):
    super(ExactPatchLoader, self).gen_code(writer)
    writer(f'// loading {self._src.name} to {self._dest.name}: # no trans, exact.')

    matrix = self._src.obj
    with writer.block(f'for (int i = 0; i < {matrix.get_actual_num_cols()}; ++i)'):

      num_hops = int(self._lid_dim / self._num_threads)
      if num_hops > 0:

        writer.insert_pragma_unroll()
        with writer.block(f'for (int counter = 0; counter < {num_hops}; ++counter)'):
          shr_mem_index = f'{self._vm.lexic.threadIdx_x} + ' \
            f'counter * {self._num_threads} + i * {self._lid_dim}'
          lhs = f'{self._dest.name}[{shr_mem_index}]'

          glob_mem_index = f'{self._vm.lexic.threadIdx_x} + ' \
            f'counter * {self._num_threads} + i * {matrix.num_rows}'
          rhs = f'{self._src.name}[{glob_mem_index}]'

          writer(f'{lhs} = {rhs};')

      # the last hop to fill shared mem with data
      if (self._lid_dim % self._num_threads) != 0:
        residue = self._lid_dim - num_hops * self._num_threads
        with writer.block(f'if ({self._vm.lexic.threadIdx_x} < {residue})'):
          finial_offset = num_hops * self._num_threads
          shr_mem_index = f'{self._vm.lexic.threadIdx_x} + {finial_offset} + i * {self._lid_dim}'
          glb_mem_index = f'{self._vm.lexic.threadIdx_x} + {finial_offset} + i * {matrix.num_rows}'
          writer(f'{self._dest.name}[{shr_mem_index}] = {self._src.name}[{glb_mem_index}];')

  def __str__(self):
    return f'{self._dest.name} = load_g2s {self._shr_mem.name}, {self._src.name};'
