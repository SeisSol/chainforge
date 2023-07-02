from chainforge.backend.writer import Writer
from chainforge.backend.symbol import DataView
from .abstract_loader import AbstractShrMemLoader, ShrMemLoaderType


class ExtendedPatchLoader(AbstractShrMemLoader):
  """A strategy which loads an entire matrix into shared memory.
  """

  def __init__(self, **kwargs):
    super(ExtendedPatchLoader, self).__init__(**kwargs)

    full_subvolume = (self._matrix.get_actual_num_cols() - 2) * self._matrix.num_rows
    cropped_subvolume = self._matrix.get_actual_num_rows() + self._matrix.num_rows
    self._shm_volume = cropped_subvolume + full_subvolume

    src_bbox = self._matrix.get_bbox()
    self._src.data_view = DataView(rows=self._matrix.num_rows,
                                   columns=self._matrix.num_cols,
                                   is_transposed=False,
                                   bbox=src_bbox)

    dst_bbox = [0, 0, src_bbox[2] - src_bbox[0], src_bbox[3] - src_bbox[1]]
    self._dest.data_view = DataView(rows=self._matrix.num_rows,
                                    columns=self._matrix.num_cols,
                                    is_transposed=False,
                                    bbox=dst_bbox)

  def get_loader_type(self):
    return ShrMemLoaderType.NOT_TRANSPOSED

  def gen_code(self, writer: Writer):
    super(ExtendedPatchLoader, self).gen_code(writer)
    writer(f'// loading {self._src.name} to {self._dest.name}: # no trans, extended')

    src_offset = self._src.data_view.get_offset()
    src_offset = f'{src_offset} + ' if src_offset else ''

    num_hops = int(self._shm_volume / self._num_threads)
    if num_hops > 0:
      writer.insert_pragma_unroll()
      with writer.block(f'for (int i = 0; i < {num_hops}; ++i)'):
        index = f'{self._vm.lexic.thread_idx_x} + i * {self._num_threads}'
        lhs = f'{self._dest.name}[{index}]'
        rhs = f'{self._src.name}[{src_offset}{index}]'
        writer(f'{lhs} = {rhs};')

    # the last hop to fill shared mem with data
    if (self._shm_volume % self._num_threads) != 0:
      residue = self._shm_volume - num_hops * self._num_threads
      with writer.block(f'if ({self._vm.lexic.thread_idx_x} < {residue})'):
        index = f'{self._vm.lexic.thread_idx_x} + {num_hops * self._num_threads}'
        lhs = f'{self._dest.name}[{index}]'
        rhs = f'{self._src.name}[{src_offset}{index}]'
        writer(f'{lhs} = {rhs};')

  def __str__(self):
    return f'{self._dest.name} = load_g2s_ext {self._shr_mem.name}, {self._src.name};'


class ExactPatchLoader(AbstractShrMemLoader):
  """A strategy which loads only a necessary part of a matrix into shared memory.
  """

  def __init__(self, **kwargs):
    super(ExactPatchLoader, self).__init__(**kwargs)
    self._shm_volume = self._matrix.get_actual_volume()
    self._src.data_view = DataView(rows=self._matrix.num_rows,
                                   columns=self._matrix.num_cols,
                                   is_transposed=False,
                                   bbox=self._matrix.get_bbox())

    self._dest.data_view = DataView(rows=self._src.data_view.get_dim_size(0),
                                    columns=self._src.data_view.get_dim_size(1),
                                    is_transposed=False)

  def get_loader_type(self):
    return ShrMemLoaderType.NOT_TRANSPOSED

  def gen_code(self, writer: Writer):
    super(ExactPatchLoader, self).gen_code(writer)
    writer(f'// loading {self._src.name} to {self._dest.name}: # no trans, exact.')

    num_data_rows = self._src.data_view.get_dim_size(0)
    src_offset = self._src.data_view.get_offset()
    src_offset = f'{src_offset} + ' if src_offset else ''

    with writer.block(f'for (int i = 0; i < {self._src.data_view.get_dim_size(1)}; ++i)'):

      num_hops = int(num_data_rows / self._num_threads)
      if num_hops > 0:

        writer.insert_pragma_unroll()
        with writer.block(f'for (int counter = 0; counter < {num_hops}; ++counter)'):
          shr_mem_index = f'{self._vm.lexic.thread_idx_x} + '
          shr_mem_index += f'counter * {self._num_threads} + i * {self._dest.data_view.get_lead_dim()}'
          lhs = f'{self._dest.name}[{shr_mem_index}]'

          glob_mem_index = f'{self._vm.lexic.thread_idx_x} + '
          glob_mem_index += f'counter * {self._num_threads} + i * {self._src.data_view.get_lead_dim()}'
          rhs = f'{self._src.name}[{src_offset}{glob_mem_index}]'
          writer(f'{lhs} = {rhs};')

      # the last hop to fill shared mem with data
      if (num_data_rows % self._num_threads) != 0:
        residue = num_data_rows - num_hops * self._num_threads
        with writer.block(f'if ({self._vm.lexic.thread_idx_x} < {residue})'):
          finial_offset = num_hops * self._num_threads
          shr_mem_index = f'{self._vm.lexic.thread_idx_x} + {finial_offset} + i * {self._dest.data_view.get_lead_dim()}'
          lhs = f'{self._dest.name}[{shr_mem_index}]'

          glb_mem_index = f'{self._vm.lexic.thread_idx_x} + {finial_offset} + i * {self._src.data_view.get_lead_dim()}'
          rhs = f'{self._src.name}[{src_offset}{glb_mem_index}]'
          writer(f'{lhs} = {rhs};')

  def __str__(self):
    return f'{self._dest.name} = load_g2s {self._shr_mem.name}, {self._src.name};'
