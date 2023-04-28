from abc import ABC, abstractmethod
from chainforge.backend.exceptions import GenerationError


class AbstractArchLexic(ABC):
  def __init__(self):
    self.thread_idx_x = None
    self.thread_idx_y = None
    self.thread_idx_z = None
    self.block_dim_x = None
    self.block_dim_y = None
    self.block_dim_z = None
    self.block_idx_x = None
    self.stream_type = None
    self.kenrnel_type = None
    self.shr_mem_kw = None
    self.dim3_type = None
    self.sync_block_threads = None
    self.sync_warp_threads = None
    self.restrict_kw = None

  def get_tid_counter(self, thread_id, block_dim, block_id):
    return f'({thread_id} + {block_dim} * {block_id}'

  @abstractmethod
  def get_launch_code(self, func_name, grid, block, stream, func_params):
    pass

  @abstractmethod
  def get_launch_bounds(self, total_num_threads_per_block, min_blocks_per_mp=None):
    pass


class AmdArchLexic(AbstractArchLexic):
  def __init__(self):
    AbstractArchLexic.__init__(self)
    self.thread_idx_y = 'hipThreadIdx_y'
    self.thread_idx_x = 'hipThreadIdx_x'
    self.thread_idx_z = 'hipThreadIdx_z'
    self.block_idx_x = 'hipBlockIdx_x'
    self.block_dim_x = 'hipBlockDim_x'
    self.block_dim_y = 'hipBlockDim_y'
    self.block_dim_z = 'hipBlockDim_z'
    self.stream_type = 'hipStream_t'
    self.kernel_type = '__global__ void'
    self.shr_mem_kw = '__shared__'
    self.dim3_type = 'dim3'
    self.sync_block_threads = '__syncthreads()'
    self.sync_warp_threads = '__syncthreads()'
    self.restrict_kw = '__restrict__'

  def get_launch_code(self, func_name, grid, block, stream, func_params):
    return f'hipLaunchKernelGGL({func_name}, {grid}, {block}, 0, {stream}, {func_params})'

  def get_launch_bounds(self, total_num_threads_per_block, min_blocks_per_mp=None):
    return ''


class NvidiaArchLexic(AbstractArchLexic):
  def __init__(self):
    AbstractArchLexic.__init__(self)
    self.thread_idx_y = 'threadIdx.y'
    self.thread_idx_x = 'threadIdx.x'
    self.thread_idx_z = 'threadIdx.z'
    self.block_idx_x = 'blockIdx.x'
    self.block_dim_x = 'blockDim.x'
    self.block_dim_y = 'blockDim.y'
    self.block_dim_z = 'blockDim.z'
    self.stream_type = 'cudaStream_t'
    self.kernel_type = '__global__ void'
    self.shr_mem_kw = '__shared__'
    self.dim3_type = 'dim3'
    self.sync_block_threads = '__syncthreads()'
    self.sync_warp_threads = '__syncwarp()'
    self.restrict_kw = '__restrict__'

  def get_launch_code(self, func_name, grid, block, stream, func_params):
    return f'{func_name}<<<{grid}, {block}, 0, {stream}>>>({func_params})'

  def get_launch_bounds(self, total_num_threads_per_block, min_blocks_per_mp=None):
    params = [str(item) for item in [total_num_threads_per_block, min_blocks_per_mp] if item]
    return f'__launch_bounds__({", ".join(params)})'


def lexic_factory(backend):
  if backend == "cuda":
    return NvidiaArchLexic()
  elif backend == "hip":
    return AmdArchLexic()
  else:
    raise GenerationError(f'unknown backend, given: {backend}')
