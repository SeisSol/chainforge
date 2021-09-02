from abc import ABC, abstractmethod
from chainforge.backend.exceptions import GenerationError


class AbstractArchLexic(ABC):
  def __init__(self):
    self.threadIdx_x = None
    self.threadIdx_y = None
    self.threadIdx_z = None
    self.blockDim_y = None
    self.blockDim_z = None
    self.blockIdx_x = None
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
    self.threadIdx_y = 'hipThreadIdx_y'
    self.threadIdx_x = 'hipThreadIdx_x'
    self.threadIdx_z = 'hipThreadIdx_z'
    self.blockIdx_x = 'hipBlockIdx_x'
    self.blockDim_y = 'hipBlockDim_y'
    self.blockDim_z = 'hipBlockDim_z'
    self.stream_type = 'hipStream_t'
    self.kernel_type = '__global__ void'
    self.shr_mem_kw = '__shared__'
    self.dim3_type = 'dim3'
    self.sync_block_threads = '__syncthreads()'
    self.sync_warp_threads = '__syncthreads()'
    self.restrict_kw = '__restrict__'

  def get_launch_code(self, func_name, grid, block, stream, func_params):
    return f'hipLaunchKernelGGL({func_name},{grid},{block},0,{stream},{func_params})'

  def get_launch_bounds(self, total_num_threads_per_block, min_blocks_per_mp=None):
    return ''


class NvidiaArchLexic(AbstractArchLexic):
  def __init__(self):
    AbstractArchLexic.__init__(self)
    self.threadIdx_y = 'threadIdx.y'
    self.threadIdx_x = 'threadIdx.x'
    self.threadIdx_z = 'threadIdx.z'
    self.blockIdx_x = 'blockIdx.x'
    self.blockDim_y = 'blockDim.y'
    self.blockDim_z = 'blockDim.z'
    self.stream_name = 'cudaStream_t'
    self.kernel_type = '__global__ void'
    self.shr_mem_kw = '__shared__'
    self.dim3_type = 'dim3'
    self.sync_block_threads = '__syncthreads()'
    self.sync_warp_threads = '__syncwarp()'
    self.restrict_kw = '__restrict__'

  def get_launch_code(self, func_name, grid, block, stream, func_params):
    return f'{func_name}<<<{grid},{block},0,{stream}>>>({func_params})'

  def get_launch_bounds(self, total_num_threads_per_block, min_blocks_per_mp=None):
    params = [str(item) for item in [total_num_threads_per_block, min_blocks_per_mp] if item]
    return f'__launch_bounds__({", ".join(params)})'


def lexic_factory(arch_name):
  if arch_name == "nvidia":
    return NvidiaArchLexic()
  elif arch_name == "amd":
    return AmdArchLexic()
  else:
    raise GenerationError('unknown architecture provided')
