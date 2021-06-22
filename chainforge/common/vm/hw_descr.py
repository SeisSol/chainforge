class HwDecription:
  def __init__(self,
               vec_unit_length,
               max_local_mem_size_per_block,
               max_threads_per_block,
               max_reg_per_block,
               max_threads_per_sm,
               max_block_per_sm,
               name):
    self.vec_unit_length = vec_unit_length
    self.max_local_mem_size_per_block = max_local_mem_size_per_block
    self.max_threads_per_block = max_threads_per_block
    self.max_reg_per_block = max_reg_per_block
    self.max_threads_per_sm = max_threads_per_sm
    self.max_block_per_sm = max_block_per_sm
    self.name = name


def hw_descr_factory(name: str, sub_name: str):
  KB = 1024
  if name == "nvidia":

    # from: https://en.wikipedia.org/wiki/CUDA
    nvidia_warp = 32
    max_reg_per_block = 64 * KB
    max_threads_per_block = 1024
    max_threads_per_sm = 2048
    max_block_per_sm = 32
    if sub_name in ['sm_60', 'sm_61', 'sm_62']:
      max_local_mem_size_per_block = 48 * KB
    elif sub_name == 'sm_70':
      max_local_mem_size_per_block = 96 * KB
    elif sub_name == 'sm_71':
      max_local_mem_size_per_block = 48 * KB
    elif sub_name == 'sm_75':
      max_block_per_sm = 16
      max_threads_per_sm = 1024
      max_local_mem_size_per_block = 64 * KB
    elif sub_name == 'sm_80':
      max_local_mem_size_per_block = 164 * KB
    elif sub_name == 'sm_86':
      max_block_per_sm = 16
      max_threads_per_sm = 1536
      max_local_mem_size_per_block = 100 * KB

    else:
      raise ValueError(f'Given nvidia SM model is not supported. Provided: {sub_name}')

    return HwDecription(nvidia_warp,
                        max_local_mem_size_per_block,
                        max_threads_per_block,
                        max_reg_per_block,
                        max_threads_per_sm,
                        max_block_per_sm,
                        name)

  elif name == "amd":
    amd_wavefront = 64
    max_reg_per_workgroup = 256 * KB
    max_threads_per_block = 1024
    if sub_name in ['gfx906']:
      max_workgroup_per_cu = 40
      max_vec_units_per_cu = 64
      max_local_mem_size_per_workgroup = 64 * KB
    else:
      raise ValueError(f'Given amd CU model is not supported. Provided: {sub_name}')

    return HwDecription(amd_wavefront,
                        max_local_mem_size_per_workgroup,
                        max_threads_per_block,
                        max_reg_per_workgroup,
                        max_vec_units_per_cu,
                        max_workgroup_per_cu,
                        name)

  else:
    raise ValueError('Unknown gpu architecture')
