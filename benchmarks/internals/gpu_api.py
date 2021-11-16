from abc import ABC, abstractmethod


class GpuAPI:
  def __init__(self):
    pass

  @abstractmethod
  def alloc(self, addr, size):
    pass

  @abstractmethod
  def copy_to(self, dev, host, size):
    pass

  @abstractmethod
  def copy_from(self, host, dev, size):
    pass

  @abstractmethod
  def sync(self):
    pass

  @abstractmethod
  def dealloc(self, addr):
    pass

  @classmethod
  def get(self, arch):
    if arch == 'cuda':
      return Cuda()
    elif arch == 'hip':
      return Hip()
    else:
      return ValueError('unknown architecture')


class Cuda(GpuAPI):
  def __init__(self):
    super().__init__()

  def alloc(self, addr, size):
    return f'cudaMalloc(&{addr}, {size})'

  def copy_to(self, dev, host, size):
    return f'cudaMemcpy({dev}, {host}, {size}, cudaMemcpyHostToDevice)'

  def copy_from(self, host, dev, size):
    return f'cudaMemcpy({host}, {dev}, {size}, cudaMemcpyDeviceToHost)'

  def sync(self):
    return f'cudaDeviceSynchronize()'

  def dealloc(self, addr):
    return f'cudaFree({addr})'


class Hip(GpuAPI):
  def __init__(self):
    super().__init__()

  def alloc(self, addr, size):
    return f'hipMalloc(&{addr}, {size})'

  def copy_to(self, dev, host, size):
    return f'hipMemcpy({dev}, {host}, {size}, hipMemcpyHostToDevice)'

  def copy_from(self, host, dev, size):
    return f'hipMemcpy({host}, {dev}, {size}, hipMemcpyDeviceToHost)'

  def sync(self):
    return f'hipDeviceSynchronize()'

  def dealloc(self, addr):
    return f'hipFree({addr})'
