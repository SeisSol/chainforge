from chainforge.frontend import Parser, PostProcessor
from chainforge.common import FloatingPointType
from chainforge.backend.generator import Generator
from chainforge.common import Context
from internals import BenchGenerator, EnryPointGenerator, Aux
from os import path, makedirs
import sys
import shutil
import yaml
from copy import deepcopy
import argparse


def does_file_exist(file_name, format):
  file_suffix = file_name[-len(format):]
  if not file_suffix == format:
    raise ValueError(f'wrong format of an input file. '
                     f' expected `{format}` but given `{file_suffix}`')

  if not path.isfile(file_name):
    raise ValueError(f'input ({file_name}) is not a file')
  return True


def make_tmp_folder(file_name):
  cur_dir = path.dirname(__file__)
  tmp_dir = path.join(cur_dir, file_name)
  if path.exists(tmp_dir):
    shutil.rmtree(tmp_dir)
  makedirs(tmp_dir)
  return tmp_dir


def main():
  cmd = argparse.ArgumentParser()
  cmd.add_argument('-i', '--input', type=str, help="input file")
  cmd.add_argument('-c', '--config', type=str, help="config file")
  cmd.add_argument('-b', '--backend', type=str, help='gpu arch (cuda, hip)')
  cmd.add_argument('-a', '--arch', type=str, help='architecture e.g., sm_60, gfx906, etc.')
  args = cmd.parse_args()

  try:
    does_file_exist(args.input, format='.cf')
    does_file_exist(args.config, format='.yaml')
  except ValueError as err:
    print(f'{err}')
    sys.exit(-1)

  curr_dir = path.join(path.dirname(__file__))
  with open(f'{curr_dir}/{args.input}', 'r') as file:
    program = file.read()

  # get AST and append symbol table with temporaries
  parser = Parser()
  ast, symbol_table = parser.parse(translation_unit=program)

  # convert AST to lists of gemms
  symbol_table.add_scope()
  processor = PostProcessor(ast, symbol_table)
  gemm_dicts = processor.process()

  stream = open(args.config, 'r')
  config = yaml.safe_load(stream)
  context = Context(arch=args.arch,
                    backend=args.backend,
                    fp_type=FloatingPointType.str2enum(config['fp_type']))

  kernels = []; launchers = []; headers = []
  benchmarks_src = []; benchmarks_names = []
  for bench_name, gemm_list in gemm_dicts.items():
    gpu_generator = Generator(gemm_list, context)
    gpu_generator.set_kernel_name(bench_name)
    gpu_generator.generate()

    # write kernel, launcher and header to files
    kernels.append(gpu_generator.get_kernel())
    launchers.append(gpu_generator.get_launcher())
    headers.append(gpu_generator.get_header())

    call_site = Aux.get_call_site(gpu_generator, gemm_list)
    bench_generator = BenchGenerator(bench_name,
                                     deepcopy(symbol_table),
                                     gemm_list,
                                     config,
                                     call_site,
                                     args.backend)

    benchmarks_names.append(bench_name)
    benchmarks_src.append(bench_generator.generate())

  # generate main file
  enty_point = EnryPointGenerator(benchmarks_names, benchmarks_src, config)
  main_src = enty_point.generate()
  tmp_dir = make_tmp_folder(file_name='tmp')
  with open(path.join(tmp_dir, 'main.cu'), 'w') as file:
    file.write(main_src)

  # write kernel, launcher and header to files
  with open(path.join(tmp_dir, 'kernel.cu'), 'w') as file:
    file.write('#include \"chainforge_aux.h\"\n')
    for kernel, launcher in zip(kernels, launchers):
      file.write(kernel)
      file.write(launcher)

  with open(path.join(tmp_dir, 'kernel.h'), 'w') as file:
    file.write('#ifndef KERNEL_H\n')
    file.write('#define KERNEL_H\n')
    for header in headers:
      file.write(header)
    file.write('#endif\n')

  with open(path.join(tmp_dir, 'cmake_params.cmake'), 'w') as file:
    real_size = 8 if config['fp_type'] == 'double' else 4
    file.write(f'set(REAL_SIZE {real_size})\n')
    file.write(f'set(ARCH {args.arch})\n')
    file.write(f'set(BACKEND {args.backend})\n')


if __name__ == '__main__':
  main()
