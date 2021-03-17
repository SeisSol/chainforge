from jinja2 import Environment, FileSystemLoader
import os

class Aux:
  @classmethod
  def print_bar(cls, length=80):
    print('*' * length)

  @classmethod
  def read_template(self, template_name):
    file_path = os.path.join(os.path.dirname(__file__), './templates')
    env = Environment(loader=FileSystemLoader(searchpath=file_path))
    return env.get_template(template_name)

  @classmethod
  def get_call_site(cls, kernel_generator, gemm_list):
    mat_names = {}
    offset_names = {}
    for gemm in gemm_list:
      mat_list = [gemm.mat_a, gemm.mat_b, gemm.mat_c]
      for mat in mat_list:
        mat_names[mat.alias] = f'dev_{mat.alias}'
        offset_names[mat.alias] = '0'

    call_site = kernel_generator.generate_call_site(mat_name_map=mat_names,
                                                    offset_name_map=offset_names,
                                                    alpha=gemm_list[-1].alpha,
                                                    beta=gemm_list[-1].beta,
                                                    num_element='num_elements',
                                                    stream=None)
    return call_site
