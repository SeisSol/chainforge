from chainforge.common import DenseMatrix
from chainforge.common import vm_factory
from chainforge.common.aux import generate_tmp_matrix
from chainforge.common import GemmDescr, FloatingPointType, Addressing
from chainforge.backend.generator import Generator


# Q += A * ((A^Trans * B) * C)
mat_q = DenseMatrix(num_rows=56,
                    num_cols=56,
                    addressing=Addressing.STRIDED,
                    bbox=[0, 0, 20, 8])

mat_a = DenseMatrix(num_rows=56,
                    num_cols=56,
                    addressing=Addressing.NONE,
                    bbox=[0, 0, 20, 8])


mat_b = DenseMatrix(num_rows=56,
                    num_cols=56,
                    addressing=Addressing.STRIDED,
                    bbox=[0, 0, 20, 8])

mat_c = DenseMatrix(num_rows=56,
                    num_cols=9,
                    bbox=[0, 0, 8, 8],
                    addressing=Addressing.NONE)

tmp1 = generate_tmp_matrix(mat_a, mat_b, True, False)
tmp2 = generate_tmp_matrix(tmp1, mat_c)


gemm_list = [GemmDescr(trans_a=True,
                       trans_b=False,
                       a=mat_a, b=mat_b, c=tmp1),
             GemmDescr(trans_a=False,
                       trans_b=False,
                       a=tmp1, b=mat_c, c=tmp2),
             GemmDescr(trans_a=False,
                       trans_b=False,
                       a=mat_b, b=tmp2, c=mat_q,
                       alpha=1.0,
                       beta=1.0)]


vm = vm_factory(name='nvidia',
                sub_name='sm_60',
                fp_type=FloatingPointType.FLOAT)

generator = Generator(gemm_list, vm)
generator.generate()

with_output = False
if with_output:
  print(generator.get_header())
  print(generator.default_generate_call_site())
  print()
  print(generator.get_launcher())
  print()
  print(generator.get_kernel())
