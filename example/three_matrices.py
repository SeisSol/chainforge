from chainforge.common import DenseMatrix
from chainforge.common import vm_factory
from chainforge.common.aux import generate_tmp_matrix
from chainforge.common import GemmDescr, FloatingPointType, Addressing
from chainforge.backend.generator import Generator


# D += A x (B x C)
mat_d = DenseMatrix(num_rows=56,
                    num_cols=9,
                    addressing=Addressing.STRIDED,
                    bbox=[0, 0, 56, 9])

mat_a = DenseMatrix(num_rows=56,
                    num_cols=56,
                    addressing=Addressing.STRIDED,
                    bbox=[0, 0, 56, 56])

mat_b = DenseMatrix(num_rows=56,
                    num_cols=9,
                    addressing=Addressing.STRIDED,
                    bbox=[0, 0, 56, 9])

mat_c = DenseMatrix(num_rows=9,
                    num_cols=9,
                    bbox=[0, 0, 9, 9],
                    addressing=Addressing.STRIDED)

tmp1 = generate_tmp_matrix(mat_b, mat_c)

gemm_list = [GemmDescr(trans_a=False,
                       trans_b=False,
                       a=mat_b,
                       b=mat_c,
                       c=tmp1),
             GemmDescr(trans_a=False,
                       trans_b=False,
                       a=mat_a, b=tmp1, c=mat_d,
                       alpha=1.0,
                       beta=1.0)]

vm = vm_factory(name='nvidia',
                sub_name='sm_60',
                fp_type=FloatingPointType.FLOAT)

generator = Generator(gemm_list, vm)
generator.generate()

print(generator.get_launcher())
print()
print(generator.get_header())
print()
print(generator.get_kernel())
