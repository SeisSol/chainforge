from chainforge.common import DenseMatrix
from chainforge.common import Context
from chainforge.common.aux import generate_tmp_matrix
from chainforge.common import GemmDescr, FloatingPointType, Addressing
from chainforge.backend.generator import Generator


# Q += A x ((A^T x B) x C)

variants = {'v0': Addressing.STRIDED,
            'v1': Addressing.NONE}

mat_q = DenseMatrix(num_rows=56,
                    num_cols=56,
                    addressing=Addressing.STRIDED,
                    bbox=[0, 0, 20, 9])

mat_a = DenseMatrix(num_rows=56,
                    num_cols=56,
                    addressing=variants['v0'],
                    bbox=[0, 0, 20, 9])

mat_b = DenseMatrix(num_rows=56,
                    num_cols=56,
                    addressing=Addressing.STRIDED,
                    bbox=[0, 0, 20, 9])

mat_c = DenseMatrix(num_rows=56,
                    num_cols=9,
                    bbox=[0, 0, 9, 9],
                    addressing=Addressing.STRIDED)


tmp1 = generate_tmp_matrix(mat_a, mat_b, True, False)
tmp2 = generate_tmp_matrix(tmp1, mat_c)


gemm_list = [GemmDescr(trans_a=True,
                       trans_b=False,
                       a=mat_a, b=mat_b, c=tmp1),
             GemmDescr(trans_a=False,
                       trans_b=False,
                       a=tmp1, b=mat_c, c=tmp2),
             GemmDescr(trans_a=False, trans_b=False,
                       a=mat_a, b=tmp2, c=mat_q,
                       alpha=1.0,
                       beta=1.0)]

context = Context(name='nvidia',
                  sub_name='sm_60',
                  fp_type=FloatingPointType.FLOAT)

generator = Generator(gemm_list, context)
generator.generate()

print(generator.get_launcher())
print()
print(generator.get_header())
print()
print(generator.get_kernel())