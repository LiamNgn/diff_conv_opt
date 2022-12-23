from my_lib import DiffConv as DiffConv
import unittest
import torch
import cvxpy as cp
import numpy.random as npr
from torch.autograd import grad


def set_seed(x):
    npr.seed(x)
    torch.manual_seed(x)


class TestDiffConv(unittest.TestCase):
    def test_example(self):
        n, m = 20, 25
        x = cp.Variable(n)
        A = cp.Parameter((m, n))
        b = cp.Parameter(m)
        constraints = [x >= 0, 1 >= x]
        objective = cp.Minimize(cp.pnorm(A @ x - b, p=2))
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()

        cvxpylayer = DiffConv(problem, parameters=[A, b], variables=[x])
        A_tch = torch.randn(m, n, requires_grad=True)
        b_tch = torch.randn(m, requires_grad=True)

        # solve the problem
        (solution,) = cvxpylayer(A_tch, b_tch)

        # compute the gradient of the sum of the solution with respect to A, b
        solution.sum().backward()


if __name__ == "__main__":
    unittest.main()
