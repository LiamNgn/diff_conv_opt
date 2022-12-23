import torch
from cvxpy.reductions.solvers.conic_solvers.scs_conif import dims_to_solver_dict
import numpy as np
import diffcp
import cvxpy as cp


def to_numpy(x: torch.FloatTensor) -> np.ndarray:
    # convert torch tensor to numpy array
    return x.cpu().detach().double().numpy()


def to_torch(x: np.ndarray, dtype = str, device = str) -> torch.FloatTensor:
    # convert numpy array to torch tensor
    return torch.from_numpy(x).type(dtype).to(device)


class DiffConv(torch.nn.Module):
    def __init__(self,
                 problem: cp.problems.problem.Problem,
                 parameters: cp.expressions.constants.parameter.Parameter,
                 variables: cp.expressions.variable.Variable):
        super(DiffConv, self).__init__()
        self.variables = variables
        self.var_lst = {v.id for v in self.variables}
        data, _, _ = problem.get_problem_data(solver=cp.SCS)
        self.compiler = data[cp.settings.PARAM_PROB]
        self.id_param = [p.id for p in parameters]
        self.cone_dims = dims_to_solver_dict(data["dims"])

    def forward(self, *params):
        f = _diff_opt_fn(
            id_param=self.id_param,
            variables=self.variables,
            var_lst=self.var_lst,
            compiler=self.compiler,
            cone_dims=self.cone_dims,
        )
        sol = f(*params)
        return sol


def _diff_opt_fn(
    id_param: list,
    variables: cp.expressions.variable.Variable,
    var_lst: set,
    compiler: cp.reductions.dcp2cone.cone_matrix_stuffing.ParamConeProg,
    cone_dims: dict,
) -> builtin_function_or_method:
    class DiffOptFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx: torch.autograd.function.DiffOptFnBackward, *params) -> tuple:
            # infer dtype, device, and whether or not params are batched
            ctx.dtype = params[0].dtype
            ctx.device = params[0].device
            params_numpy = []
            for p in params:
                params_numpy.append(to_numpy(p))

            # canonicalize problem
            As, bs, cs, cone_dicts, ctx.shapes = [], [], [], [], []

            params_numpy_i = []
            for p, sz in zip(params_numpy, [0, 0]):
                if sz == 0:
                    params_numpy_i.append(p)
                else:
                    params_numpy_i.append(p[0])

            c, _, neg_A, b = compiler.apply_parameters(
                dict(zip(id_param, params_numpy_i)), keep_zeros=True
            )
            A = -neg_A  # cvxpy canonicalizes -A
            As.append(A)
            bs.append(b)
            cs.append(c)
            cone_dicts.append(cone_dims)
            ctx.shapes.append(A.shape)
            xs, _, _, _, ctx.DT_batch = diffcp.solve_and_derivative_batch(
                As, bs, cs, cone_dicts
            )

            # extract solutions and append along batch dimension
            sol = [[] for _ in range(len(variables))]
            sltn_dict = compiler.split_solution(xs[0], active_vars=var_lst)
            for j, v in enumerate(variables):
                sol[j].append(
                    to_torch(sltn_dict[v.id], ctx.dtype, ctx.device).unsqueeze(0)
                )
            return tuple([torch.cat(s, 0).squeeze(0) for s in sol])

        @staticmethod
        def backward(ctx: torch.autograd.function.DiffOptFnBackward, *dvars) -> tuple:
            dvars_numpy_first = []
            for dvar in dvars:
                dvars_numpy_first.append(to_numpy(dvar))
            dvars_numpy = []
            for dvar in dvars_numpy_first:
                dvars_numpy.append(np.expand_dims(dvar, 0))
            dxs, dys, dss = (
                [],
                [np.zeros(ctx.shapes[0][0])],
                [np.zeros(ctx.shapes[0][0])],
            )
            del_vars = {}
            for v, dv in zip(variables, [dv[0] for dv in dvars_numpy]):
                del_vars[v.id] = dv
            dxs.append(compiler.split_adjoint(del_vars))
            dAs, dbs, dcs = ctx.DT_batch(dxs, dys, dss)
            grad = [[] for _ in range(len(id_param))]
            del_param_dict = compiler.apply_param_jac(dcs[0], -dAs[0], dbs[0])
            for j, pid in enumerate(id_param):
                grad[j] += [
                    to_torch(del_param_dict[pid], ctx.dtype, ctx.device).unsqueeze(0)
                ]
            return tuple([torch.cat(g, 0).squeeze(0) for g in grad])

    return DiffOptFn.apply
