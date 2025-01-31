#%%

import numpy as np
import matplotlib.pyplot as plt
# print("svailables stules", plt.style.available)
# plt.style.use("seaborn-v0_8-white")
from scipy.integrate import solve_ivp
import argparse
import json
import ast
import math
from PIL import Image

def parse_arguments():
    try:
        __IPYTHON__
        _in_ipython_session = True
    except NameError:
        _in_ipython_session = False

    if _in_ipython_session:
        args = argparse.Namespace(split='adapt_train', 
                                  savepath="../data_2D_small*/", 
                                  seed=2024, 
                                  verbose=1, 
                                  dimension=2,
                                  nb_steps=100)
        return args

    else:
        parser = argparse.ArgumentParser(description='Generate ODE data for multiple dynamical systems')
        parser.add_argument('--split', type=str, choices=['train', 'test', 'adapt_train', 'adapt_test'], default='train', help='Data split to generate')
        parser.add_argument('--savepath', type=str, default='../tmp/', help='Path to save generated data')
        parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
        parser.add_argument('--verbose', type=int, default=1, help='Verbosity level')
        parser.add_argument('--dimension', type=int, default=2, help='Dimension of ODEs')
        parser.add_argument('--nb_steps', type=int, default=100, help='Number of time steps to simulate')
        return parser.parse_args()

def parse_lambda(lambda_str):
    """Parse a lambda function string and return a callable function."""
    lambda_ast = ast.parse(lambda_str).body[0].value
    return eval(compile(ast.Expression(lambda_ast), '<string>', 'eval'))

def load_ode_definitions(dimension):
    # Load ODE definitions: Symbolic Regression of Dynamical Systems with Transformers
    with open(f'ode_definitions_{dimension}D.json', 'r') as f:
        ode_defs = json.load(f)

    # Parse the lambda functions
    for ode in ode_defs.values():
        ode['function'] = parse_lambda(ode['function'])
    
    return ode_defs

def generate_environments(reference_params, n_envs, adaptation=False):
    if adaptation:
        # Generate 2 environments in training domain and 2 outside
        envs = []
        scalings = np.linspace(0.8, 1.2, n_envs)
        for i in range(n_envs):
            env = {}
            for param, value in reference_params.items():
                env[param] = np.round(value * scalings[i], 2)
            envs.append(env)
    else:
        # Generate training environments
        envs = []
        # for _ in range(n_envs):
        #     env = {}
        #     for param, value in reference_params.items():
        #         env[param] = value * np.random.uniform(0.8, 1.2)
        #     envs.append(env)
        # print(envs)

        ## Keep one paramter constant and vary the other, repeat until n_envs   
        # envs = []
        # env_count = 0
        # while env_count < n_envs:
        #     for focus_param, focus_value in reference_params.items():
        #         env = {}
        #         env[focus_param] = focus_value * np.random.uniform(0.8, 1.2)
        #         for param, value in reference_params.items():
        #             if param != focus_param:
        #                 env[param] = value * np.random.uniform(0.8, 1.2)
        #         envs.append(env)
        #         env_count += 1
        #         if env_count >= n_envs:
        #             break

        ## Make a grid with log(nenvs)/log(nparams) points in each dimension
        n_params = len(reference_params)
        n_per_dim = math.ceil(n_envs ** (1/n_params))
        param_values = [np.round(np.linspace(0.8*ref, 1.2*ref, n_per_dim), 2) for ref in reference_params.values()]

        # print("param_values", param_values)
        # for param_values_comb in np.meshgrid(*param_values):
        #     print("param_values_comb", param_values_comb)

        for param_values_comb in np.array(np.meshgrid(*param_values)).T.reshape(-1, n_params):

            env = {param: value for param, value in zip(reference_params.keys(), param_values_comb)}
            envs.append(env)
            if len(envs) >= n_envs:
                break

        print("     Original Parameters", reference_params)
        print("     Created Environments:", envs)

    return envs

def generate_initial_conditions(reference_ic, n_ic):
    ic1, ic2 = reference_ic

    # return [np.array(reference_ic) * np.random.uniform(0.8, 1.2, size=len(reference_ic)) for _ in range(n_ic)]
    # return [np.array(reference_ic)]

    ## Sample a paramter to interpolate between the two initial conditions
    # alpha = np.random.uniform(0, 1, size=1)
    # return [alpha * np.array(ic1) + (1 - alpha) * np.array(ic2)]

    # ## The two reference ics, then n_ic-2 interpolates between the two initial conditions
    # ic1, ic2 = np.array(ic1), np.array(ic2)
    # # return [ic1, ic2] + [alpha * ic1 + (1 - alpha) * ic2 for alpha in np.linspace(0.1, 0.9, n_ic-2)]
    # # return [ic1, ic2] + [alpha * ic1 + (1 - alpha) * ic2 for alpha in np.random.uniform(0, 2, size=ic1.shape)]
    # ret = [ic1]
    # for _ in range(n_ic-2):
    #     alpha = np.round(np.random.uniform(0.1, 0.9, size=ic1.shape), 2)
    #     # ret.append(alpha * ic1 + (1 - alpha) * ic2)
    #     beta = np.round(np.random.uniform(0.1, 0.9, size=ic1.shape), 2)
    #     normed_alpha = alpha / (alpha + beta)
    #     normed_beta = beta / (alpha + beta)
    #     ret.append(normed_alpha * ic1 + normed_beta * ic2)
    # ret.append(ic2)
    # return ret[:n_ic]

    ## Sample the initial condition component by component to interpolate between the two initial conditions. Uniform distribution between the min and max of the two condition, dimension-wise
    ic1, ic2 = np.array(ic1), np.array(ic2)
    nb_dims = ic1.shape[0]
    mins = np.minimum(ic1, ic2)
    maxs = np.maximum(ic1, ic2)
    eps = 1e-3

    ret = [ic1]
    for _ in range(n_ic-2):
        ic = [np.random.uniform(mins[i]+eps, maxs[i]-eps) for i in range(nb_dims)]
        ret.append(ic)
    ret.append(ic2)
    return ret[:n_ic]

    ## Sample a paramter to interpolate. Higer probability to sample either of the two initial conditions themselves. Have one sample paramter for each dimension
    # initial_conditions = []
    # for _ in range(n_ic):
    #     alpha = np.random.uniform(0, 1, size=len(reference_ic))
    #     # alpha = np.where(alpha < 0.1, 0, alpha)
    #     # alpha = np.where(alpha > 0.9, 1, alpha)
    #     initial_conditions.append(alpha * np.array(ic1) + (1 - alpha) * np.array(ic2))

    # return initial_conditions

    ## Rndomly pick one of the two initial conditions
    # return [np.array(ic1) if np.random.uniform(0, 1) < 0.5 else np.array(ic2) for _ in range(n_ic)]


def simulate_ode(ode_func, t_span, initial_state, args, dt):
    t_eval = np.arange(t_span[0], t_span[1], dt)
    # print("t_eval", t_eval.shape)
    solution = solve_ivp(ode_func, t_span, initial_state, args=args, t_eval=t_eval, method='RK45')
    # print("solution", solution.y.shape, t_eval.shape, t_eval[-1], solution.t[-1])
    return solution.t, solution.y.T

def main():
    args = parse_arguments()
    np.random.seed(args.seed)

    ode_definitions = load_ode_definitions(args.dimension)

    if args.split == 'train':
        # n_envs, n_ic = 16, 4
        n_envs, n_ic = 5, 4
    elif args.split == 'test':
        # n_envs, n_ic = 16, 32
        n_envs, n_ic = 5, 32
    elif args.split == 'adapt_train':
        # n_envs, n_ic = 4, 1
        n_envs, n_ic = 1, 1
    elif args.split == 'adapt_test':
        # n_envs, n_ic = 4, 32
        n_envs, n_ic = 1, 32

    all_data = []
    all_t_eval = []
    all_environments = {}

    for ode_id, ode_info in ode_definitions.items():
        if args.verbose:
            print(f"Processing ODE {ode_id}: {ode_info['name']}")

        # Generate environments and initial conditions
        environments = generate_environments(ode_info['parameters'], n_envs, args.split in ['adapt', 'adapt_test'])
        initial_conditions = generate_initial_conditions(ode_info['initial_values'], n_ic)
        # print("initial_conditions", initial_conditions)

        # Adjust time parameters
        T = ode_info.get("time_horizon", 1)
        dt = T / args.nb_steps  # Approximately 20 time steps per ODE

        ode_data = np.zeros((n_envs, n_ic, int(T/dt), len(ode_info['initial_values'])))

        for i, env in enumerate(environments):
            for j, ic in enumerate(initial_conditions):
                # print("Currently using IC", ic)
                t, trajectory = simulate_ode(ode_info['function'], (0, T), ic, tuple(env.values()), dt)
                ode_data[i, j] = trajectory

        all_data.append(ode_data)
        all_t_eval.append(t)
        all_environments[ode_id] = environments

    # Save data
    filename = f"{args.savepath}/{args.split}.npz"
    np.savez(filename, t=np.stack(all_t_eval), X=np.stack(all_data))

    ## Save normalising constants
    norm_consts = np.max(np.abs(all_data), axis=(2,3), keepdims=True)
    np.save(f"{args.savepath}/{args.split}_bounds.npy", norm_consts)

    # Save environments
    with open(f"{args.savepath}/{args.split}_envs.json", 'w') as f:
        json.dump(all_environments, f, indent=4)

    if args.verbose:
        print(f"Data saved to {filename}")



if __name__ == "__main__":
    main()






#%%

if __name__ == "__main__":

    args = parse_arguments()
    ode_defs = load_ode_definitions(args.dimension)

    ## Collect all the plots to form a gif
    all_plots = []
    ## Load the data
    filename = f"{args.savepath}/{args.split}.npz"
    data = np.load(filename)
    print(data['t'].shape, data['X'].shape)

    for ode_id, ode_def in enumerate(ode_defs.keys()):
        t = data['t'][ode_id]
        X = data['X'][ode_id]

        plt.figure(figsize=(10, 4))

        ## Plot all environments from the same initial condition
        for i in range(X.shape[1]):
            # print('IC is ', X[0, i, 0])
            plt.plot(t, X[:, i, :, 0].T, color="royalblue", alpha=(i+1)/(X.shape[1]+1))
            plt.plot(t, X[:, i, :, 1].T, color="crimson", alpha=(i+1)/(X.shape[1]+1))

        plt.title(f"ODEBench - {ode_def}")

        plt.xlabel("Time")
        # plt.legend()
        # plt.show()
        plt.draw()

        # all_plots.append(plt.gcf())
        plt.savefig(f"../tmp/odebench_{ode_def}.png")
        all_plots.append(Image.open(f"../tmp/odebench_{ode_def}.png"))

    ## Save the plots as a gif
    all_plots[0].save("odebench_2D.gif", save_all=True, append_images=all_plots[1:], duration=100, loop=0, optimize=True)

