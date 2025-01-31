from ._config import *

# import jax
import jax.numpy as jnp

from scipy.interpolate import griddata

import torch

import optax
from functools import partial

import time
# import cProfile





def seconds_to_hours(seconds):
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return hours, minutes, seconds

## Simply returns a suitable key for all jax operations
def get_new_key(key=None, num=1):
    if key is None:
        print("WARNING: No key provided, using time as seed")
        key = jax.random.PRNGKey(time.time_ns())

    elif isinstance(key, int):
        key = jax.random.PRNGKey(key)

    keys = jax.random.split(key, num=num)

    return keys if num > 1 else keys[0]

def generate_new_keys(key=None, num=1):
    if key is None:
        print("WARNING: No key provided, using time as seed")
        key = jax.random.PRNGKey(time.time_ns())

    elif isinstance(key, int):
        key = jax.random.PRNGKey(key)

    return jax.random.split(key, num=num)

## Wrapper function for matplotlib and seaborn
def sbplot(*args, ax=None, figsize=(6,3.5), x_label=None, y_label=None, title=None, x_scale='linear', y_scale='linear', xlim=None, ylim=None, **kwargs):
    if ax==None:
        _, ax = plt.subplots(1, 1, figsize=figsize)
    # sns.despine(ax=ax)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    ax.plot(*args, **kwargs)
    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)
    if "label" in kwargs.keys():
        ax.legend()
    if ylim:
        ax.set_ylim(ylim)
    if xlim:
        ax.set_xlim(xlim)
    plt.tight_layout()
    return ax

## Wrapper function for matplotlib and seaborn for imshow
def sbimshow(*args, ax=None, figsize=(6,3.5), x_label=None, y_label=None, title=None, x_scale='linear', y_scale='linear', xlim=None, ylim=None, **kwargs):
    if ax==None:
        _, ax = plt.subplots(1, 1, figsize=figsize)
    # sns.despine(ax=ax)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    ax.imshow(*args, **kwargs)
    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)
    if "label" in kwargs.keys():
        ax.legend()
    if ylim:
        ax.set_ylim(ylim)
    if xlim:
        ax.set_xlim(xlim)
    plt.tight_layout()
    return ax

## Alias for sbplot
def plot(*args, ax=None, figsize=(6,3.5), x_label=None, y_label=None, title=None, x_scale='linear', y_scale='linear', xlim=None, ylim=None, **kwargs):
  return sbplot(*args, ax=ax, figsize=figsize, x_label=x_label, y_label=y_scale, title=title, x_scale=x_scale, y_scale=y_scale, xlim=xlim, ylim=ylim, **kwargs)


def pvplot(x, y, show=True, xlabel=None, ylabel=None, title=None, ax=None, **kwargs):
    import pyvista as pv
    # pv.start_xvfb()           ## TODO Only do this on LINUX

    if ax is None:
        ax = pv.Chart2D()

    _ = ax.line(x, y, **kwargs)

    if xlabel is not None:
        ax.x_label = xlabel

    if ylabel is not None:
        ax.y_label = ylabel

    if title is not None:
        ax.title = title

    if show == True:
        ax.show()

    return ax


def flatten_pytree(pytree):
    """ Flatten the leaves of a pytree into a single array. Return the array, the shapes of the leaves and the tree_def. """

    leaves, tree_def = jax.tree_util.tree_flatten(pytree)
    flat = jnp.concatenate([x.flatten() for x in leaves])
    shapes = [x.shape for x in leaves]
    return flat, shapes, tree_def

def unflatten_pytree(flat, shapes, tree_def):
    """ Reconstructs a pytree given its leaves flattened, their shapes, and the treedef. """

    leaves_prod = [0]+[np.prod(x) for x in shapes]

    lpcum = np.cumsum(leaves_prod)
    leaves = [flat[lpcum[i-1]:lpcum[i]].reshape(shapes[i-1]) for i in range(1, len(lpcum))]

    return jax.tree_util.tree_unflatten(tree_def, leaves)


# def default_optimizer_schedule(init_lr, nb_epochs):
#     return optax.piecewise_constant_schedule(init_value=init_lr,
#                         boundaries_and_scales={int(nb_epochs*0.25):0.2,
#                                                 int(nb_epochs*0.5):0.1,
#                                                 int(nb_epochs*0.75):0.01})


def get_id_current_time():
    """ Returns a string of the current time in the format as an ID """
    # return time.strftime("%Y%m%d-%H%M%S")
    return time.strftime("%H%M%S")



def vec_to_mats(vec_uv, res=32, nb_mats=2):
    """ Reshapes a vector into a set of 2D matrices """
    UV = jnp.split(vec_uv, nb_mats)
    return [jnp.reshape(UV[i], (res, res)) for i in range(nb_mats)]

def mats_to_vec(mats, res):
    """ Flattens a set of 2D matrices into a single vector """
    return jnp.concatenate([jnp.reshape(mats[i], res * res) for i in range(len(mats))])





## Function to calculate losses
def params_norm(params):
    """ norm of the parameters """
    return jnp.array([jnp.linalg.norm(x) for x in jax.tree_util.tree_leaves(params)]).sum()

def params_diff_norm(params1, params2):
    """ norm of the parameters difference"""
    params1 = eqx.filter(params1, eqx.is_array, replace=jnp.zeros(1))
    params2 = eqx.filter(params2, eqx.is_array, replace=jnp.zeros(1))

    # diff_tree = jax.tree_util.tree_map(lambda x, y: x-y, params1, params2)
    # return params_norm(diff_tree)

    # return jnp.array([jnp.linalg.norm(x-y) for x, y in zip(jax.tree_util.tree_leaves(params1), jax.tree_util.tree_leaves(params2))]).sum()

    ## Flatten the difference and calculate the norm
    diff_flat, _, _ = flatten_pytree(jax.tree_util.tree_map(lambda x, y: x-y, params1, params2))
    return jnp.linalg.norm(diff_flat)

@eqx.filter_jit
def params_diff_norm_squared(params1, params2):
    """ normalised squared norm of the parameters difference """
    params1 = eqx.filter(params1, eqx.is_array, replace=jnp.zeros(1))
    params2 = eqx.filter(params2, eqx.is_array, replace=jnp.zeros(1))
    diff_flat, _, _ = flatten_pytree(jax.tree_util.tree_map(lambda x, y: x-y, params1, params2))
    return (diff_flat.T@diff_flat) / diff_flat.shape[0]

@eqx.filter_jit
def params_norm_squared(params):
    """ normalised squared norm of the parameter """
    params = eqx.filter(params, eqx.is_array, replace=jnp.zeros(1))
    diff_flat, _, _ = flatten_pytree(params)
    return (diff_flat.T@diff_flat) / diff_flat.shape[0]



def spectral_norm(params):
    """ spectral norm of the parameters """
    return jnp.array([jnp.linalg.svd(x, compute_uv=False)[0] for x in jax.tree_util.tree_leaves(params) if jnp.ndim(x)==2]).sum()

def spectral_norm_estimation(model, nb_iters=5, *, key=None):
    """ estimating the spectral norm with the power iteration: https://arxiv.org/abs/1802.05957 """
    params = eqx.filter(model, eqx.is_array)
    matrices = [x for x in jax.tree_util.tree_leaves(params) if jnp.ndim(x)==2]
    nb_matrices = len(matrices)
    keys = generate_new_keys(key, num=nb_matrices)
    us = [jax.random.normal(k, (x.shape[0],)) for k, x in zip(keys, matrices)]
    vs = [jax.random.normal(k, (x.shape[1],)) for k, x in zip(keys, matrices)]

    for _ in range(nb_iters):
        for i in range(nb_matrices):
            vs[i] = matrices[i].T@us[i]
            vs[i] = vs[i] / jnp.linalg.norm(vs[i])
            us[i] = matrices[i]@vs[i]
            us[i] = us[i] / jnp.linalg.norm(us[i])

    sigmas = [u.T@x@v for x, u, v in zip(matrices, us, vs)]
    return jnp.array(sigmas).sum()

def infinity_norm_estimation(model, xs, ctx):
    xs_flat = jnp.reshape(xs, (-1, xs.shape[-1]))
    ys = jax.vmap(model, in_axes=(None, 0, None))(None, xs_flat, ctx)
    return jnp.mean(jnp.linalg.norm(ys, axis=-1) / jnp.linalg.norm(xs_flat, axis=-1))

def l2_norm_traj(xs, xs_hat):
    total_loss = jnp.mean((xs - xs_hat)**2, axis=-1)   ## TODO mean or sum ? Norm of d-dimensional vectors
    return jnp.sum(total_loss) / (xs.shape[-2] * xs.shape[-3])






def RK4(fun, t_span, y0, args, *, t_eval=None, subdivisions=1, **kwargs):
    """ Perform numerical integration with a time step divided by the evaluation subdivision factor (Not necessarily equally spaced). If we get NaNs, we can try to increasing the subdivision factor for finer time steps."""
    if t_eval is None:
        if t_span[0] is None:
            t_eval = jnp.array([t_span[1]])
            raise Warning("t_span[0] is None. Setting t_span[0] to 0.")
        elif t_span[1] is None:
            raise ValueError("t_span[1] must be provided if t_eval is not.")
        else:
            t_eval = jnp.array(t_span)

    hs = t_eval[1:] - t_eval[:-1]
    t_ = t_eval[:-1, None] + jnp.arange(subdivisions)[None, :]*hs[:, None]/subdivisions
    t_solve = jnp.concatenate([t_.flatten(), t_eval[-1:]])
    eval_indices = jnp.arange(0, t_solve.size, subdivisions)

    def step(state, t):
        t_prev, y_prev = state
        h = t - t_prev
        k1 = h * fun(t_prev, y_prev, args)
        k2 = h * fun(t_prev + h/2., y_prev + k1/2., args)
        k3 = h * fun(t_prev + h/2., y_prev + k2/2., args)
        k4 = h * fun(t + h, y_prev + k3, args)
        y = y_prev + (k1 + 2*k2 + 2*k3 + k4) / 6.
        return (t, y), y

    _, ys = jax.lax.scan(step, (t_solve[0], y0), t_solve[:])
    return ys[eval_indices, :]


def count_params(module):
    return sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(module, eqx.is_array)) if x is not None)










GAUSSIAN_ATTRIBUTE_COUNT_2D = 8         ## Global variable for the number of attributes in a 2D gaussian

def init_gaussian(key, width=32., height=32.) -> jnp.ndarray:
    """Returns the initial model params."""
    keys = jax.random.split(key, 6)

    ## Uniformly initialise parameters of a 2D gaussian
    # mean = jax.random.uniform(keys[0], (2,), minval=0, maxval=min(width, height))
    # scaling = jax.random.uniform(keys[1], (2,), minval=0, maxval=min(width, height)/1)

    # mean = jax.random.uniform(keys[0], (2,), minval=0.1, maxval=0.90)
    max_side = max(width, height)
    scaling = jax.random.uniform(keys[1], (2,), minval=1/(max_side*1000), maxval=1/(max_side*5))
    rotation = jax.random.uniform(keys[2], (1,), minval=0, maxval=1.)
    colour = jax.random.uniform(keys[3], (3,), minval=0, maxval=1)


    mean = jnp.array([0.5, 0.5])
    # scaling = jnp.array([1/(width*20), 1/(height*20)])
    # rotation = jnp.array([0.5])
    # colour = jnp.array([0.5, 0.5, 0.5])
    # # colour = jax.random.uniform(keys[3], (3,), minval=0, maxval=1)

    # opacity = jax.random.uniform(keys[4], (1,), minval=0, maxval=1)
    # objectness = jax.random.uniform(keys[5], (1,), minval=0, maxval=1)
    # return jnp.concatenate([mean, scaling, rotation, colour, opacity, objectness])

    return jnp.concatenate([mean, scaling, rotation, colour])

def init_gaussians(key, img_shape, N: int) -> jnp.ndarray:
    """Returns the initial model params."""
    keys = jax.random.split(key, N)
    gaussians = [init_gaussian(keys[i], img_shape[0], img_shape[1]) for i in range(N)]
    return jnp.stack(gaussians, axis=0)


def get_gaussian_density(mean, scaling, rotation, x):
    """Calculate the density of the gaussian at a given point."""

    def make_rotation_matrix(angle):
        cos, sin = jnp.cos(angle), jnp.sin(angle)
        return jnp.array([[cos, -sin], [sin, cos]]).squeeze()

    def get_covariance(scaling, rotation_angle):
        """Calculate the covariance matrix. """
        scaling_matrix = jnp.diag(scaling)
        rotation_matrix = make_rotation_matrix(rotation_angle)

        covariance = rotation_matrix @ scaling_matrix @ scaling_matrix.T @ rotation_matrix.T 

        # jax.debug.print("Is positive semi-definite: {}", is_positive_semi_definite(covariance))

        # import jax.numpy as jnp
        # jax.debug.breakpoint()
        return covariance

    x_ = (x - mean)[:, None]

    ## Compute the inverse covariance matrix
    # den = jnp.exp(-0.5 * x_.T @ jnp.linalg.inv(get_covariance(scaling, rotation)) @ x_).squeeze()

    ## Compute a more stable inverse of the covariance matrix via linar solve
    # cov_mat = get_covariance(scaling, rotation)
    cov_mat = get_covariance(scaling*32, rotation * 2*jnp.pi)       ## TODO: 32 is stil hardcoded

    # den = jnp.exp(-0.5 * x_.T @ jnp.linalg.solve(cov_mat, x_)).squeeze()

    # jax.debug.print("Is cov mat: {}\n", cov_mat)
    # sol = lx.linear_solve(lx.MatrixLinearOperator(cov_mat), x_.squeeze(), solver=lx.LU())
    # den = jnp.exp(-0.5 * x_.T @ sol.value).squeeze()

    # den = jnp.exp(-0.5 * x_.T @ jax.scipy.linalg.solve(cov_mat, x_, assume_a="pos")).squeeze()
    # den = jnp.exp(-0.5 * x_.T @ jnp.linalg.pinv(cov_mat, hermitian=True, rcond=1e-4) @ x_).squeeze()
    den = jnp.exp(-0.5 * x_.T @ jnp.linalg.pinv(cov_mat, hermitian=True) @ x_).squeeze()

    # ## Compute the invrse of this 2x2 matrix after computing the determinant
    # det = cov_mat[0, 0] * cov_mat[1, 1] - cov_mat[0, 1] * cov_mat[1, 0]
    # # print("Determinant: ", det)
    # jax.debug.print("Determinant: ", det)
    # inv_cov_mat = jnp.array([[cov_mat[1, 1], -cov_mat[0, 1]], [-cov_mat[1, 0], cov_mat[0, 0]]]) / (det * 32**2)
    # # inv_cov_mat = jnp.linalg.inv(cov_mat) / (32**2)

    # den = jnp.exp(-0.5 * x_.T @ inv_cov_mat @ x_).squeeze()
    # # jax.debug.print("Is positive semi-definite:\n {}\n {}\n {}", cov_mat, inv_cov_mat, det)

    return den
    # return jnp.nan_to_num(den, nan=0.0, posinf=0.0, neginf=0.0)

def render_pixel(gaussians: jnp.ndarray, x: jnp.ndarray):
    """Render a single pixel coordinates x from multiple gaussians. """

    ## First, clip all values in the gaussians between 0 and 1
    # gaussians = jnp.clip(gaussians, 0., 1.)

    means = gaussians[:, :2]
    scalings = gaussians[:, 2:4]
    rotations = gaussians[:, 4:5]
    colours = gaussians[:, 5:8]
    # opacities = gaussians[:, 8:9]

    densities = jax.vmap(get_gaussian_density, in_axes=(0, 0, 0, None))(means, scalings, rotations, x)[:, None]
    # densities = jnp.nan_to_num(densities, nan=0.0, posinf=0.0, neginf=0.0)

    # return jnp.clip(jnp.sum(densities * colours, axis=0), 0., 1.)
    # return jnp.clip(jnp.mean(densities*colours, axis=0), 0., 1.)
    return jnp.mean(densities*colours, axis=0)


def render_image(gaussians: jnp.ndarray, img_shape: jnp.ndarray):
    """
    Render a complete image.
    """

    render_pixels_1D = jax.vmap(render_pixel, in_axes=(None, 0), out_axes=0)
    render_pixels_2D = jax.vmap(render_pixels_1D, in_axes=(None, 1), out_axes=1)

    # meshgrid = jnp.meshgrid(jnp.arange(0, img_shape[0]), jnp.arange(0, img_shape[1]))
    meshgrid = jnp.meshgrid(jnp.linspace(0, 1, img_shape[0]), jnp.linspace(0, 1, img_shape[1]))
    pixels = jnp.stack(meshgrid, axis=0).T

    image = render_pixels_2D(gaussians, pixels)

    # return jnp.nan_to_num(image.squeeze(), nan=0.0, posinf=0.0, neginf=0.0)
    return image.squeeze()











def interpolate_2D_image(known_points, known_pixels, img_size=(32, 32), method="linear"):
    """ Interpolate a 2D image from known points and pixels using griddata. """

    # Step 1: Prepare coordinates for interpolation
    grid_y, grid_x = np.meshgrid(np.linspace(0, 1, img_size[0]), np.linspace(0, 1, img_size[1]), indexing='ij')

    # Step 2: Perform interpolation for each RGB channel separately using griddata
    # Create an empty array to store interpolated RGB values
    interpolated_image = np.zeros(img_size)

    # Step 3. Interpolate for each channel separately
    for channel in range(3):
        # Use griddata to interpolate the known values over the grid
        interpolated_channel = griddata(known_points, known_pixels[:, channel], (grid_y, grid_x), method=method)
        interpolated_image[:, :, channel] = interpolated_channel

    return np.clip(interpolated_image, 0, 1)



# def make_image(xy_coords, rgb_pixels, img_size=(32, 32)):
#     img = np.zeros(img_size)
#     x_coords = (xy_coords[:, 0] * img_size[0]).astype(int)
#     y_coords = (xy_coords[:, 1] * img_size[1]).astype(int)
#     img[x_coords, y_coords, :] = np.clip(rgb_pixels, 0., 1.)
#     return img

def make_image(xy_coords, rgb_pixels, img_size=(32, 32, 3)):
    img = jnp.zeros(img_size)
    x_coords = (xy_coords[:, 0] * img_size[0]).astype(int)
    y_coords = (xy_coords[:, 1] * img_size[1]).astype(int)
    img = img.at[x_coords, y_coords, :].set(jnp.clip(rgb_pixels, 0., 1.))
    return img




def make_run_folder(parent_path='./runs/'):
    """ Create a new folder for the run. """
    if not os.path.exists(parent_path):
        os.mkdir(parent_path)

    # run_folder = parent_path+time.strftime("%y%m%d-%H%M%S")+'/'
    run_folder = os.path.join(parent_path, time.strftime("%y%m%d-%H%M%S")+'/')
    if not os.path.exists(run_folder):
        os.mkdir(run_folder)
        print("Created a new run folder at:", run_folder)

    return run_folder


def setup_run_folder(folder_path, script_name, datagen_folder=None):
    """ Copy the run script, the module files, and create a folder for the adaptation results. """

    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
        print("Created a new run folder at:", folder_path)

    # Save the run scripts in that folder
    os.system(f"cp {script_name} {folder_path}")

    # Save the selfmod module files as well
    # module_folder = os.path.join(os.path.dirname(__file__), "../")
    module_folder = os.path.join(os.path.dirname(__file__))
    os.system(f"cp -r {module_folder} {folder_path}")
    print(" Backed up run script and module files ")

    ## Create a folder for the adaptation results
    adapt_folder = folder_path+"adapt/"
    if not os.path.exists(adapt_folder):
        os.mkdir(adapt_folder)
        print(" Created an adaptation folder at:", adapt_folder)

    ## Create a folder for the chckpoints results
    checkpoints_folder = folder_path+"checkpoints/"
    if not os.path.exists(checkpoints_folder):
        os.mkdir(checkpoints_folder)
        print(" Created a checkpoints folder at:", checkpoints_folder)

    ## Create a folder for the generation data scripts
    if datagen_folder is not None:
        data_folder = os.path.join(folder_path, "data/")
        if not os.path.exists(data_folder):
            os.mkdir(data_folder)
            print(" Created a data folder at:", data_folder)
        try:
            os.system(f"cp -r {datagen_folder}/ {data_folder}")
            print(" Attempting copy of the data generation scripts")
        except Exception as e:
            print(f" Could not copy the data generation scripts due to: {e}")
    else:
        data_folder = None

    return adapt_folder, checkpoints_folder, data_folder


def xavier_uniform(key, shape):
    lim = 1 / np.sqrt(shape[0])
    return jax.random.uniform(key, shape, minval=-lim, maxval=lim)

def xavier_uniform_transposed(key, shape):
    lim = 1 / np.sqrt(shape[-1])
    return jax.random.uniform(key, shape, minval=-lim, maxval=lim)
