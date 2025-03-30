#%%[markdown]
# Hierarchical Shallow Piece-Wise Recurrent Neural Netwoek on Synthetic Control Data

#%%
# %load_ext autoreload
# %autoreload 2

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# os.environ["JAX_PLATFORMS"] = 'cpu'

from selfmod import *
# jax.config.update('jax_platform_name', 'cpu')

from matplotlib import animation
# import jax
# jax.config.update("jax_debug_nans", True)


#%%

## For reproducibility
seed = 20248
np.random.seed(seed)
torch.manual_seed(seed)

## Dataloader hps
nb_families = 3
nb_experts = nb_families
nb_envs_per_fam = (600//nb_families, 600//nb_families)

num_envs = (nb_envs_per_fam[0]*nb_families, 600)
num_shots = (-1, -1)
num_workers = 24
shuffle = False
train_proportion = 1.0  ## Min proporrion of the trajectory for training
test_proportion = 1.0

## Learner/model hps
context_pool_size = 20
context_size = 10
taylor_orders = (2, 0)
skip_steps = 1
loss_contributors = 600
max_ret_env_states = num_envs[0]
split_context = False
shift_context = True
self_reweighting = False

meta_learner = "hier-shPLRNN"
data_size = 1
hidden_size = 16*2
latent_size = data_size
tf_alpha_min = 0.0  ## Teacher forcing alpha (1. means no teacher forcing)

## Train and adapt hps
init_lrs = (1e-3, 1e-3)
sched_factor = 0.4
max_train_batches = 1
max_adapt_batches = 1
proximal_betas = (10., 10.)       ## For the model, context and the gate, in that order

nb_outer_steps = 500
nb_inner_steps = (12, 12)
nb_adapt_epochs = 100
validate_every = 10
print_error_every = (10, 10)

gate_update_strategy = "least_squares"      ## "least_squares" or "gradient_descent"
gate_update_every = 1                       ## Update the gate every x inner steps (useful in least_squares mode)
context_regularization = False               ## Regularize the context with an L1 penalty
same_expert_optstate = False                  ## Use the same optstate for all experts

meta_train = True
meta_test = False

run_folder = None if meta_train else "./"
# run_folder = "./runs/250110-115030-Test/" if meta_train else "./"

data_folder = "./data/" if meta_train else "../../data/"


#%%

if run_folder==None:
    run_folder = make_run_folder('./runs/')
else:
    print("Using existing run folder:", run_folder)

adapt_folder, checkpoints_folder, _ = setup_run_folder(run_folder, os.path.basename(__file__))



#%%[markdown]
# ## Meta-training






#%%

## Read the file line by line
time_series = []
with open(data_folder+"synthetic_control.data", 'r') as f:
    for line in f:
        time_series.append(list(map(float, line.split())))

print("Number of time series:", len(time_series))
print("Time series 0", time_series[0])
time_series = np.array(time_series)

## Normalise the time series (as the Dataset below will do as well)
time_series = (time_series - np.mean(time_series, axis=0)) / np.std(time_series, axis=0)

## Plot 6 randomly chosen time series
fig, ax = plt.subplots(2, 3, figsize=(6*3, 6))
ax = ax.flatten()

## Set the samme y limits for all plots
ylim = np.min(time_series), np.max(time_series)

np.random.seed(0)
for i in range(6):
    ts_id = np.random.randint(0, len(time_series))
    ax[i].plot(time_series[ts_id])
    ax[i].set_title(f"Time Series {ts_id}")
    ax[i].set_ylim(ylim)

plt.tight_layout()
plt.savefig(run_folder+"train_trajectories.png")

print("Time series shape:", type(time_series[0]), time_series.dtype, time_series.shape)

#%%

## Define 4 keys for dataloader(s), learner(s), trainer(s) and visualtester(s)
mother_key = jax.random.PRNGKey(seed)
data_key, model_key, trainer_key, test_key = jax.random.split(mother_key, num=4)

train_dataloader = NumpyLoader(TrendsDataset(data_dir=data_folder, 
                                               skip_steps=skip_steps, 
                                               traj_prop_min=train_proportion,
                                               use_full_traj=True), 
                              batch_size=num_envs[0],
                              shuffle=shuffle,
                              num_workers=num_workers,
                              drop_last=False)

val_dataloader = NumpyLoader(TrendsDataset(data_dir=data_folder, 
                                             skip_steps=skip_steps,
                                             traj_prop_min=test_proportion,
                                             use_full_traj=True),
                              batch_size=num_envs[0],
                              shuffle=shuffle,
                              num_workers=num_workers,
                              drop_last=False)



#%%



def env_loss_fn(model, ctx, y_hat, y):
    """
    Loss function for one environment. Leading dimension of y_hat corresponds to the pool size !
    """

    term1 = jnp.mean((y_hat-y)**2)
    if context_regularization:
        term2 = jnp.mean(jnp.abs(ctx))
        loss_val = term1 + 1e-3*term2
    else:
        term2 = 0.
        loss_val = term1

    # term3 = params_norm_squared(model)
    # loss_val = term1

    return loss_val, (term1, term2, 0.)

## Example context to use
contexts = ArrayContextParams(nb_envs=num_envs[0], context_size=context_size, key=None)

## Parameters for a built-in RNN model
expert_params = {"data_size":data_size,
                "hidden_size":hidden_size,
                "latent_size":latent_size,
                "context_size":context_size,
                "ctx_utils":None,
                "tf_alpha_min":tf_alpha_min,
                "shift_context":shift_context}

model = DirectMapping(MixER_S2S(key=model_key,
                                nb_experts=nb_experts,
                                meta_learner=meta_learner,
                                split_context=split_context,
                                same_expert_init=True,
                                use_gate_bias=True,
                                gate_update_strategy=gate_update_strategy,
                                **expert_params),
                    taylor_order=taylor_orders[0])

learner = Learner(model=model,
                context_size=contexts.eff_context_size, 
                context_pool_size=context_pool_size,
                env_loss_fn=env_loss_fn, 
                contexts=contexts,
                reuse_contexts=True,
                loss_contributors=loss_contributors,
                pool_filling="NF",      ## The closest envs are used in the inner loss
                loss_filling="NF",      ## The closest envs are used in the outer loss
                self_reweighting=self_reweighting,  ## Reweight the outer loss by its own softmax 
                key=model_key)

model_params = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)) if x is not None)
print("\n\nTotal number of parameters in the model:", model_params)
print("Total number of parameters in one context:", contexts.eff_context_size)


#%%

## Define optimiser and train the model
init_lr_model, init_lr_ctx = init_lrs

total_steps = nb_outer_steps*nb_inner_steps[0]
bd_scales = {total_steps//3:sched_factor, 2*total_steps//3:sched_factor}
sched_model = optax.piecewise_constant_schedule(init_value=init_lr_model, boundaries_and_scales=bd_scales)
sched_ctx = optax.piecewise_constant_schedule(init_value=init_lr_ctx, boundaries_and_scales=bd_scales)
opt_model = optax.adabelief(sched_model)
opt_ctx = optax.adabelief(init_lr_ctx)

trainer = NCFTrainer(learner, (opt_model, opt_ctx), key=trainer_key)

#%%

## Meta-training
if meta_train == True:
    trainer.meta_train_gated(dataloader=train_dataloader, 
                        nb_epochs=1, 
                        nb_outer_steps=nb_outer_steps, 
                        nb_inner_steps=nb_inner_steps, 
                        proximal_betas=proximal_betas, 
                        max_train_batches=max_train_batches, 
                        print_error_every=print_error_every, 
                        save_checkpoints=True, 
                        validate_every=validate_every, 
                        save_path=run_folder, 
                        val_dataloader=val_dataloader, 
                        val_nb_steps=nb_adapt_epochs,
                        val_criterion_id=0, 
                        max_val_batches=max_train_batches,
                        update_gate_every=gate_update_every,
                        same_expert_optstate=same_expert_optstate,
                        verbose=False,
                        key=trainer_key)
else:
    print("Skipping meta-training ...")
    restore_folder = run_folder
    trainer.restore_trainer(path=run_folder)



#%%[markdown]
# ## Post-training analysis




#%%
## Test and visualise the results on a test dataloader
visualtester = DynamicsVisualTester(trainer, key=test_key)

ind_crit, all_ind_crit = visualtester.evaluate(train_dataloader, 
                                    taylor_order=taylor_orders[1], 
                                    nb_steps=nb_adapt_epochs,
                                    print_error_every=print_error_every, 
                                    criterion_id=0,
                                    verbose=True,
                                    val_dataloader=val_dataloader,
                                    max_ret_env_states=max_ret_env_states,
                                    max_adapt_batches=max_adapt_batches,
                                    stochastic=False)

visualtester.visualize_artefacts(save_path=run_folder+"artefacts.png", ylim=None)
print("Loss per InD environment:", all_ind_crit[0].tolist())

ctx_shifts = [learner.model.vectorfield.neuralnet.experts[e].ctx_shift.squeeze() for e in range(nb_experts)]
print("After training, the context shifts are:", jnp.array(ctx_shifts))

#%%
visualtester.visualize_dynamics(save_path=run_folder+"dynamics.png",
                                data_loader=val_dataloader,
                                envs=jnp.arange(0, nb_envs_per_fam[0]*nb_families, 100).tolist(),
                                dims=(0,0),
                                traj=0,
                                share_axes=False,
                                key=test_key)


#%%
## Inspect the context, and evalualte the gate layer
contexts = learner.contexts
network = trainer.learner.model.vectorfield.neuralnet

gate_vals = eqx.filter_vmap(network.gating_function)(contexts.params)

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7*2, 6))
ax.hist(gate_vals.flatten(), bins=50);
ax.set_title("Gate Values Histogram")

## inshow on ax2
img = ax2.imshow(gate_vals, aspect='auto', cmap='turbo', interpolation=None)
plt.colorbar(img)
ax2.set_xlabel("Experts")
ax2.set_ylabel("Environments")

## Set yticks in steps of nb_envs_per_fam[0]
y_labels = np.arange(0, nb_envs_per_fam[0]*nb_families, nb_envs_per_fam[0])
ax2.set_yticks(y_labels)
ax2.set_yticklabels(y_labels)

x_labels = np.arange(0, nb_experts, 1)
ax2.set_xticks(x_labels)
ax2.set_xticklabels(x_labels)

ax2.set_title("Gate Values Heatmap")

plt.draw()
plt.savefig(run_folder+"gate_values.png")



#%%

@eqx.filter_vmap(in_axes=(None, 0))
def gate_anim_fn(network, ctx):
    return network.gating_function(ctx)

## We want to do an animation of how the gate values change over time
all_gate_vals = []
for outer_step in list(range(0, nb_outer_steps, print_error_every[0]))+[nb_outer_steps-1]:
    contexts_ = eqx.tree_deserialise_leaves(checkpoints_folder+f"contexts_outstep_{outer_step:06d}.eqx", learner.contexts)
    network_ = eqx.tree_deserialise_leaves(checkpoints_folder+f"model_outstep_{outer_step:06d}.eqx", learner.model).vectorfield.neuralnet

    all_gate_vals.append(gate_anim_fn(network_, contexts_.params))

all_gate_vals = jnp.stack(all_gate_vals, axis=0)

## Plot the gate values as an animation
fig, ax = plt.subplots(1, 1, figsize=(6, 7))
img = ax.imshow(all_gate_vals[0], aspect='auto', cmap='turbo', interpolation="nearest")
plt.colorbar(img)
ax.set_xlabel("Experts")
ax.set_ylabel("Environments")

ax.set_title(f"Outer Step {0}")

ax.set_yticks(y_labels)
ax.set_yticklabels(y_labels)

ax.set_xticks(x_labels)

def animate(i):
    img.set_data(all_gate_vals[i])
    ax.set_title(f"Outer Step {i*print_error_every[0]}")
    return img,

ani = animation.FuncAnimation(fig, animate, frames=len(all_gate_vals), interval=100, blit=True)
ani.save(run_folder+"gate_values_animation.gif", writer='pillow', fps=20)


#%%

perp = nb_families if nb_families > 1 else 4
visualtester.visualize_context_clusters(perplexities=(perp, perp),
                                        key=test_key,
                                        save_path=run_folder+"context_clusters.png")

#%%
X = learner.contexts.params
# 1-100   Normal
# 101-200 Cyclic
# 201-300 Increasing trend
# 301-400 Decreasing trend
# 401-500 Upward shift
# 501-600 Downward shift

## We have 600 samples and 6 classes as above. Create the labels
labels = np.zeros((600,), dtype=int)
labels[100:200] = 1 
labels[200:300] = 2
labels[300:400] = 3
labels[400:500] = 4
labels[500:600] = 5

color_table = {0:"royalblue", 1:"crimson", 2:"forestgreen", 3:"darkorange", 4:"purple", 5:"black"}
colors = [color_table[l] for l in labels]

conditions = {0:"Normal", 1:"Cyclic", 2:"Increasing trend", 3:"Decreasing trend", 4:"Upward shift", 5:"Downward shift"}

## Use PCA instead
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
# X_reduced = X

# Plotting
plt.figure(figsize=(10, 7))
# plt.scatter(X_reduced[:, 0], X_reduced[:, 1], s=50, c=colors)

markers = {0:'o', 1:'x', 2:'^', 3:'s', 4:'D', 5:'P'}
for class_label in range(6):
    marker = markers[class_label]
    plt.scatter(X_reduced[labels==class_label, 0], X_reduced[labels==class_label, 1], s=50, c=color_table[class_label], label=conditions[class_label], marker=marker)

plt.legend()

plt.title("Contexts Clustering - PCA", fontsize=24)
plt.xlabel("PC 1")
plt.ylabel("PC 2")

plt.draw()
plt.savefig(run_folder+"clusters_pca.png", bbox_inches='tight');




#%%

## Use Umap instead
import umap
reducer = umap.UMAP(n_components=2)
X_reduced = reducer.fit_transform(X)

# Plotting
plt.figure(figsize=(10, 7))

markers = {0:'o', 1:'x', 2:'^', 3:'s', 4:'D', 5:'P'}
for class_label in range(6):
    marker = markers[class_label]
    plt.scatter(X_reduced[labels==class_label, 0], X_reduced[labels==class_label, 1], s=50, c=color_table[class_label], label=conditions[class_label], marker=marker)

plt.legend()

plt.title("Contexts Clustering - UMAP", fontsize=24)
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")

plt.draw()
plt.savefig(run_folder+"clusters_umap.png", bbox_inches='tight')







#%%
## After training, copy nohup.log to the runfolder
try:
    __IPYTHON__ ## in a jupyter notebook
except NameError:
    os.system(f"cp nohup.log {run_folder}")

#%%
