#%%[markdown]
# Implementation of MixER on the ODEBench dataset

#%%

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from mixer import *           ## Import the necessary modules
from matplotlib import animation

#%%

## For reproducibility
seed = 20960
np.random.seed(seed)
torch.manual_seed(seed)

## Dataloader hps
nb_families = 10          ## Total number of ODE families in the dataset (useful for visualisation)
nb_experts = 10           ## Total number of experts in the MixER layer
nb_envs_per_fam = (16, 4)

num_envs = (nb_envs_per_fam[0]*nb_families, nb_envs_per_fam[1]*nb_families)
num_shots = (-1, -1)
num_workers = 8
shuffle = False
train_proportion = 0.6      ## Minimal proportion of the trajectory for training
test_proportion = 1.0
skip_steps = 5
normalize_data = False

## Learner/model hps
context_pool_size = 1                 ## Number of context to keep in the pool for contextual self-modulation (if needed)
taylor_orders = (0, 0)
context_size = 4*nb_experts         ## Size (dimension) of the context vector
ivp_args = {"return_traj":True, "max_steps":256*16, "integrator":diffrax.Tsit5(), "rtol": 1e-3, "atol":1e-6, "clip_sol":None, "adjoint": diffrax.RecursiveCheckpointAdjoint()}
loss_contributors = nb_envs_per_fam[0]*1
max_ret_env_states = num_envs[0]
split_context = True
shift_context = True
self_reweighting = False
same_expert_optstate = False                  ## Use the same optimizer state for all experts ?

meta_learner = "GEPS"                         ## The meta-learner "NCF", "CoDA", "GEPS", or "hier-shPLRNN"
data_size = 2
width_main = 32*2                             ## Width of the main network
depth_main = 3
depth_data = 1
depth_ctx = 1
intermediate_size = 32
activation = "swish"

## Train and adapt hps
init_lrs = (1e-3, 1e-3)
sched_factor = 0.4
max_train_batches = 1
max_adapt_batches = 1
proximal_betas = (10., 10.)       ## For the model, context and the gate, in that order

nb_outer_steps = 250*6
nb_inner_steps = (12, 12)
nb_adapt_epochs = 250*6
validate_every = 1*10
print_error_every = (1*10, 1*10)

gate_update_strategy = "least_squares"      ## "least_squares" or "gradient_descent" to update the gating network
gate_update_every = 1                       ## Update the gate every x inner steps (useful in least_squares mode)
context_regularization = False              ## Regularize the context with an L1 penalty

meta_train = True
meta_test = True

run_folder = None if meta_train else "./"
data_folder = "./data_2D/" if meta_train else "../../../data_2D/"


#%%

if run_folder==None:
    if meta_learner=="NCF":
        base_folder = "./runs/_02_NCFRuns/"
    elif meta_learner=="CoDA":
        base_folder = "./runs/_03_CoDARuns/"
    elif meta_learner=="GEPS":
        base_folder = "./runs/_04_GEPSRuns/"
    else:
        base_folder = "./runs/"
    run_folder = make_run_folder(base_folder)
else:
    print("Using existing run folder:", run_folder)

adapt_folder, checkpoints_folder, _ = setup_run_folder(folder_path=run_folder, 
                                                        script_name=os.path.basename(__file__), 
                                                        datagen_folder=f"{os.path.dirname(__file__)}/data_gen")

#%%[markdown]
# ## Meta-training




#%%

## Define 4 keys for dataloader(s), learner(s), trainer(s) and visualtester(s)
mother_key = jax.random.PRNGKey(seed)
data_key, model_key, trainer_key, test_key = jax.random.split(mother_key, num=4)

train_dataloader = NumpyLoader(ODEBenchDataset(data_dir=data_folder+"train.npz", 
                                               norm_consts=data_folder+"train_bounds.npy" if normalize_data else None,
                                               num_shots=num_shots[0], 
                                               skip_steps=skip_steps, 
                                               traj_prop_min=train_proportion), 
                              batch_size=num_envs[0],
                              shuffle=shuffle,
                              num_workers=num_workers,
                              drop_last=False)

val_dataloader = NumpyLoader(ODEBenchDataset(data_dir=data_folder+"test.npz", 
                                             norm_consts=data_folder+"train_bounds.npy" if normalize_data else None,
                                             num_shots=num_shots[1], 
                                             skip_steps=skip_steps,
                                             traj_prop_min=test_proportion),
                              batch_size=num_envs[0],
                              shuffle=shuffle,
                              num_workers=num_workers,
                              drop_last=False)

#%%

# ## Data exploration - plot all trajectories in a few environments
(ins, ts), outs = next(iter(train_dataloader))
plt_data = outs
plt_t = ts

print("Typical shapes of data and t_eval:", plt_data.shape, plt_t.shape)

E_plot = nb_families
E_ = nb_envs_per_fam[0]

# fig, ax = plt.subplots(2, E_plot//2, figsize=(6*E_plot//2, 3*2))
fig, ax = plt.subplots(E_plot, 1, figsize=(6, 3*E_plot))
ax = ax.flatten() if E_plot>1 else [ax]
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'r', 'g', 'b', 'c', 'm', 'y']
for e in range(E_plot):
    e_plot_data_0 = plt_data[e*E_:(e+1)*E_, 0:1, :, 0]
    e_plot_data_1 = plt_data[e*E_:(e+1)*E_, 0:1, :, 1]
    e_t_eval = plt_t[e*E_:(e+1)*E_]
    for e_ in range(E_):
        ax[e].plot(e_t_eval[e_], e_plot_data_0[e_].T, '-', color=colors[e_], markersize=5, lw=2)
        ax[e].plot(e_t_eval[e_], e_plot_data_1[e_].T, '-', color=colors[e_], markersize=5, alpha=0.5, lw=3)
    ax[e].set_title(f"Family {e+1}", fontsize=16)
    if e==E_plot-1:
        ax[e].set_xlabel("Time $t$")
    ax[e].set_ylabel(f"$x$")

plt.tight_layout()
plt.draw()
plt.savefig(run_folder+"train_trajectories.png")
lims = [ax[e].get_ylim() for e in range(E_plot)]


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

## Context to use
contexts = ArrayContextParams(nb_envs=num_envs[0], context_size=context_size, key=None)

## Parameters for a built-in NCF model
if meta_learner == "NCF":
    expert_params = {"data_size":data_size,
                    "width_main":width_main,
                    "depth_main":depth_main,
                    "depth_data":depth_data,
                    "depth_ctx":depth_ctx,
                    "context_size":context_size,
                    "intermediate_size":intermediate_size,
                    "ctx_utils":None,
                    "activation":"swish",
                    "shift_context":shift_context}
elif meta_learner in ["CoDA", "GEPS"]:
    expert_params = {"data_size":data_size,
                    "width":width_main,
                    "depth":depth_main,
                    "context_size":context_size,
                    "activation":"swish",
                    "shift_context":shift_context}
else:
    raise ValueError("Unknown meta-learner")

neuralnet = MixER(key=model_key,
                nb_experts=nb_experts,
                meta_learner=meta_learner,
                split_context=split_context,
                same_expert_init=True,
                use_gate_bias=True,
                gate_update_strategy=gate_update_strategy,
                **expert_params)

model = NeuralODE(neuralnet=neuralnet,
                taylor_order=taylor_orders[0],
                ivp_args=ivp_args,
                t_eval=None,
                taylor_ad_mode="forward")

learner = Learner(model=model,
                context_size=contexts.eff_context_size, 
                context_pool_size=context_pool_size,
                env_loss_fn=env_loss_fn, 
                contexts=contexts,
                reuse_contexts=True,
                loss_contributors=loss_contributors,
                pool_filling="NF",      ## The closest envs are used in the inner loss (for NCF)
                loss_filling="NF",      ## The closest envs are used in the outer loss (for all meta-learners)
                self_reweighting=self_reweighting,  ## Reweight the outer loss by its own softmaxed values
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
                        # val_dataloader=val_dataloader, 
                        val_nb_steps=nb_adapt_epochs,
                        val_criterion_id=0, 
                        max_val_batches=max_train_batches,
                        update_gate_every=gate_update_every,
                        same_expert_optstate=same_expert_optstate,
                        verbose=True,
                        key=trainer_key)
else:
    print("Skipping meta-training ...")
    restore_folder = run_folder
    trainer.restore_trainer(path=run_folder)


#%%[markdown]
# ## Post-training analysis



#%%
import copy
## Test and visualise the results on a test dataloader
visualtester = DynamicsVisualTester(trainer, key=test_key)

@eqx.filter_vmap(in_axes=(None, 0))
def gate_anim_fn(network, ctx):
    return network.gating_function(ctx)

def rel_l2(y, y_hat):
    clean_y = jnp.maximum(jnp.abs(y), 1e-6)
    return jnp.mean(jnp.linalg.norm(y_hat-y, axis=-1)**2/jnp.linalg.norm(clean_y, axis=-1)**2)

## We want to do an animation of how the gate values change over time
all_gate_vals = []
val_losses = []
val_steps = []
base_contexts = copy.deepcopy(learner.contexts)
base_model = copy.deepcopy(learner.model)

for outer_step in list(range(0, nb_outer_steps, print_error_every[0]))+[nb_outer_steps-1]:
    contexts_ = eqx.tree_deserialise_leaves(checkpoints_folder+f"contexts_outstep_{outer_step:06d}.eqx", base_contexts)
    network_ = eqx.tree_deserialise_leaves(checkpoints_folder+f"model_outstep_{outer_step:06d}.eqx", base_model)

    all_gate_vals.append(gate_anim_fn(network_.vectorfield.neuralnet, contexts_.params))

    visualtester.trainer.learner.contexts = contexts_
    visualtester.trainer.learner.model = network_
    ind_crit, all_ind_crit = visualtester.evaluate(val_dataloader, 
                                        taylor_order=taylor_orders[1], 
                                        nb_steps=nb_adapt_epochs,
                                        print_error_every=print_error_every, 
                                        criterion_id=0,
                                        loss_criterion=rel_l2,
                                        verbose=False,
                                        val_dataloader=val_dataloader,
                                        max_ret_env_states=max_ret_env_states,
                                        # max_ret_env_states=8,
                                        max_adapt_batches=max_adapt_batches,
                                        stochastic=False)

    val_losses.append(jnp.mean(all_ind_crit[0]))
    val_steps.append(outer_step)

all_gate_vals = jnp.stack(all_gate_vals, axis=0)

#%%
## Plot the gate values as an animation
fig, ax = plt.subplots(1, 1, figsize=(6, 7))

img = ax.imshow(all_gate_vals[0], aspect='auto', cmap='turbo', interpolation="kaiser")
plt.colorbar(img)
ax.set_xlabel("Experts")
ax.set_ylabel("Environments")

ax.set_title(f"Outer Step {0}")
y_labels = np.arange(0, nb_envs_per_fam[0]*nb_families, nb_envs_per_fam[0])
ax.set_yticks(y_labels)
ax.set_yticklabels(y_labels)

x_labels = np.arange(0, nb_experts, 1)
ax.set_xticks(x_labels)

def animate(i):
    img.set_data(all_gate_vals[i])
    ax.set_title(f"Outer Step {i*print_error_every[0]}")
    return img,

ani = animation.FuncAnimation(fig, animate, frames=len(all_gate_vals), interval=100, blit=True)
ani.save(run_folder+"gate_values_animation.gif", writer='pillow', fps=20)


#%%

## Plot the individual losses (as the training progresses)
plt.figure(figsize=(8, 4))
xs = list(range(0, nb_outer_steps, print_error_every[0])) + [nb_outer_steps-1]
plt.plot(xs, val_losses)
plt.xlabel("Outer Steps")
plt.ylabel("Rel L2 Loss")
plt.title("Relative validation MSE throughout training")
plt.yscale("log")
plt.draw()

## BEST MODEL found at the end training outer step
best_idx = val_steps[np.argmin(val_losses)]
print("=== Best model found at Outer Step:", best_idx)

## Load and restore the best model and the context
contexts = eqx.tree_deserialise_leaves(checkpoints_folder+f"contexts_outstep_{best_idx:06d}.eqx", base_contexts)
network = eqx.tree_deserialise_leaves(checkpoints_folder+f"model_outstep_{best_idx:06d}.eqx", base_model)
learner.contexts = contexts
learner.model = network



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
ax2.set_yticks(y_labels)
ax2.set_yticklabels(y_labels)

ax2.set_xticks(x_labels)
ax2.set_xticklabels(x_labels)

ax2.set_title("Gate Values Heatmap")

plt.draw()
plt.savefig(run_folder+"gate_values.png")



#%%

ind_crit, all_ind_crit = visualtester.evaluate(train_dataloader, 
                                    taylor_order=taylor_orders[1], 
                                    nb_steps=nb_adapt_epochs,
                                    print_error_every=print_error_every, 
                                    criterion_id=0,
                                    loss_criterion=rel_l2,
                                    verbose=True,
                                    val_dataloader=val_dataloader,
                                    max_ret_env_states=max_ret_env_states,
                                    # max_ret_env_states=8,
                                    max_adapt_batches=max_adapt_batches,
                                    stochastic=False)

visualtester.visualize_artefacts(save_path=run_folder+"artefacts.png", ylim=None)
print("Losses per InD environment:", all_ind_crit[0].tolist())

ctx_shifts = [learner.model.vectorfield.neuralnet.experts[e].ctx_shift.squeeze() for e in range(nb_experts)]
print("After training, the context shifts are:", jnp.array(ctx_shifts))

print(f"\nLoss InD (mean)  : {np.mean(all_ind_crit[0]):.3f}")

#%%
visualtester.visualize_dynamics(save_path=run_folder+"dynamics.png",
                                data_loader=val_dataloader,
                                dims=(0,1),
                                envs=jnp.arange(0, nb_envs_per_fam[0]*nb_families).tolist(),
                                traj=0,
                                share_axes=False,
                                key=test_key)


#%%

perp = nb_families if nb_families > 1 else 4
visualtester.visualize_context_clusters(perplexities=(perp, perp),
                                        key=test_key,
                                        save_path=run_folder+"context_clusters.png")

#%%

## UMAP visualisation of the context vectors
X = learner.contexts.params
labels = np.arange(nb_families).repeat(nb_envs_per_fam[0])
color_table = {0:"red", 1:"royalblue", 2:"green", 3:"orange", 4:"purple", 5:"brown", 6:"pink", 7:"gray", 8:"cyan", 9:"magenta"}
colors = [color_table[l] for l in labels]

import umap
umap_reducer = umap.UMAP(n_components=2, random_state=int(test_key[0]))

# Fit and transform the data
X_reduced = umap_reducer.fit_transform(X)

# Plotting
plt.figure(figsize=(10, 7))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], s=50, c=colors)
plt.title("Training Context Dimensionality Reduction", fontsize=24)
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")

# Adding annotations for each point
for i in range(0, X_reduced.shape[0], nb_envs_per_fam[0]):
    label = labels[i]
    # label = i
    plt.text(X_reduced[i, 0], X_reduced[i, 1]+5e-1, str(label), fontsize=16, ha='left', va='bottom', color='black', weight='bold')

plt.draw()
plt.savefig(run_folder+"clusters_umap.png", bbox_inches='tight')



#%%[markdown]
# ## Adaptation to a new dataset


















#%%
## Adapt the model to the new dataset (in one go, i.e. bulk adaptation)
if meta_test:
    adapt_dataset = ODEBenchDataset(data_dir=data_folder+"adapt_train.npz", 
                                    adaptation=True,
                                    norm_consts=data_folder+"adapt_train_bounds.npy" if normalize_data else None,
                                    num_shots=num_shots[0], 
                                    skip_steps=skip_steps,
                                    traj_prop_min=train_proportion)
    adapt_dataloader = NumpyLoader(dataset=adapt_dataset,
                                batch_size=num_envs[1], 
                                # batch_size=1, 
                                shuffle=shuffle,
                                num_workers=num_workers,
                                drop_last=False)

    adapt_dataset_test = ODEBenchDataset(data_dir=data_folder+"adapt_test.npz", 
                                            adaptation=True,
                                            norm_consts=data_folder+"adapt_train_bounds.npy" if normalize_data else None,
                                            num_shots=num_shots[0], 
                                            skip_steps=skip_steps,
                                            traj_prop_min=test_proportion)
    adapt_dataloader_test = NumpyLoader(dataset=adapt_dataset_test,
                                batch_size=num_envs[1],
                                shuffle=shuffle,
                                num_workers=num_workers,
                                drop_last=False)

    ood_crit, all_ood_crit = visualtester.evaluate(adapt_dataloader, 
                                        taylor_order=taylor_orders[1], 
                                        nb_steps=nb_adapt_epochs,
                                        print_error_every=print_error_every, 
                                        loss_criterion=rel_l2,
                                        criterion_id=0,
                                        verbose=False,
                                        val_dataloader=adapt_dataloader_test,
                                        max_ret_env_states=num_envs[1],
                                        max_adapt_batches=max_adapt_batches,
                                        stochastic=False)

    print("Loss per OoD environment:", all_ood_crit[0].tolist())
    print("Bulk Mean Adapt Loss:", np.mean(all_ood_crit[0]))
    bulk_contexts = learner.contexts_adapt.params

visualtester.visualize_artefacts(save_path=adapt_folder+"artefacts_bulk.png", adaptation=True)
visualtester.visualize_dynamics(save_path=adapt_folder+"dynamics_bulk.png",
                                data_loader=adapt_dataloader_test,
                                dims=(0,1),
                                envs=list(range(num_envs[1])),
                                traj=0,
                                share_axes=False,
                                key=test_key)



#%%
## Adapt the model to the new dataset (in a sequential manner)
if meta_test:
    sequential_losses = []
    sequential_contexts = []

    for adapt_id in range(num_envs[1]):

        adapt_dataset = ODEBenchDataset(data_dir=data_folder+"adapt_train.npz", 
                                        adaptation=True,
                                        norm_consts=data_folder+"adapt_train_bounds.npy" if normalize_data else None,
                                        num_shots=num_shots[0], 
                                        skip_steps=skip_steps,
                                        traj_prop_min=train_proportion)
        adapt_dataset.total_envs = 1
        adapt_dataset.dataset = adapt_dataset.dataset[adapt_id:adapt_id+1, :, :, :]
        adapt_dataset.t_eval = adapt_dataset.t_eval[adapt_id:adapt_id+1, :]

        adapt_dataloader = NumpyLoader(dataset=adapt_dataset,
                                    batch_size=1, 
                                    shuffle=shuffle,
                                    num_workers=num_workers,
                                    drop_last=False)

        adapt_dataset_test = ODEBenchDataset(data_dir=data_folder+"adapt_test.npz", 
                                                adaptation=True,
                                                norm_consts=data_folder+"adapt_train_bounds.npy" if normalize_data else None,
                                                num_shots=num_shots[0], 
                                                skip_steps=skip_steps,
                                                traj_prop_min=test_proportion)
        adapt_dataset_test.total_envs = 1
        adapt_dataset_test.dataset = adapt_dataset_test.dataset[adapt_id:adapt_id+1, :, :, :]
        adapt_dataset_test.t_eval = adapt_dataset_test.t_eval[adapt_id:adapt_id+1, :]

        adapt_dataloader_test = NumpyLoader(dataset=adapt_dataset_test,
                                    batch_size=1,
                                    shuffle=shuffle,
                                    num_workers=num_workers,
                                    drop_last=False)

        ood_crit, all_ood_crit = visualtester.evaluate(adapt_dataloader, 
                                            taylor_order=taylor_orders[1], 
                                            nb_steps=nb_adapt_epochs,
                                            print_error_every=print_error_every, 
                                            loss_criterion=rel_l2,
                                            criterion_id=0,
                                            verbose=False,
                                            val_dataloader=adapt_dataloader_test,
                                            max_ret_env_states=1,
                                            max_adapt_batches=max_adapt_batches,
                                            stochastic=False)

        print("Loss per OoD environment:", all_ood_crit[0].tolist())
        sequential_losses.append(np.mean(all_ood_crit[0]))
        sequential_contexts.append(learner.contexts_adapt.params)

        visualtester.visualize_dynamics(save_path=adapt_folder+f"dynamics_seq_{adapt_id}.png",
                                        data_loader=adapt_dataloader_test,
                                        dims=(0,1),
                                        envs=[0],
                                        traj=0,
                                        share_axes=False,
                                        key=test_key)

    visualtester.visualize_artefacts(save_path=adapt_folder+"artefacts.png", adaptation=True)

    print("Sequential Mean Adapt Loss:", np.mean(sequential_losses))
    seq_contexts = jnp.concatenate(sequential_contexts, axis=0)

    ## Save the bulk and sequential contexts
    np.savez(adapt_folder+"contexts_adaptation.npz", bulk=bulk_contexts, sequential=seq_contexts)

#%%

# perp = nb_families if nb_families > 1 else 4
# visualtester.visualize_context_clusters(perplexities=(perp, perp),
#                                         key=test_key,
#                                         save_path=adapt_folder+"context_clusters.png")


#%%
## Let's cluster the contexts from the two types of adaptation using PCA
X_bulk = bulk_contexts
X_seq = seq_contexts

print("Bulk Contexts Shape:", X_bulk.shape, "Sequential Contexts Shape:", X_seq.shape)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_bulk_reduced = pca.fit_transform(X_bulk)
X_seq_reduced = pca.transform(X_seq)

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.scatter(X_bulk_reduced[:, 0], X_bulk_reduced[:, 1], s=50, label="Bulk")

ax.scatter(X_seq_reduced[:, 0], X_seq_reduced[:, 1], s=50, label="Sequential")
ax.set_title("Bulk vs. Sequential Adaptation")

ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.legend()

plt.draw()
plt.savefig(adapt_folder+"clusters_pca.png", bbox_inches='tight')

