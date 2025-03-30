#%%[markdown]
# Neural ODE on Lotka-Volterra Dataset

#%%
# %load_ext autoreload
# %autoreload 2

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from selfmod import *

from matplotlib import animation

# import jax
# jax.config.update("jax_debug_nans", True)

#%%

## For reproducibility
seed = 202402
np.random.seed(seed)
torch.manual_seed(seed)

## Dataloader hps
nb_families = 3          ## Total number of ODE families in the dataset
nb_experts = nb_families
nb_envs_per_fam = (3, 1)

num_envs = (9, 4)
num_shots = (-1, -1)
num_workers = 0
shuffle = False
train_proportion = 1.0  ## Minimal proportion of the trajectory for training
test_proportion = 1.0

## Learner/model hps
context_pool_size = 1
context_size = 6
taylor_orders = (0, 0)
ivp_args = {"return_traj":True, "max_steps":256*16, "integrator":diffrax.Tsit5(), "rtol": 1e-3, "atol":1e-6, "clip_sol":None, "adjoint": diffrax.RecursiveCheckpointAdjoint()}
# ivp_args = {"subdivisions":5, "integrator":RK4}
skip_steps = 1
loss_contributors = 3
max_ret_env_states = num_envs[0]
split_context = True
shift_context = True

meta_learner="GEPS"
data_size = 2
width_main = 256
depth_main = 4
depth_data = 2
depth_ctx = 2
intermediate_size = 32
activation = "swish"

## Train and adapt hps
init_lrs = (1e-3, 1e-3)
sched_factor = 0.4
max_train_batches = 1
max_adapt_batches = 1
proximal_betas = (10., 10.)       ## For the model, context and the gate, in that order

nb_outer_steps = 5000
nb_inner_steps = (12, 12)
nb_adapt_epochs = 5000
validate_every = 10
print_error_every = (10, 10)

gate_update_strategy = "least_squares"   ## "least_squares" or "gradient_descent"
gate_update_every = 5                       ## Update the gate every x inner steps (useful in least_squares mode)
context_regularization = True               ## Regularize the context with an L1 penalty

meta_train = True
meta_test = True

run_folder = None if meta_train else "./"
# run_folder = "./runs/241213-150028-Test/" if meta_train else "./"

data_folder = "./data/" if meta_train else "../../data/"


#%%

if run_folder==None:
    run_folder = make_run_folder('./runs/')
else:
    print("Using existing run folder:", run_folder)

adapt_folder, checkpoints_folder, _ = setup_run_folder(folder_path=run_folder, 
                                                        script_name=os.path.basename(__file__))

#%%[markdown]
# ## Meta-training




#%%

## Define 4 keys for dataloader(s), learner(s), trainer(s) and visualtester(s)
mother_key = jax.random.PRNGKey(seed)
data_key, model_key, trainer_key, test_key = jax.random.split(mother_key, num=4)

train_dataloader = NumpyLoader(MixerDynamicsDataset(data_dir=data_folder+"train_data.npz", 
                                               num_shots=num_shots[0], 
                                               skip_steps=skip_steps, 
                                               adaptation=False),
                              batch_size=num_envs[0],
                              shuffle=shuffle,
                              num_workers=num_workers,
                              drop_last=False)

val_dataloader = NumpyLoader(MixerDynamicsDataset(data_dir=data_folder+"test_data.npz", 
                                             num_shots=num_shots[1], 
                                             skip_steps=skip_steps,
                                             adaptation=False),
                              batch_size=num_envs[0],
                              shuffle=shuffle,
                              num_workers=num_workers,
                              drop_last=False)

#%%

# ## Data exploration - plot all trajectories in the first environments
(ins, ts), outs = next(iter(train_dataloader))
plt_data = outs
plt_t = ts

print("Typical shapes of data and t_eval:", plt_data.shape, plt_t.shape)

E_plot = 9
E_ = 1

y_lim = (plt_data.min(), plt_data.max())
fig, ax = plt.subplots(3, E_plot//3, figsize=(6*E_plot//3, 3*3))
ax = ax.flatten() if E_plot>1 else [ax]
colors = ['indigo']
for e in range(E_plot):
    e_plot_data_0 = plt_data[e*E_:(e+1)*E_, 0:1, :, 0]
    e_plot_data_1 = plt_data[e*E_:(e+1)*E_, 0:1, :, 1]
    e_t_eval = plt_t[e*E_:(e+1)*E_]
    for e_ in range(E_):
        ax[e].plot(e_t_eval[e_], e_plot_data_0[e_].T, '-', color=colors[e_], markersize=5, lw=2)
        ax[e].plot(e_t_eval[e_], e_plot_data_1[e_].T, '-', color=colors[e_], markersize=5, alpha=0.5, lw=3)
    ax[e].set_title(f"Env {e+1}", fontsize=16)
    if e==E_plot-1:
        ax[e].set_xlabel("Time $t$")
    ax[e].set_ylabel(f"$x$")
    ax[e].set_ylim(y_lim)

plt.tight_layout()
plt.draw()
plt.savefig(run_folder+"train_trajectories.png")
# plt.savefig(run_folder+"train_trajectories.pdf", bbox_inches='tight', dpi=100)


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
                pool_filling="NF",      ## The closest envs are used in the inner loss
                loss_filling="NF",      ## The closest envs are used in the outer loss
                self_reweighting=True,  ## Reweight the outer loss by its own softmax 
                key=model_key)


model_params = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)) if x is not None)
active_model_params = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(neuralnet.experts[0], eqx.is_array)) if x is not None)
print("\n\nTotal number of parameters in the model:", model_params)
print("Total number of parameters in one expert:", active_model_params)
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
                                envs=jnp.arange(0, nb_envs_per_fam[0]*nb_families).tolist(),
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

print("Adaptation contexts", trainer.learner.contexts_latest.params)

perp = nb_families if nb_families > 1 else 3
visualtester.visualize_context_clusters(perplexities=(perp, perp),
                                        key=test_key,
                                        save_path=run_folder+"context_clusters.png")

#%%
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
## Adapt the model to the new dataset
if meta_test:

    ## Adapt in a sequential manner
    adapt_losses = []
    
    for adapt_id in range(num_envs[1]):

        adapt_dataset = MixerDynamicsDataset(data_dir=data_folder+"adapt_train.npz", 
                                                    num_shots=num_shots[1], 
                                                    skip_steps=skip_steps,
                                                    adaptation=True)
        adapt_dataset.total_envs = 1
        adapt_dataset.dataset = adapt_dataset.dataset[adapt_id:adapt_id+1, :, :, :]
        # adapt_dataset.dataset = adapt_dataset.t_eval[adapt_id:, :, :]

        adapt_dataloader = NumpyLoader(dataset=adapt_dataset,
                                    batch_size=1, 
                                    shuffle=shuffle,
                                    num_workers=num_workers,
                                    drop_last=False)

        adapt_dataset_test = MixerDynamicsDataset(data_dir=data_folder+"adapt_test.npz", 
                                                    num_shots=num_shots[1], 
                                                    skip_steps=skip_steps,
                                                    adaptation=True)
        adapt_dataset_test.total_envs = 1
        adapt_dataset_test.dataset = adapt_dataset_test.dataset[adapt_id:adapt_id+1, :, :, :]

        adapt_dataloader_test = NumpyLoader(dataset=adapt_dataset_test,
                                    batch_size=1,
                                    shuffle=shuffle,
                                    num_workers=num_workers,
                                    drop_last=False)

        ood_crit, all_ood_crit = visualtester.evaluate(adapt_dataloader, 
                                            taylor_order=taylor_orders[1], 
                                            nb_steps=nb_adapt_epochs,
                                            print_error_every=print_error_every, 
                                            criterion_id=0,
                                            verbose=True,
                                            val_dataloader=adapt_dataloader_test,
                                            max_ret_env_states=1,
                                            max_adapt_batches=max_adapt_batches,
                                            stochastic=False)

        print("Loss per OoD environment:", all_ood_crit[0].tolist())

        adapt_losses.append(ood_crit)

## Print the mean loss over all environments
print("\nMean OoD loss over all environments:", np.mean(adapt_losses))

#%%
visualtester.visualize_artefacts(save_path=adapt_folder+"artefacts.png", adaptation=True)

visualtester.visualize_dynamics(save_path=adapt_folder+"dynamics.png",
                                data_loader=adapt_dataloader_test,
                                envs=[0],
                                traj=0,
                                share_axes=False,
                                key=test_key)

#%%

perp = nb_families if nb_families > 1 else 3
visualtester.visualize_context_clusters(perplexities=(perp, perp),
                                        key=test_key,
                                        save_path=adapt_folder+"context_clusters.png")


#%%
## After training, copy nohup.log to the runfolder
try:
    __IPYTHON__ ## in a jupyter notebook
except NameError:
    os.system(f"cp nohup.log {run_folder}")

#%%

