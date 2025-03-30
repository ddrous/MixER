#%%[markdown]
# Hierarchical Shallow Piece-Wise Recurrent Neural Network on Epilepsy Data

#%%
# %load_ext autoreload
# %autoreload 2

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# os.environ["JAX_PLATFORMS"] = 'cpu'

from selfmod import *
# jax.config.update('jax_platform_name', 'cpu')
# jax.config.update("jax_debug_nans", True)

from matplotlib import animation


#%%

## For reproducibility
seed = 200022
np.random.seed(seed)
torch.manual_seed(seed)

## Dataloader hps
nb_families = 2
nb_experts = 5

nb_envs_per_fam = ((64+20)//2, (64+20)//2)   ## (Expected)
num_envs = (nb_envs_per_fam[0]*nb_families, nb_envs_per_fam[1]*nb_families)
num_shots = (-1, -1)
num_workers = 24
shuffle = False

noisy_lorentz = True
num_train_steps = 1000
num_adapt_steps = 5500

## Learner/model hps
context_pool_size = 1
context_size = 3
taylor_orders = (0, 0)
skip_steps = 1
loss_contributors = nb_envs_per_fam[0] * nb_families
max_ret_env_states = num_envs[0]
split_context = False
shift_context = True

meta_learner = "hier-shPLRNN"
data_size = 10
hidden_size = 32
latent_size = data_size
same_expert_init = True
tf_alpha_start = 1.0  ## Teacher forcing alpha (0. means no teacher forcing)
tf_gamma = 0.9995      ## Teacher forcing gamma (alpha_min is multiplied by this value every outer steps)

## Train and adapt hps
init_lrs = (1e-3, 1e-3)
sched_factor = 1.0
max_train_batches = 1
max_adapt_batches = 1
proximal_betas = (10., 10.)       ## For the model, context and the gate, in that order

nb_outer_steps = 50*10
nb_inner_steps = (12, 12)
nb_adapt_epochs = 250
validate_every = 100
print_error_every = (100, 100)

gate_update_strategy = "least_squares"      ## "least_squares" or "gradient_descent"
gate_update_every = 1                       ## Update the gate every x inner steps (useful in least_squares mode)
gating_hyperparams = {"max_kmeans":30, "convergence_tol":1e-3, "noise_level":1e-2}

context_regularization = False               ## Regularize the context with an L1 penalty
same_expert_optstate = False                  ## Use the same optstate for all experts
self_reweighting = False                     ## Reweight the outer loss by its own softmax

meta_train = True
meta_test = False

run_folder = None if meta_train else "./"
# run_folder = "./runs/250103-123848-Test/" if meta_train else "./"

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

## Define 4 keys for dataloader(s), learner(s), trainer(s) and visualtester(s)
mother_key = jax.random.PRNGKey(seed)
data_key, model_key, trainer_key, test_key = jax.random.split(mother_key, num=4)

train_dataloader = NumpyLoader(LorentzDataset(data_dir=data_folder,
                                              noisy=noisy_lorentz,
                                               num_steps=num_train_steps,), 
                              batch_size=num_envs[0],
                              shuffle=shuffle,
                              num_workers=num_workers,
                              drop_last=False)

val_dataloader = NumpyLoader(LorentzDataset(data_dir=data_folder,
                                              noisy=noisy_lorentz,
                                               num_steps=num_train_steps),
                              batch_size=num_envs[0],
                              shuffle=shuffle,
                              num_workers=num_workers,
                              drop_last=False)

#%%

# ## Plot the trajectories in the a few environments

(outs, ts), _ = next(iter(train_dataloader))
def get_lorentz_labels():
    labs = np.zeros(outs.shape[0])
    labs[:64] = 1
    return labs.astype(int)
train_labels = get_lorentz_labels()

print("Shapes of data and t_eval:", outs.shape, ts.shape)

E_plot = 15
fig, ax = plt.subplots(E_plot//3, 3, figsize=(6*3, 3*5), sharex=True, sharey=True)
ax = ax.flatten()
if E_plot==1:
    ax = [ax]
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'r', 'g', 'b', 'c', 'm', 'y']
xlim = 0, 1
dim0, dim1 = 0, 1
for e, e_ in enumerate(np.random.choice(outs.shape[0], E_plot, replace=False)):
    # ax[e].plot(ts[e_], outs[e_].squeeze(), '-', color=colors[e])
    ax[e].plot(outs[e_, ..., dim0].squeeze(), outs[e_, ..., dim1].squeeze(), '-', color=colors[e])
    ax[e].set_title(f"Env {e_}" + f" - Class {train_labels[e_]}")
    if e >= E_plot-3:
        ax[e].set_xlabel("Normalized Time")
    if e%3 == 0:
        ax[e].set_ylabel(f"$EEG$")

plt.tight_layout()
plt.draw()
plt.savefig(run_folder+"train_trajectories.png")



#%%


def nll_loss(inp, target, log_cov):
    md = 0.5 * jnp.sum((inp - target)**2 / jnp.exp(log_cov), axis=-1)
    logdet = 0.5 * log_cov.sum(axis=-1)
    return jnp.mean(logdet + md)

def env_loss_fn(model, ctx, y_hat, y):
    """
    Loss function for one environment. Leading dimension of y_hat corresponds to the pool size !
    """

    # term1 = jnp.mean((y_hat-y)**2)

    if hasattr(model, "vectorfield"):
        log_cov = model.vectorfield.neuralnet.cov_model(ctx)
        term1 = nll_loss(y_hat, y, log_cov)
    else:
        # log_cov = model.cov_model(ctx)
        # log_cov = jnp.zeros_like(y_hat)
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
                "tf_alpha_start":tf_alpha_start,
                "tf_gamma":tf_gamma,
                "shift_context":shift_context}

model = DirectMapping(MixER_S2S(key=model_key,
                                nb_experts=nb_experts,
                                meta_learner=meta_learner,
                                split_context=split_context,
                                same_expert_init=same_expert_init,
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
                        gating_hyperparams=gating_hyperparams,
                        verbose=False,
                        same_expert_optstate=same_expert_optstate,
                        key=trainer_key)
else:
    print("Skipping meta-training ...")
    restore_folder = run_folder
    trainer.restore_trainer(path=run_folder)


#%%
## Put the model in eval mode to avoid any issues later on
# trainer.learner.model = trainer.learner.inference_mode(trainer.learner.model)



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
print("MSE Loss per InD environment:", all_ind_crit[0].tolist())


#%%
def hellinger_criterion(y_hat, y):
    """ Hellinger distance between two time series """
    pspec_y = compute_and_smooth_power_spectrum(y, 0.0)
    # jax.debug.print("Materialise y_hat {}", jnp.isfinite(y_hat).any())
    pspec_y_hat = compute_and_smooth_power_spectrum(y_hat+1e-7, 0.0)        ## Some dimension might be zero
    return power_spectrum_error(pspec_y_hat, pspec_y)

ind_crit, all_ind_crit = visualtester.evaluate(train_dataloader, 
                                    taylor_order=taylor_orders[1], 
                                    nb_steps=nb_adapt_epochs,
                                    print_error_every=print_error_every, 
                                    loss_criterion=hellinger_criterion,
                                    criterion_id=0,
                                    verbose=True,
                                    val_dataloader=val_dataloader,
                                    max_ret_env_states=max_ret_env_states,
                                    max_adapt_batches=max_adapt_batches,
                                    stochastic=False)

visualtester.visualize_artefacts(save_path=run_folder+"artefacts.png", ylim=None)
print("Hellinger Loss per InD environment:", all_ind_crit[0].tolist())

#%%
ctx_shifts = [learner.model.vectorfield.neuralnet.experts[e].ctx_shift.squeeze() for e in range(nb_experts)]
print("After training, the context shifts are:", jnp.array(ctx_shifts))

#%%
visualtester.visualize_dynamics(save_path=run_folder+"dynamics.png",
                                data_loader=val_dataloader,
                                envs=jnp.arange(0, num_envs[0], 1).tolist(),
                                dims=(0,1),
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
y_labels = np.array([0, 64, 84])
ax2.set_yticks(y_labels)
ax2.set_yticklabels(y_labels)

x_labels = np.arange(0, nb_experts, 1)
ax2.set_xticks(x_labels)
ax2.set_xticklabels(x_labels)

ax2.set_title("Gate Values Heatmap")

plt.draw()
plt.savefig(run_folder+"gate_values.png")
plt.savefig(run_folder+"gate_values.pdf")



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
ani.save(run_folder+"gate_values_animation.gif", writer='pillow', fps=8)


#%%

perp = nb_families if nb_families > 1 else 4
visualtester.visualize_context_clusters(perplexities=(perp, perp),
                                        key=test_key,
                                        save_path=run_folder+"context_clusters.png")






#%%[markdown]
# ## Meta-testing


#%%
## Adapt the model to the new dataset
if meta_test:

    adapt_contexts = []
    all_losses = []

    labels_adapt = []

    ## We want to adapt in batches of 5 environments
    envs_per_batch = 84
    for batch_id, i in enumerate(range(0, num_envs[1], envs_per_batch)):
        print("iteration:", batch_id, "Out of total:", np.ceil(num_envs[1]/envs_per_batch).astype(int))
        print("Adapting on environments:", i, "to", i+envs_per_batch)

        adapt_dataset = LorentzDataset(data_dir=data_folder, 
                                        noisy=noisy_lorentz,
                                        num_steps=num_adapt_steps,
                                        adaptation=True)
        adapt_dataset.total_envs = envs_per_batch
        adapt_dataset.dataset = adapt_dataset.dataset[i:i+envs_per_batch, :, :, :]
        adapt_dataset.t_eval = adapt_dataset.t_eval[i:i+envs_per_batch:, :]

        print("Adaptation dataset shape:", adapt_dataset.dataset.shape)
        nb_adapt_envs, _, _, _ = adapt_dataset.dataset.shape

        adapt_dataloader = NumpyLoader(dataset=adapt_dataset,
                                    # batch_size=num_envs[1], 
                                    batch_size=envs_per_batch, 
                                    shuffle=shuffle,
                                    num_workers=num_workers,
                                    drop_last=False)

        adapt_dataset_test = LorentzDataset(data_dir=data_folder, 
                                            noisy=noisy_lorentz,
                                            num_steps=num_adapt_steps,
                                            adaptation=True)
        adapt_dataset_test.total_envs = envs_per_batch
        adapt_dataset_test.dataset = adapt_dataset_test.dataset[i:i+envs_per_batch, :, :, :]
        adapt_dataset_test.t_eval = adapt_dataset_test.t_eval[i:i+envs_per_batch:, :]

        adapt_dataloader_test = NumpyLoader(dataset=adapt_dataset_test,
                                    # batch_size=num_envs[1], 
                                    batch_size=envs_per_batch,
                                    shuffle=shuffle,
                                    num_workers=num_workers,
                                    drop_last=False)

        ood_crit, all_ood_crit = visualtester.evaluate(adapt_dataloader, 
                                            taylor_order=taylor_orders[1], 
                                            nb_steps=nb_adapt_epochs,
                                            print_error_every=(nb_adapt_epochs, nb_adapt_epochs), 
                                            criterion_id=0,
                                            verbose=True,
                                            val_dataloader=adapt_dataloader_test,
                                            max_ret_env_states=envs_per_batch,
                                            max_adapt_batches=max_adapt_batches,
                                            stochastic=False)
        print("Loss per OoD environment:", all_ood_crit[0].tolist())

        adapt_contexts.append(learner.contexts_latest.params)
        all_losses.append(all_ood_crit[0].tolist())

        labels = get_lorentz_labels()[i:i+envs_per_batch]
        labels_adapt.append(labels)

    adapt_contexts = jnp.concatenate(adapt_contexts, axis=0)
    all_losses = jnp.array(all_losses)
    labels_adapt = jnp.concatenate(labels_adapt, axis=0)

    ## Save these to files
    np.save(adapt_folder+"adapt_losses.npy", all_losses)
    np.save(adapt_folder+"adapt_contexts.npy", adapt_contexts)
    np.save(adapt_folder+"adapt_labels.npy", labels_adapt)


#%%
if meta_test:
    visualtester.visualize_artefacts(save_path=adapt_folder+"artefacts_adapt.png", adaptation=True)

    visualtester.visualize_dynamics(save_path=adapt_folder+"dynamics_adapt.png",
                                    data_loader=adapt_dataloader_test,
                                    nb_envs=1,
                                    traj=0,
                                    dims=(0,1),     ## The Data is 1-dimensional
                                    share_axes=False,
                                    key=test_key)

    perp = nb_families if nb_families > 1 else 4
    visualtester.visualize_context_clusters(perplexities=(perp, perp),
                                            key=test_key,
                                            save_path=adapt_folder+"context_clusters.png")

    # X_adapt = learner.contexts_latest.params
    X_adapt = np.load(adapt_folder+"adapt_contexts.npy")

    # y_test = labels_adapt
    y_test = np.load(adapt_folder+"adapt_labels.npy")










#%%[markdown]
# ## Post-Adaptation analysis



#%%
X = learner.contexts.params
labels = get_lorentz_labels()[:num_envs[0]]

color_table = {0:"royalblue", 1:"crimson"}
colors = [color_table[l] for l in labels]
conditions = {0:"Lorentz96", 1:"Lorentz63"}

## Use PCA 
from sklearn.decomposition import PCA
pca = PCA(n_components=2)

nb_train_samples = X.shape[0]
X_total = np.concatenate([X, X_adapt], axis=0) if meta_test else X
X_reduced = pca.fit_transform(X_total) if X.shape[1] > 2 else X

plt.figure(figsize=(10, 7))

for class_label in [0,1]:
    marker = "^" if class_label==0 else "x"
    plt.scatter(X_reduced[:nb_train_samples][labels==class_label, 0], X_reduced[:nb_train_samples][labels==class_label, 1], s=50, c=color_table[class_label], label=conditions[class_label], marker=marker)

if meta_test:
    X_adapt_rec = pca.transform(X_adapt) if X_adapt.shape[1] > 2 else X_adapt
    labels_adapt = y_test
    for class_label in [0,1]:
        marker = "." if class_label==0 else "."
        plt.scatter(X_reduced[nb_train_samples:][labels_adapt==class_label, 0], X_reduced[nb_train_samples:][labels_adapt==class_label, 1], s=20, c=color_table[class_label], label=conditions[class_label]+" (Test)", marker=marker, alpha=0.5)

plt.legend()

plt.title("Contexts Clustering - PCA", fontsize=24)
plt.xlabel("PC 1") if X.shape[1] > 1 else plt.xlabel("Ctx 1")
plt.ylabel("PC 2") if X.shape[1] > 1 else plt.ylabel("Ctx 2")

plt.draw()
plt.savefig(run_folder+"clusters_pca.png", bbox_inches='tight');


#%%


#%%
X = learner.contexts.params
labels = get_lorentz_labels()[:num_envs[0]]

color_table = {0:"royalblue", 1:"crimson"}
colors = [color_table[l] for l in labels]
conditions = {0:"Lorentz96", 1:"Lorentz63"}

import umap
nb_train_samples = X.shape[0]
umap_reducer = umap.UMAP(n_components=2, random_state=int(test_key[0]), min_dist=0.0, spread=1.0, metric="euclidean")
X_total = np.concatenate([X, X_adapt], axis=0) if meta_test else X
X_reduced = umap_reducer.fit_transform(X_total) if X.shape[1] > 2 else X

plt.figure(figsize=(10, 7))

for class_label in [0,1]:
    marker = "^" if class_label==0 else "x"
    plt.scatter(X_reduced[:nb_train_samples][labels==class_label, 0], X_reduced[:nb_train_samples][labels==class_label, 1], s=50, c=color_table[class_label], label=conditions[class_label], marker=marker)

if meta_test:
    X_adapt_rec = pca.transform(X_adapt) if X_adapt.shape[1] > 2 else X_adapt
    labels_adapt = y_test
    for class_label in [0,1]:
        marker = "." if class_label==0 else "."
        plt.scatter(X_reduced[nb_train_samples:][labels_adapt==class_label, 0], X_reduced[nb_train_samples:][labels_adapt==class_label, 1], s=20, c=color_table[class_label], label=conditions[class_label]+" (Test)", marker=marker, alpha=0.5)

plt.legend()

plt.title("Contexts Clustering - UMAP", fontsize=24)
plt.xlabel("UMAP 1") if X.shape[1] > 1 else plt.xlabel("Ctx 1")
plt.ylabel("UMAP 2") if X.shape[1] > 1 else plt.ylabel("Ctx 2")

plt.draw()
plt.savefig(run_folder+"clusters_umap.png", bbox_inches='tight');


#%%

X = learner.contexts.params
y = get_lorentz_labels()[:num_envs[0]]

## Let's do the classification with SVM and a non-linear kernel
from sklearn.svm import SVC
clf = SVC(kernel='rbf', random_state=seed)
clf.fit(X, y)

# ## Let's clasify with a GMM instead
# from sklearn.mixture import GaussianMixture
# clf = GaussianMixture(n_components=2, random_state=seed)
# clf.fit(X)

if meta_test:
    print("SVM trained on the contexts")
    y_pred = clf.predict(X_adapt)
    print("Results shapes ", y_pred.shape, y_test.shape)

    ## Calculate accuracy with sklearn metrics
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)

    print("Y_test = ", y_test)
    print("Y_pred = ", y_pred)





#%%
## After training, copy nohup.log to the runfolder
try:
    __IPYTHON__ ## in a jupyter notebook
except NameError:
    os.system(f"cp nohup.log {run_folder}")

# %%
