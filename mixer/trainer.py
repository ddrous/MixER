import pickle
from typing import Any, Tuple

from mixer.dataloader import DataLoader
from mixer.learner import Learner
from mixer.visualtester import VisualTester
from ._utils import *





#%%

class Trainer:
    def __init__(self, learner:Learner, optimisers, key=None):
        """ Base class for training the models"""

        if key is None:
            raise ValueError("You must provide a key for the trainer")
        self.key = key      ## Default training key

        if not isinstance(learner, Learner):
            raise ValueError("The learner must be an instance of Learner")
        else:
            self.learner = learner
        self.opt_model, self.opt_ctx = optimisers

        self.opt_state_model = self.opt_model.init(eqx.filter(self.learner.model, eqx.is_array))

        self.losses_model = []
        self.losses_ctx = []
 
    def save_trainer(self, path, ignore_losses=False, ignore_learner=False):
        assert path[-1] == "/", "ERROR: The path must end with /"
        # print(f"\nSaving model and results into {path} folder ...\n")

        if not ignore_losses:
            np.savez(path+"train_histories.npz",
                    losses_model=jnp.vstack(self.losses_model), 
                    losses_ctx=jnp.vstack(self.losses_ctx))

            if hasattr(self, 'val_losses'):
                try:
                    np.save(path+"val_losses.npy", jnp.vstack(self.val_losses))
                except Exception as e:
                    # print(f"Error when saving the validation losses - {e}")
                    ## Print fail silently
                    pass

        pickle.dump(self.opt_state_model, open(path+"opt_state_model.pkl", "wb"))
        pickle.dump(self.opt_state_ctx, open(path+"opt_state_ctx.pkl", "wb"))

        # if not hasattr(self, 'val_losses'):
        if not ignore_learner:
            self.learner.save_learner(path)

    def restore_trainer(self, path):
        assert path[-1] == "/", "ERROR: Invalidn parovided. The path must end with /"
        print(f"\nLoading model and results from {path} folder ...\n")

        if os.path.exists(path+"train_histories.npz"):
            histories = np.load(path+"train_histories.npz")
        elif os.path.exists(path+"checkpoints/train_histories.npz"):
            print("WARNING: No training history found in the provided path. Using checkpointed ones.")
            histories = np.load(path+"checkpoints/train_histories.npz")
        else:
            print("WARNING: No training history found at all. Using tens.")
            histories = {'losses_model': jnp.inf*np.ones((1,1)), 'losses_ctx': jnp.inf*np.ones((1,1))}
        self.losses_model = [histories['losses_model']]
        self.losses_ctx = [histories['losses_ctx']]

        if os.path.exists(path+"val_losses.npy"):
            self.val_losses = [np.load(path+"val_losses.npy")]
        elif os.path.exists(path+"checkpoints/val_losses.npy"):
            print("WARNING: No validation history found in the provided path. Using checkpointed ones.")
            self.val_losses = [np.load(path+"checkpoints/val_losses.npy")]
        else:
            print("WARNING: No validation history found at all. Using ten.")
            # self.val_losses = [jnp.inf*np.ones((1,1))]
            self.val_losses = []

        if os.path.exists(path+"opt_state_model.pkl"):
            try:
                self.opt_state_model = pickle.load(open(path+"opt_state_model.pkl", "rb"))
            except Exception as e:
                print(f"Error when loading the model optimiser state - {e}")
                self.opt_state_model = None
            try:
                self.opt_state_ctx = pickle.load(open(path+"opt_state_ctx.pkl", "rb"))
            except Exception as e:
                print(f"Error when loading the context optimiser state - {e}")
                self.opt_state_ctx = None
        else:
            print("WARNING: No optimiser state found in the provided path.")
            # self.opt_state_model = None
            # self.opt_state_ctx = None

        self.learner.load_learner(path)


    def save_adapted_trainer(self, path):
        print(f"\nSaving adaptation parameters into {path} folder ...\n")

        np.savez(path+"adapt_histories_.npz", losses_adapt=jnp.vstack(self.losses_adapt))
        # pickle.dump(self.opt_state_adapt, open(path+"/opt_state_adapt.pkl", "wb"))
        eqx.tree_serialise_leaves(path+"/adapted_contexts_.eqx", self.learner.contexts_adapt)






















class NCFTrainer(Trainer):
    def __init__(self, learner:Learner, optimisers, schedulers=None, key=None):
        """ Trainer class for the proximal gradient descent algorithms (NCF) """

        if schedulers is not None:
            self.scheduler_model, self.scheduler_ctx = schedulers
        else:
            self.scheduler_model, self.scheduler_ctx = None, None

        if self.scheduler_model is None:
            self.scheduler_model = optax.constant_schedule(1e-4)
        elif isinstance(self.scheduler_model, float):
            self.scheduler_model = optax.constant_schedule(self.scheduler_model)
        if self.scheduler_ctx is None:
            self.scheduler_ctx = optax.constant_schedule(1e-4)
        elif isinstance(self.scheduler_ctx, float):
            self.scheduler_ctx = optax.constant_schedule(self.scheduler_ctx)

        super().__init__(learner, optimisers, key)

    def meta_train(self,
                    dataloader: DataLoader, 
                    nb_epochs,
                    nb_outer_steps,
                    nb_inner_steps=(1, 10),
                    inner_tols=(1e-12, 1e-12), 
                    proximal_betas=(100., 100.), 
                    max_train_batches=None,
                    patience=None, 
                    print_error_every=(1,1), 
                    save_checkpoints=False,
                    validate_every=100,
                    save_path=False, 
                    val_dataloader=None, 
                    val_criterion_id=None, 
                    max_val_batches=None,
                    val_nb_steps=10,
                    key=None):
        """ Train the model using the proximal gradient descent algorithm (PAM) """

        key = key if key is not None else self.key

        ## Try to load checkpoints if they exist
        try:
            self.restore_trainer(path=save_path)
            print(f"Restored the trainer from the path {save_path}")
            # if self.opt_state_model is None:
            #     print(f"No checkpoints or error after loading training. Initialising a new one ...")

        except Exception as e:
            print(f"No checkpoints or error when attempting to load training configs - '{e}' - Starting from scratch ...")

        if isinstance(nb_inner_steps, int):
            nb_inner_steps = (nb_inner_steps, nb_inner_steps)
        nb_inner_steps_model, nb_inner_steps_ctx = nb_inner_steps

        inner_tol_model, inner_tol_ctx = inner_tols
        proximal_reg_model, proximal_reg_ctx = proximal_betas

        loss_fn = self.learner.loss_fn
        model = self.learner.model
        opt_state_model = self.opt_state_model
        loss_filling = self.learner.loss_filling
        nb_loss_contr = self.learner.loss_contributors

        if save_checkpoints:
            backup_folder = save_path+"checkpoints/"
            if not os.path.exists(backup_folder):
                os.makedirs(backup_folder)

        @eqx.filter_jit
        def train_step_model(model, model_old, contexts, batch, weightings, opt_state, key):
            print('     ### Compiling function "train_step" for the model ...  ')

            def prox_loss_fn(model, contexts, batch, weightings, key):
                loss, aux_data = loss_fn(model, contexts, batch, weightings, key)
                diff_norm = params_diff_norm_squared(model, model_old)
                return loss + proximal_reg_model * diff_norm / 2., (*aux_data, diff_norm)

            (loss, aux_data), grads = eqx.filter_value_and_grad(prox_loss_fn, has_aux=True)(model, contexts, batch, weightings, key)

            updates, opt_state = self.opt_model.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)

            return model, contexts, opt_state, loss, aux_data


        @eqx.filter_jit
        def train_step_ctx(model, contexts, contexts_old, batch, weightings, opt_state, key):
            print('     ### Compiling function "train_step" for the contexts ...  ')

            def prox_loss_fn(contexts, model, batch, weightings, key):
                loss, aux_data = loss_fn(model, contexts, batch, weightings, key)
                diff_norm = params_diff_norm_squared(contexts, contexts_old)
                return loss + proximal_reg_ctx * diff_norm / 2., (*aux_data, diff_norm)

            (loss, aux_data), grads = eqx.filter_value_and_grad(prox_loss_fn, has_aux=True)(contexts, model, batch, weightings, key)

            updates, opt_state = self.opt_ctx.update(grads, opt_state)
            contexts = eqx.apply_updates(contexts, updates)

            return model, contexts, opt_state, loss, aux_data

        # if not isinstance(dataloader, DataLoader):
        #     raise ValueError("The dataloader must be an instance of DataLoader")
        if val_dataloader is not None:
            tester = VisualTester(self, key=key)

        validate_every = validate_every if validate_every > 0 else 1

        print(f"\n\n=== Beginning Meta-Training ... ===")
        print(f"    Number of examples in a batch along envs: {dataloader.batch_size}")
        print(f"    Maximum number of batches (along envs): {dataloader.num_batches}")
        print(f"    Total number of epochs: {nb_epochs}")
        print(f"    Number of outer minimizations: {nb_outer_steps}")
        print(f"    Maximum numbers of inner steps per outer minimizations: {nb_inner_steps_model, nb_inner_steps_ctx}")

        if max_train_batches is None or max_train_batches<1 or max_train_batches>dataloader.num_batches:
            max_train_batches = dataloader.num_batches
        print(f"    Training on {max_train_batches} batches")
        if val_dataloader is not None:
            if max_val_batches is None or max_val_batches<1 or max_val_batches>val_dataloader.num_batches:
                max_val_batches = val_dataloader.num_batches
            print(f"    Validating on {max_val_batches} batches")

        if isinstance(print_error_every, int):
            print_error_every = (print_error_every, print_error_every)
        print_every_out_step, print_every_batch = print_error_every

        start_time = time.time()

        losses_model = []
        losses_ctx = []
        if val_dataloader is not None:
            val_losses = []

        loss_key, _ = jax.random.split(key)
        early_stopping_count = 0

        for epoch in range(nb_epochs):
            # print(f"\nEPOCH {epoch} ... ")

            for env_batch, batch in enumerate(dataloader):
                if env_batch >= max_train_batches:
                    break
                # if env_batch%10==0:
                #     print(f"  Learning on batch {env_batch} ...")

                nb_envs_in_batch = batch[1].shape[0]
                # weightings = jnp.ones(nb_envs_in_batch) / nb_envs_in_batch

                # ## Approach #1
                # if loss_filling == "NF-iW":
                #     all_env_losses = 1e-2*jnp.ones(nb_envs_in_batch)
                # else:
                #     all_env_losses = 10*jnp.ones(nb_envs_in_batch)
                # min_loss_weight = 1e-3

                ## Approach #2
                if loss_filling == "NF-iW":
                    all_env_losses = jnp.inf*jnp.ones(nb_envs_in_batch)
                elif loss_filling == "NF-W":
                    all_env_losses = jnp.zeros(nb_envs_in_batch)
                else:
                    all_env_losses = jnp.ones(nb_envs_in_batch) / nb_envs_in_batch

                contexts = self.learner.reset_contexts(nb_envs_in_batch)
                opt_state_ctx = self.opt_ctx.init(eqx.filter(contexts, eqx.is_array))

                for out_step in range(nb_outer_steps):
                    # print(f"    Staring outer step {out_step} ...")

                    loss_epochs_model = []
                    loss_epochs_ctx = []

                    start_time_step = time.perf_counter()

                    model_old = jax.tree_util.tree_map(lambda x: x, model)
                    contexts_old = jax.tree_util.tree_map(lambda x: x, contexts)

                    ## Model proximal innner minimization
                    # model_prev = jax.tree_util.tree_map(lambda x: x, model)
                    for in_step_model in range(nb_inner_steps_model):

                        loss_key, _ = jax.random.split(loss_key)

                        model, contexts, opt_state_model, loss_model, (term1, term2, _, loss_contrs, _) = train_step_model(model, model_old, contexts, batch, all_env_losses, opt_state_model, loss_key)

                        # ###========== Approach #1 to only sample relevent environments.
                        # all_env_losses = all_env_losses.at[loss_contrs].set(term1)
                        # # all_env_losses = jnp.clip(all_env_losses, 1e-4, 10.)
                        # all_env_losses = jnp.clip(all_env_losses, min=all_env_losses.min()+min_loss_weight, 
                        #                           max=all_env_losses.min()+min_loss_weight+10*min_loss_weight)

                        # ###========== Approach #2
                        all_env_losses = all_env_losses.at[loss_contrs].set(term1)

                        # print(loss_contrs.tolist(), end=" - ")
                        loss_epochs_model.append(loss_model)

                        ## TODO Update the weightings based on loss progress
                        # keys = jax.random.split(key, num=contexts.params.shape[0])

                        # diff_model = params_diff_norm_squared(model, model_prev) / params_norm_squared(model_prev)
                        # if diff_model < inner_tol_model or out_step==0:
                        #     break
                        # model_prev = model


                    ## Contexts proximal innner minimization
                    # contexts_prev = jax.tree_util.tree_map(lambda x: x, contexts)
                    for in_step_ctx in range(nb_inner_steps_ctx):

                        loss_key, _ = jax.random.split(loss_key)

                        model, contexts, opt_state_ctx, loss_ctx, (term1, term2, term3, loss_contrs, _) = train_step_ctx(model, contexts, contexts_old, batch, all_env_losses, opt_state_ctx, loss_key)

                        # ###========== Approach #1
                        # all_env_losses = all_env_losses.at[loss_contrs].set(term1)
                        # # all_env_losses = jnp.clip(all_env_losses, 1e-4, 10.)
                        # all_env_losses = jnp.clip(all_env_losses, min=all_env_losses.min()+min_loss_weight, 
                        #                           max=all_env_losses.min()+min_loss_weight+10*min_loss_weight)

                        # ###========== Approach #2
                        all_env_losses = all_env_losses.at[loss_contrs].set(term1)

                        # print(loss_contrs.tolist(), end=" - " if in_step_ctx < nb_inner_steps_ctx-1 else "\n")
                        loss_epochs_ctx.append(loss_ctx)

                        # diff_ctx = params_diff_norm_squared(contexts, contexts_prev) / params_norm_squared(contexts_prev)
                        # if diff_ctx < inner_tol_ctx or out_step==0:
                        #     break
                        # contexts_prev = contexts

                    # print("Current contributors: ", loss_contrs, "\nall env losses:\n", all_env_losses.tolist())

                    if in_step_model < 1 and in_step_ctx < 1:
                        early_stopping_count += 1
                    else:
                        early_stopping_count = 0

                    if (patience is not None) and (early_stopping_count >= patience):
                        print(f"Stopping early after {patience} steps with no improvement in the loss. Consider increasing the tolerances for the inner minimizations.")
                        break

                    # losses_model.append(loss_model)
                    # losses_ctx.append(loss_ctx)

                    # losses_model.append(jnp.mean(jnp.array(loss_epochs_model)))
                    # losses_ctx.append(jnp.mean(jnp.array(loss_epochs_ctx)))

                    losses_model.append(jnp.median(jnp.array(loss_epochs_model)))
                    losses_ctx.append(jnp.median(jnp.array(loss_epochs_ctx)))

                    if env_batch%print_every_batch==0 or env_batch==max_train_batches-1:
                        if out_step%print_every_out_step==0 or out_step==nb_outer_steps-1:
                            print(f"{time.strftime('%H:%M:%S')}      Epoch: {epoch:-3d}      Batch: {env_batch:-3d}      OuterStep: {out_step:-3d}      LossModel: {losses_model[-1]:-.8f}     LossTerm2: {jnp.mean(term2):-.8f}      Time/Step(s): {time.perf_counter()-start_time_step:-.4f}", flush=True, end="\r")
                            # print(f"\n\t-NbInnerStepsMod: {in_step_model+1:4d}\n\t-NbInnerStepsCxt: {in_step_ctx+1:4d}\n\t-DiffMod:   {diff_model:.2e}\n\t-DiffCxt:   {diff_ctx:.2e}", flush=True, end="\r")
                            # print(f"Training losses per environment: {all_env_losses.tolist()}", flush=True, end="\n")
                            print("Term3 quantities: \n", term3[1], flush=True, end="\n")
                            print("Contributed losses: ", loss_contrs, flush=True, end="\n")
                            # print("Gate weights in model are: \n", model.vectorfield.neuralnet.gate_weight, flush=True, end="\n")

                            if save_checkpoints:
                                ## Save the context and model with the right suffix
                                # context_save_path = backup_folder+f"contexts_outstep_{out_step:06d}.npy"
                                # np.save(context_save_path, contexts.params)
                                context_save_path = backup_folder+f"contexts_outstep_{out_step:06d}.eqx"
                                eqx.tree_serialise_leaves(context_save_path, contexts)
                                eqx.tree_serialise_leaves(backup_folder+f"model_outstep_{out_step:06d}.eqx", model)
                                np.savez(backup_folder+"train_histories.npz",
                                    losses_model=jnp.vstack([jnp.vstack(losses_model)]), 
                                    losses_ctx=jnp.vstack([jnp.vstack(losses_ctx)]))
                                # np.save(backup_folder+"val_losses.npy", jnp.vstack([jnp.vstack(val_losses)]))

                    if val_dataloader is not None and (out_step != 0 and (out_step%validate_every==0 or out_step==nb_outer_steps-1)):
                        self.learner.model = model
                        self.learner.contexts = contexts
                        # print("Setting contexts in the metatrainer: \n", contexts.params)
                        self.learner.all_env_losses = all_env_losses

                        self.opt_state_model = opt_state_model
                        self.opt_state_ctx = opt_state_ctx

                        ind_crit,_ = tester.evaluate(val_dataloader,
                                                    criterion_id=val_criterion_id,
                                                    max_adapt_batches=max_val_batches,
                                                    nb_steps=val_nb_steps,
                                                    taylor_order=0, 
                                                    max_ret_env_states=self.learner.loss_contributors,
                                                    stochastic=False,
                                                    verbose=False)
                        print(f"        Validation Criterion: {ind_crit:-.8f}", flush=True)
                        val_losses.append(np.array([out_step, ind_crit]))

                        ## Check if val loss is lowest to save the model
                        if ind_crit <= jnp.stack(val_losses)[:,1].min() and save_path:
                            print(f"        Saving best model so far ...")
                            self.save_trainer(save_path, ignore_losses=True)
                            # self.learner.save_learner(save_path)
                        ## Restore the learner at the last evaluation step
                        if out_step == nb_outer_steps-1:
                            self.save_trainer(save_path, ignore_losses=True)
                            # self.learner.load_learner(save_path)

                    ###========== Approach #2 to only sample relevent environments. Find the worst contributors
                    loss_sort = jnp.argsort(all_env_losses)
                    if loss_filling == "NF-iW":
                        # all_env_losses = all_env_losses.at[loss_sort[-nb_envs_in_batch//2:]].set(jnp.inf)
                        all_env_losses = all_env_losses.at[loss_sort[-nb_loss_contr:]].set(jnp.inf)
                    elif loss_filling == "NF-W":
                        all_env_losses = all_env_losses.at[loss_sort[:nb_loss_contr]].set(0.)

                # print(f"\n\t-NbInnerStepsMod: {in_step_model+1:4d}\n\t-NbInnerStepsCxt: {in_step_ctx+1:4d}\n\t-DiffMod:   {diff_model:.2e}\n\t-DiffCxt:   {diff_ctx:.2e}", flush=True, end="\r")


        wall_time = time.time() - start_time
        time_in_hmsecs = seconds_to_hours(wall_time)
        print("\nTotal gradient descent training time: %d hours %d mins %d secs" %time_in_hmsecs)

        self.losses_model.append(jnp.vstack(losses_model))
        self.losses_ctx.append(jnp.vstack(losses_ctx))

        if val_dataloader is not None:
            if not hasattr(self, 'val_losses'):
                self.val_losses = []
            self.val_losses.append(jnp.vstack(val_losses))

        self.opt_state_model = opt_state_model
        if val_dataloader is None:
            self.learner.model = model

        ## DO NOT TRUST. Just for visualisation purposes
        self.opt_ctx_state = opt_state_ctx
        self.learner.contexts = contexts

        # Save the model and results
        if save_path:
            self.save_trainer(save_path)




    def meta_train_gated(self,
                        dataloader: DataLoader, 
                        nb_epochs,
                        nb_outer_steps,
                        nb_inner_steps=(10, 10),
                        proximal_betas=(100., 100.), 
                        max_train_batches=None,
                        print_error_every=(1,1), 
                        save_checkpoints=False,
                        validate_every=100,
                        save_path=False, 
                        val_dataloader=None, 
                        val_criterion_id=None, 
                        max_val_batches=None,
                        val_nb_steps=10,
                        update_gate_every=1,
                        gating_hyperparams={"noise_level": 1e-4, "max_kmeans": 20, "convergence_tol": 1e-4},
                        same_expert_optstate=False,
                        verbose=False,
                        key=None):
        """ Train the model using the proximal gradient descent algorithm (PAM) """

        key = key if key is not None else self.key

        ## Attempt to load checkpoints if any exists
        try:
            self.restore_trainer(path=save_path)
            print(f"Restored the trainer from the path {save_path}")
        except Exception as e:
            print(f"Unable to load training checkpoints - '{e}'\nStarting from scratch ...")

        if isinstance(nb_inner_steps, int):
            nb_inner_steps = (nb_inner_steps, nb_inner_steps, nb_inner_steps)
        nb_inner_steps_model_max, nb_inner_steps_ctx_max = nb_inner_steps
        proximal_reg_model_max, proximal_reg_ctx_max = proximal_betas

        loss_fn = self.learner.loss_fn
        model = self.learner.model
        loss_filling = self.learner.loss_filling
        nb_loss_contr = self.learner.loss_contributors

        if same_expert_optstate and model.vectorfield.neuralnet.is_moe:
            ## 1 optimizer state is used for all the experts, and the gate dealt with seperately
            expert_0 = model.vectorfield.neuralnet.experts[0]
            gate = model.vectorfield.neuralnet.gate
            opt_state_model = self.opt_model.init(eqx.filter(expert_0, eqx.is_array))
            opt_state_gate = self.opt_model.init(eqx.filter(gate, eqx.is_array))
            opt_state_model = (opt_state_model, opt_state_gate)
        else:
            opt_state_model = self.opt_state_model

        if save_checkpoints:
            backup_folder = save_path+"checkpoints/"
            if not os.path.exists(backup_folder):
                os.makedirs(backup_folder)

        @eqx.filter_jit
        def train_step_model(model, model_aux, contexts, batch, weightings, opt_state, key):            ## TODO stop the model from updateing the weights
            print('     ### Compiling function "train_step" for the model ...  ')

            model_old, proximal_reg_model = model_aux

            def prox_loss_fn(model, contexts, batch, weightings, key):
                loss, aux_data = loss_fn(model, contexts, batch, weightings, key)
                diff_norm = params_diff_norm_squared(model, model_old)
                return loss + proximal_reg_model * diff_norm / 2., (*aux_data, diff_norm)

            (loss, aux_data), grads = eqx.filter_value_and_grad(prox_loss_fn, has_aux=True)(model, contexts, batch, weightings, key)

            if same_expert_optstate:
                ## Let's make all experts share the same opt_state
                opt_state_expt, opt_state_gate = opt_state
                experts = []
                for e in range(model.vectorfield.neuralnet.n_experts):
                    updates, opt_state_expt = self.opt_model.update(grads.vectorfield.neuralnet.experts[e], opt_state_expt)
                    expert_e = eqx.apply_updates(model.vectorfield.neuralnet.experts[e], updates)
                    experts.append(expert_e)
                model = eqx.tree_at(lambda m:m.vectorfield.neuralnet.experts, model, experts)

                ## Update the gate weights
                if not least_squares_update:
                    updates, opt_state_gate = self.opt_model.update(grads.vectorfield.neuralnet.gate, opt_state_gate)
                    gate = eqx.apply_updates(model.vectorfield.neuralnet.gate, updates)
                    model = eqx.tree_at(lambda m:m.vectorfield.neuralnet.gate, model, gate)

                opt_state = (opt_state_expt, opt_state_gate)
            else:
                updates, opt_state = self.opt_model.update(grads, opt_state)
                model = eqx.apply_updates(model, updates)

            return model, contexts, opt_state, loss, aux_data


        @eqx.filter_jit
        def train_step_ctx(model, contexts, contexts_aux, batch, weightings, opt_state, key):
            print('     ### Compiling function "train_step" for the contexts ...  ')

            contexts_old, proximal_reg_ctx = contexts_aux

            def prox_loss_fn(contexts, model, batch, weightings, key):
                loss, aux_data = loss_fn(model, contexts, batch, weightings, key)
                diff_norm = params_diff_norm_squared(contexts, contexts_old)
                return loss + proximal_reg_ctx * diff_norm / 2., (*aux_data, diff_norm)

            (loss, aux_data), grads = eqx.filter_value_and_grad(prox_loss_fn, has_aux=True)(contexts, model, batch, weightings, key)

            updates, opt_state = self.opt_ctx.update(grads, opt_state)
            contexts = eqx.apply_updates(contexts, updates)

            return model, contexts, opt_state, loss, aux_data


        least_squares_update = model.vectorfield.neuralnet.gate_update_strategy == "least_squares"

        gate_noise_level = gating_hyperparams["noise_level"]
        gate_max_kmeans = gating_hyperparams["max_kmeans"]
        gate_convergence_tol = gating_hyperparams["convergence_tol"]

        gate_noise_level = gate_noise_level if gate_noise_level is not None else 1e-4
        gate_max_kmeans = gate_max_kmeans if gate_max_kmeans>0 else 20
        gate_convergence_tol = gate_convergence_tol if gate_convergence_tol>0 else 1e-3

        # @eqx.filter_jit
        def train_step_gates(model, contexts, init_centroids, batch, key):
            # print('     ### Compiling function "train_step" for the gates ...  ')

            ############ Least Squares Strategy to Select the Best Expert ############

            ## Step 1. See how each expert performs on each environment individually
            loss, aux_data = self.learner.loss_fn_multitask(model, contexts, batch, key)
            expert_losses = aux_data[0]     ## Shape: (nb_envs, nb_experts) i.e. for each environment, we know the loss of each expert

            ## Step 2. Find environment clusters
            nb_envs, context_size = contexts.params.shape
            nb_experts = nb_clusters = expert_losses.shape[1]
            max_iter = gate_max_kmeans
            ctx = contexts.params
            if init_centroids is None:
                centroids = jax.random.uniform(key, (nb_clusters, context_size), minval=-1e-3, maxval=1e-3)
            else:
                centroids = init_centroids

            for i in range(max_iter):
                old_centroids = centroids       ## Used to check for convergence

                ## Assign each point to the nearest centroid
                # distances = jnp.linalg.norm(ctx[:, None, :] - centroids[None, :, :], axis=-1)
                distances = jnp.abs(ctx[:, None, :] - centroids[None, :, :]).mean(axis=-1)
                cluster_assignments = jnp.argmin(distances, axis=-1)

                ## Update the centroids
                for j in range(nb_clusters):
                    cluster_points = ctx[cluster_assignments == j]

                    ## if cluster_points is empty, then skip this update step and wait
                    if cluster_points.shape[0] == 0:
                        if True:
                            print(f"❌ Cluster {j} is empty. Skipping gating least squares and waiting for next turn.")
                        return model, contexts, loss, None, jnp.zeros((nb_envs, nb_experts))
                    else:
                        centroids = centroids.at[j].set(jnp.mean(cluster_points, axis=0))

                ## Check for convergence
                if jnp.allclose(centroids, old_centroids, atol=gate_convergence_tol):
                    if True:
                        print(f"✔️ Gating k-means converged after {i+1} iterations", flush=False)
                    break

            if i == max_iter-1:
                if True:
                    print(f"❌ Gating k-means did not converge despite {max_iter} iterations. Waiting next turn.", flush=False)
                    return model, contexts, loss, None, jnp.zeros((nb_envs, nb_experts))

            ## Step 3. Calculate the agregate loss for each cluster for each expert
            expert_cluster_losses = []
            for j in range(nb_clusters):
                # expert_cluster_losses.append(expert_losses[cluster_assignments==j].mean(axis=0))
                expert_cluster_losses.append(jnp.median(expert_losses[cluster_assignments==j], axis=0))
            all_expert_cluster_losses = jnp.stack(expert_cluster_losses, axis=0)

            if verbose:
                print(f"Median expert losses per cluster: \n{all_expert_cluster_losses}\n", flush=False)

            ## Step 4. Assign clusters to the experts
            used_experts = []  ## The experts that have been assigned to a cluster
            chosen_experts = [] ## The experts that have been chosen as the best expert for a cluster
            for j in range(nb_clusters):
                ## Find the expert with the lowest loss in the cluster
                sorted_mins = jnp.argsort(all_expert_cluster_losses[j])
                min_expert = sorted_mins[0]
                while min_expert in used_experts:
                    sorted_mins = sorted_mins[1:]
                    min_expert = sorted_mins[0]
                used_experts.append(min_expert)
                chosen_experts.append(min_expert)

            ## Step 5. Define the gating problem to solve: X=ctx, Y=chosen_experts (one-hoe)
            Y = jnp.zeros((nb_envs, nb_experts))
            for j in range(nb_clusters):
                Y = Y.at[cluster_assignments==j].set(jax.nn.one_hot(chosen_experts[j], nb_experts))

            X = contexts.params + jax.random.normal(key, contexts.params.shape) * gate_noise_level
            if model.vectorfield.neuralnet.use_gate_bias:
                X = jnp.concatenate([X, jnp.ones((nb_envs, 1))], axis=-1)
            W = jnp.linalg.lstsq(X, Y, rcond=None)[0]       ## TODO: check if solved correctly

            ## Step 6. Update the gating weights
            model = eqx.tree_at(lambda m:m.vectorfield.neuralnet.gate, model, W)

            return model, contexts, loss, centroids, Y


        if val_dataloader is not None:
            tester = VisualTester(self, key=key)

        validate_every = validate_every if validate_every > 0 else 1

        print(f"\n\n=== Beginning Meta-Training ... ===")
        print(f"    Number of examples in a batch along envs: {dataloader.batch_size}")
        print(f"    Maximum number of batches (along envs): {dataloader.num_batches}")
        print(f"    Total number of epochs: {nb_epochs}")
        print(f"    Number of outer minimizations: {nb_outer_steps}")
        print(f"    Maximum numbers of inner steps per outer minimizations: {nb_inner_steps_model_max, nb_inner_steps_ctx_max}")

        if max_train_batches is None or max_train_batches<1 or max_train_batches>dataloader.num_batches:
            max_train_batches = dataloader.num_batches
        print(f"    Training on {max_train_batches} batches")
        if val_dataloader is not None:
            if max_val_batches is None or max_val_batches<1 or max_val_batches>val_dataloader.num_batches:
                max_val_batches = val_dataloader.num_batches
            print(f"    Validating on {max_val_batches} batches")

        if isinstance(print_error_every, int):
            print_error_every = (print_error_every, print_error_every)
        print_every_out_step, print_every_batch = print_error_every

        start_time = time.time()

        losses_model = []
        losses_ctx = []
        losses_gates = []
        if val_dataloader is not None:
            val_losses = []

        loss_key, _ = jax.random.split(key)

        for epoch in range(nb_epochs):
            # print(f"\nEPOCH {epoch} ... ")

            for env_batch, batch in enumerate(dataloader):
                if env_batch >= max_train_batches:
                    break

                nb_envs_in_batch = batch[1].shape[0]

                ## All environments losses might be useful in evaluating the gating
                if loss_filling in ["NF-iW", "NF-B"]:
                    all_env_losses = jnp.inf*jnp.ones(nb_envs_in_batch)
                elif loss_filling in ["NF-W"]:
                    all_env_losses = jnp.zeros(nb_envs_in_batch)
                else:
                    all_env_losses = jnp.ones(nb_envs_in_batch) / nb_envs_in_batch

                contexts = self.learner.reset_contexts(nb_envs_in_batch)
                opt_state_ctx = self.opt_ctx.init(eqx.filter(contexts, eqx.is_array))

                centroids = None    ## Meaning initial random initialisation of the centroids

                for out_step in range(nb_outer_steps):

                    loss_epochs_model = []
                    loss_epochs_ctx = []

                    start_time_step = time.perf_counter()

                    model_old = jax.tree_util.tree_map(lambda x: x, model)
                    contexts_old = jax.tree_util.tree_map(lambda x: x, contexts)

                    ######### Contexts proximal innner minimization #########
                    for in_step_ctx in range(nb_inner_steps_ctx_max):

                        loss_key, _ = jax.random.split(loss_key)

                        model, contexts, opt_state_ctx, loss_ctx, (term1, term2, term3, loss_contrs, _) = train_step_ctx(model, contexts, (contexts_old, proximal_reg_ctx_max), batch, all_env_losses, opt_state_ctx, loss_key)

                        ## If using least squares update, update the gating weights
                        if least_squares_update and in_step_ctx%update_gate_every == 0:
                            model, contexts, loss_gates, centroids, _ = train_step_gates(model, contexts, centroids, batch, loss_key)

                        all_env_losses = all_env_losses.at[loss_contrs].set(term1)
                        loss_epochs_ctx.append(loss_ctx)

                    ######### Model proximal innner minimization #########
                    for in_step_model in range(nb_inner_steps_model_max):

                        loss_key, _ = jax.random.split(loss_key)

                        model, contexts, opt_state_model, loss_model, (term1, term2, _, loss_contrs, _) = train_step_model(model, (model_old, proximal_reg_model_max), contexts, batch, all_env_losses, opt_state_model, loss_key)

                        if least_squares_update and in_step_model%update_gate_every == 0:
                            model, contexts, loss_gates, centroids, _ = train_step_gates(model, contexts, centroids, batch, loss_key)

                        all_env_losses = all_env_losses.at[loss_contrs].set(term1)
                        loss_epochs_model.append(loss_model)

                    losses_model.append(jnp.median(jnp.array(loss_epochs_model)))
                    losses_ctx.append(jnp.median(jnp.array(loss_epochs_ctx)))
                    losses_gates.append(loss_gates if least_squares_update else 0.)

                    if env_batch%print_every_batch==0 or env_batch==max_train_batches-1:
                        if out_step%print_every_out_step==0 or out_step==nb_outer_steps-1:
                            print(f"{time.strftime('%H:%M:%S')}      Epoch: {epoch:-3d}      Batch: {env_batch:-3d}      OuterStep: {out_step:-3d}      LossModel: {losses_model[-1]:-.8f}     LossTerm2: {jnp.mean(term2):-.8f}     LossGates: {losses_gates[-1]:-.8f}      Time/Step(s): {time.perf_counter()-start_time_step:-.4f}", flush=True, end="\r")

                            if verbose:
                                print("Contributing envs to the losses: ", loss_contrs, flush=True, end="\n")

                                gate_fun = model.vectorfield.neuralnet.gating_function
                                gate_vals = eqx.filter_vmap(gate_fun)(contexts.params)
                                print("Gate values for each environment (all envs): \n", gate_vals, flush=True, end="\n")

                            if save_checkpoints:
                                ## Save the context and model with the right suffix
                                eqx.tree_serialise_leaves(backup_folder+f"contexts_outstep_{out_step:06d}.eqx", contexts)
                                eqx.tree_serialise_leaves(backup_folder+f"model_outstep_{out_step:06d}.eqx", model)
                                np.savez(backup_folder+"train_histories.npz",
                                    losses_model=jnp.vstack([jnp.vstack(losses_model)]), 
                                    losses_ctx=jnp.vstack([jnp.vstack(losses_ctx)]))

                    if val_dataloader is not None and (out_step != 0 and (out_step%validate_every==0 or out_step==nb_outer_steps-1)):
                        self.learner.model = model
                        self.learner.contexts = contexts
                        self.learner.all_env_losses = all_env_losses

                        self.opt_state_model = opt_state_model
                        self.opt_state_ctx = opt_state_ctx

                        ind_crit,_ = tester.evaluate(val_dataloader,
                                                    criterion_id=val_criterion_id,
                                                    max_adapt_batches=max_val_batches,
                                                    nb_steps=val_nb_steps,
                                                    taylor_order=0, 
                                                    max_ret_env_states=-1,
                                                    stochastic=False,
                                                    verbose=False)
                        print(f"        Validation Criterion: {ind_crit:-.8f}", flush=True)
                        val_losses.append(np.array([out_step, ind_crit]))

                        ## Check if val loss is lowest to save the model
                        if ind_crit <= jnp.stack(val_losses)[:,1].min() and save_path:
                            print(f"        Saving best model so far ...")
                            self.save_trainer(save_path, ignore_losses=False)

                    ###========== Impute 'previous losses' to not seen environments
                    loss_sort = jnp.argsort(all_env_losses)
                    if loss_filling == "NF-iW":
                        all_env_losses = all_env_losses.at[loss_sort[-nb_loss_contr:]].set(jnp.inf)
                    elif loss_filling == "NF-W":
                        all_env_losses = all_env_losses.at[loss_sort[:nb_loss_contr]].set(0.)

        wall_time = time.time() - start_time
        time_in_hmsecs = seconds_to_hours(wall_time)
        print("\nTotal gradient descent training time: %d hours %d mins %d secs" %time_in_hmsecs)

        self.losses_model.append(jnp.vstack(losses_model))
        self.losses_ctx.append(jnp.vstack(losses_ctx))

        if val_dataloader is not None:
            if not hasattr(self, 'val_losses'):
                self.val_losses = []
            self.val_losses.append(jnp.vstack(val_losses))

        self.opt_state_model = opt_state_model
        if val_dataloader is None:
            self.learner.model = model

        ## DO NOT TRUST. Just for visualisation purposes
        self.opt_state_ctx = opt_state_ctx
        self.learner.contexts = contexts

        # Save the model and results
        if save_path:
            self.save_trainer(save_path, ignore_learner=True)








    def meta_train_noalm(self,
                    dataloader: DataLoader, 
                    nb_epochs,
                    nb_outer_steps,
                    max_train_batches=None,
                    print_error_every=1, 
                    validate_every=100,
                    save_path=False, 
                    save_checkpoints=False,
                    val_dataloader=None, 
                    val_criterion_id=None, 
                    max_val_batches=None,
                    val_nb_steps=10,
                    key=None):
        """ Train the model using the proximal gradient descent algorithm (PAM) """

        key = key if key is not None else self.key


        loss_fn = self.learner.loss_fn
        model = self.learner.model
        loss_filling = self.learner.loss_filling
        nb_loss_contr = self.learner.loss_contributors
        # if hasattr(self, 'opt_state'):
        # opt_state_model = self.opt_state_model

        if save_checkpoints:
            backup_folder = save_path+"checkpoints/"
            if not os.path.exists(backup_folder):
                os.makedirs(backup_folder)

        @eqx.filter_jit
        def train_step(mega_model, batch, weightings, opt_state, key):
            print('     ### Compiling function "train_step" for both model and contexts ...  ')

            def mega_loss_fn(mega_model, batch, weightings, key):
                model, contexts = mega_model
                loss, aux_data = loss_fn(model, contexts, batch, weightings, key)
                return loss, aux_data

            (loss, aux_data), grads = eqx.filter_value_and_grad(mega_loss_fn, has_aux=True)(mega_model, batch, weightings, key)

            updates, opt_state = self.opt_model.update(grads, opt_state)
            mega_model = eqx.apply_updates(mega_model, updates)

            return mega_model, opt_state, loss, aux_data

        if val_dataloader is not None:
            tester = VisualTester(self, key=key)

        print(f"\n\n=== Beginning Meta-Training ... ===")
        print(f"    Number of examples in a batch along envs: {dataloader.batch_size}")
        print(f"    Maximum number of batches (along envs): {dataloader.num_batches}")
        print(f"    Total number of epochs: {nb_epochs}")
        print(f"    Number of outer minimizations: {nb_outer_steps}")

        if max_train_batches is None or max_train_batches<1 or max_train_batches>dataloader.num_batches:
            max_train_batches = dataloader.num_batches
        print(f"    Training on {max_train_batches} batches")
        if val_dataloader is not None:
            if max_val_batches is None or max_val_batches<1 or max_val_batches>val_dataloader.num_batches:
                max_val_batches = val_dataloader.num_batches
            print(f"    Validating on {max_val_batches} batches")

        if isinstance(print_error_every, int):
            print_error_every = (print_error_every, print_error_every)
        print_every_batch, print_every_out_step = print_error_every

        start_time = time.time()

        losses = []
        if val_dataloader is not None:
            val_losses = []

        loss_key, _ = jax.random.split(key)

        for epoch in range(nb_epochs):
            # print(f"\nEPOCH {epoch} ... ")

            for env_batch, batch in enumerate(dataloader):
                if env_batch >= max_train_batches:
                    break

                loss_epochs = 0.
                nb_batches = 0

                nb_envs_in_batch = batch[1].shape[0]
                # weightings = jnp.ones(nb_envs_in_batch) / nb_envs_in_batch

                contexts = self.learner.reset_contexts(nb_envs_in_batch)
                mega_model = (model, contexts)
                opt_state = self.opt_model.init(eqx.filter(mega_model, eqx.is_array))

                ## Approach #2
                if loss_filling == "NF-iW":
                    all_env_losses = jnp.inf*jnp.ones(nb_envs_in_batch)
                else:
                    all_env_losses = jnp.zeros(nb_envs_in_batch)

                for out_step in range(nb_outer_steps):
                    # print(f"    Staring outer step {out_step} ...")
                    start_time_step = time.perf_counter()

                    loss_key, _ = jax.random.split(loss_key)

                    mega_model, opt_state, loss, (term1, term2, _, loss_contrs) = train_step(mega_model, batch, all_env_losses, opt_state, loss_key)
                    all_env_losses = all_env_losses.at[loss_contrs].set(term1)

                    losses.append(loss)

                    if env_batch%print_every_batch==0 or env_batch==max_train_batches-1:
                        if out_step%print_every_out_step==0 or out_step==nb_outer_steps-1:
                            print(f"Epoch: {epoch:-3d}      Batch: {env_batch:-3d}      OuterStep: {out_step:-3d}      LossModel: {losses[-1]:-.8f}     ContextsNorm: {jnp.mean(term2):-.8f}      Time/Step(s): {time.perf_counter()-start_time_step:-.4f}", flush=True, end="\r")
                            print(f"Training losses per environment: {all_env_losses.tolist()}", flush=True, end="\n")

                        if save_checkpoints:
                            ## Save the context and model with the right suffix
                            context_save_path = backup_folder+f"contexts_outstep_{out_step:06d}.eqx"
                            eqx.tree_serialise_leaves(context_save_path, contexts)
                            eqx.tree_serialise_leaves(backup_folder+f"model_outstep_{out_step:06d}.eqx", model)
                            np.savez(backup_folder+"train_histories.npz",
                                losses_model=jnp.vstack([jnp.vstack(losses)]),
                                losses_ctx=jnp.vstack([jnp.vstack(losses)]))

                    model, contexts = mega_model
                    if val_dataloader is not None and (out_step != 0 and (out_step%validate_every==0 or out_step==nb_outer_steps-1)):
                        self.learner.model = model
                        self.learner.contexts = contexts

                        ind_crit,_ = tester.evaluate(val_dataloader,
                                                    criterion_id=val_criterion_id,
                                                    max_adapt_batches=max_val_batches,
                                                    nb_steps=val_nb_steps,
                                                    taylor_order=0, 
                                                    max_ret_env_states=self.learner.loss_contributors,
                                                    verbose=False)
                        print(f"        Validation Criterion: {ind_crit:-.8f}", flush=True)
                        val_losses.append(np.array([out_step, ind_crit]))

                        ## Check if val loss is lowest to save the model
                        if ind_crit <= jnp.stack(val_losses)[:,1].min() and save_path:
                            print(f"        Saving best model so far ...")
                            self.learner.save_learner(save_path)
                        ## Restore the learner at the last evaluation step
                        if out_step == nb_outer_steps-1:
                            self.learner.load_learner(save_path)

                    ###========== Approach #2 to only sample relevent environments.
                    loss_sort = jnp.argsort(all_env_losses)
                    if loss_filling == "NF-iW":
                        all_env_losses = all_env_losses.at[loss_sort[-nb_loss_contr:]].set(jnp.inf)
                    else:
                        all_env_losses = all_env_losses.at[loss_sort[:nb_loss_contr]].set(0.)

                loss_epochs += loss
                nb_batches += 1

        wall_time = time.time() - start_time
        time_in_hmsecs = seconds_to_hours(wall_time)
        print("\nTotal gradient descent training time: %d hours %d mins %d secs" %time_in_hmsecs)

        self.losses_model.append(jnp.vstack(losses))
        self.losses_ctx.append(jnp.vstack(losses))

        if val_dataloader is not None:
            if not hasattr(self, 'val_losses'):
                self.val_losses = []
            self.val_losses.append(jnp.vstack(val_losses))

        self.opt_state_model = opt_state[0]
        if val_dataloader is None:
            self.learner.model = model

        ## DO NOT TRUST. Just for visualisation purposes
        self.opt_ctx_state = opt_state[1]
        self.learner.contexts = contexts

        # Save the model and results
        if save_path:
            self.save_trainer(save_path)





    def meta_train_palm(self,
                    dataloader: DataLoader, 
                    nb_epochs,
                    nb_outer_steps,
                    # proximal_betas=(100., 100.), 
                    max_train_batches=None,
                    print_error_every=1,
                    validate_every=100,
                    save_path=False,
                    val_dataloader=None,
                    val_criterion_id=None,
                    max_val_batches=None,
                    val_nb_epochs=10,
                    key=None):
        """ Train the model using the proximal alternating linearized minimisation (PALM) from Driggs et al. 2021 """

        key = key if key is not None else self.key

        loss_fn = self.learner.loss_fn
        model = self.learner.model
        opt_state_model = self.opt_state_model

        def prox_l1(v, lamb):
            """Proximal operator for the L1 norm (soft thresholding), see page 188 of Proximal Algorithms
            by Neal Parikh and Stephen Boyd """
            # return jnp.sign(v) * jnp.maximum(jnp.abs(v) - lamb, 0.)
            v_dyn, v_stat = eqx.partition(v, eqx.is_array)
            new_v = jax.tree_map(lambda x: jnp.sign(x) * jnp.maximum(jnp.abs(x) - lamb, 0.), v_dyn)
            return eqx.combine(new_v, v_stat)

        def prox_l2(v, lamb):
            """Proximal operator for the L2 norm squared (shrinkage), see page 174 of Proximal Algorithms
            by Neal Parikh and Stephen Boyd """
            # return v / (1. + lamb)
            v_dyn, v_stat = eqx.partition(v, eqx.is_array)
            new_v = jax.tree_map(lambda x: x / (1. + lamb), v_dyn)
            return eqx.combine(new_v, v_stat)

        if self.learner.model_reg == 'l1':
            proximal_reg_model = prox_l1
        elif self.learner.model_reg == 'l2':
            proximal_reg_model = prox_l2
        else:
            raise NotImplementedError("Invalid model regularizer. Must be either 'l1' or 'l2' for now.")


        def lipschitz_constant_approx(fn, x, y, aux_inputs, nb_samples=5):
            """Approximate the Lipschitz constant of the function fn(x, y, aux_input) wrt x"""

            # print("This is the x I was given: ", x)
            x_dyn, x_stat = eqx.partition(x, eqx.is_array)
            flat_x, shapes, tree_def = flatten_pytree(x_dyn)

            ## Generate 5 flatttened vectors like flat_x
            keys = jax.random.split(key, nb_samples)
            x_perturb = [flat_x] + [flat_x + 1e-3 * jax.random.normal(k, flat_x.shape) for k in keys[:2]]
            x_perturb += [jax.random.uniform(k, flat_x.shape, minval=flat_x.min(), maxval=flat_x.max()) for k in keys[3:]]
            x_perturb = jnp.vstack(x_perturb)

            def eval_fn(flat_x):
                x_dyn = unflatten_pytree(flat_x, shapes, tree_def)
                x_full = eqx.combine(x_dyn, x_stat)
                return fn(x_full, y, *aux_inputs, key)[0][0]

            ## Evaluate the function at the perturbed points
            # keys = jax.random.split(key, 5)
            outs = jax.vmap(eval_fn)(x_perturb)

            ## Compute the product of all possible pairwise differences
            diffs_outs = jnp.abs(outs[:, None] - outs[None, :])
            diffs_ins = jnp.sum(jnp.abs(x_perturb[:, None] - x_perturb[None, :]), axis=-1)

            ## Compute the Lipschitz constant
            lipschitz_constant = jnp.max(diffs_outs / diffs_ins)

            return lipschitz_constant



        @eqx.filter_jit
        def train_step_model(model, contexts, batch, weightings, opt_state, key):
            print('     ### Compiling function "train_step" for the model ...  ')

            partial_loss_fn = eqx.filter_value_and_grad(loss_fn, has_aux=True)
            (loss, aux_data), grads = partial_loss_fn(model, contexts, batch, weightings, key)

            # ## TODO Approximate the Lipschitz constant of the gradient wrt model
            # l_model = lipschitz_constant_approx(partial_loss_fn, model, contexts, (batch, weightings), nb_samples=5)
            # # print(f"    Lipschitz constant of the gradient wrt model: {l_model:.2e}")
            # jax.debug.print("    Lipschitz constant of the gradient wrt model: {}", l_model)

            ## To perform this update, the learning rate should be 1/L(ctx), where L(ctx) is the Lipschitz constant of the derivative wrt model (See paper Driggs et al. 2021, page 1961)
            updates, opt_state = self.opt_model.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)

            ## For now, let's retrieve whatever learning the optimiser is using
            _, scale_by_schedule_state = opt_state
            learning_rate = self.scheduler_model(scale_by_schedule_state.count)

            # ## TODO Proximal step
            # model = proximal_reg_model(model, learning_rate)

            return model, contexts, opt_state, loss, aux_data

        if self.learner.context_reg == 'l1':
            proximal_reg_ctx = prox_l1
        elif self.learner.context_reg == 'l2':
            proximal_reg_ctx = prox_l2
        else:
            raise NotImplementedError("Invalid context regularizer. Must be either 'l1' or 'l2' for now.")

        @eqx.filter_jit
        def train_step_ctx(model, contexts, batch, weightings, opt_state, key):
            print('     ### Compiling function "train_step" for the contexts ...  ')

            def ctx_loss_fn(contexts, model, batch, weightings, key):
                loss, aux_data = loss_fn(model, contexts, batch, weightings, key)
                return loss, aux_data

            partial_loss_fn = eqx.filter_value_and_grad(ctx_loss_fn, has_aux=True)
            (loss, aux_data), grads = partial_loss_fn(contexts, model, batch, weightings, key)

            # ## TODO Approximate the Lipschitz constant of the gradient wrt contexts
            # l_ctx = lipschitz_constant_approx(partial_loss_fn, contexts, model, (batch, weightings), nb_samples=5)
            # jax.debug.print("    Lipschitz constant of the gradient wrt contexts: {}", l_ctx)

            ## Learning rate should be 1/L
            updates, opt_state = self.opt_ctx.update(grads, opt_state)
            contexts = eqx.apply_updates(contexts, updates)

            ## Let's retrieve whatever learning the optimiser is using
            _, scale_by_schedule_state = opt_state
            learning_rate = self.scheduler_ctx(scale_by_schedule_state.count)

            ## Proximal step    ## TODO This is super hars, it causes the contexts to vanish
            contexts = proximal_reg_ctx(contexts, learning_rate)

            return model, contexts, opt_state, loss, aux_data


        # if not isinstance(dataloader, DataLoader):
        #     raise ValueError("The dataloader must be an instance of DataLoader")
        if val_dataloader is not None:
            tester = VisualTester(self, key=key)

        print(f"\n\n=== Beginning Meta-Training ... ===")
        print(f"    Number of examples in a batch along envs: {dataloader.batch_size}")
        print(f"    Maximum number of batches (along envs): {dataloader.num_batches}")
        print(f"    Total number of epochs: {nb_epochs}")
        print(f"    Number of outer minimizations: {nb_outer_steps}")

        if max_train_batches is None or max_train_batches<1 or max_train_batches>dataloader.num_batches:
            max_train_batches = dataloader.num_batches
        print(f"    Training on {max_train_batches} batches")
        if val_dataloader is not None:
            if max_val_batches is None or max_val_batches<1 or max_val_batches>val_dataloader.num_batches:
                max_val_batches = val_dataloader.num_batches
            print(f"    Validating on {max_val_batches} batches")

        if isinstance(print_error_every, int):
            print_error_every = (print_error_every, print_error_every)
        print_every_batch, print_every_out_step = print_error_every

        start_time = time.time()

        losses_model = []
        losses_ctx = []
        if val_dataloader is not None:
            val_losses = []

        loss_key, _ = jax.random.split(key)
        early_stopping_count = 0

        for epoch in range(nb_epochs):
            # print(f"\nEPOCH {epoch} ... ")

            for env_batch, batch in enumerate(dataloader):
                if env_batch >= max_train_batches:
                    break
                # if env_batch%10==0:
                #     print(f"  Learning on batch {env_batch} ...")

                loss_epochs_model = 0.
                loss_epochs_ctx = 0.
                nb_batches = 0

                nb_envs_in_batch = batch[0].shape[0]
                weightings = jnp.ones(nb_envs_in_batch) / nb_envs_in_batch

                contexts = self.learner.reset_contexts(nb_envs_in_batch)
                opt_state_ctx = self.opt_ctx.init(eqx.filter(contexts, eqx.is_array))

                for out_step in range(nb_outer_steps):
                    # print(f"    Staring outer step {out_step} ...")

                    loss_key, _ = jax.random.split(loss_key)

                    model, contexts, opt_state_model, loss_model, (_, term2, term3) = train_step_model(model, contexts, batch, weightings, opt_state_model, loss_key)


                    loss_key, _ = jax.random.split(loss_key)

                    model, contexts, opt_state_ctx, loss_ctx, (_, term2, term3) = train_step_ctx(model, contexts, batch, weightings, opt_state_ctx, loss_key)

                    losses_model.append(loss_model)
                    losses_ctx.append(loss_ctx)

                    if env_batch%print_every_batch==0 or env_batch==max_train_batches-1:
                        if out_step%print_every_out_step==0 or out_step==nb_outer_steps-1:
                            print(f"Epoch: {epoch:-3d}      Batch: {env_batch:-3d}      OuterStep: {out_step:-3d}      LossModel: {losses_model[-1]:-.8f}     ContextsNorm: {jnp.mean(term2):-.8f}", flush=True, end="\r")

                    if val_dataloader is not None and (out_step != 0 and (out_step%validate_every==0 or out_step==nb_outer_steps-1)):
                        self.learner.model = model
                        self.learner.contexts = contexts
                        # print("Setting contexts in the metatrainer: \n", contexts.params)

                        ind_crit,_ = tester.evaluate(val_dataloader,
                                                    criterion_id=val_criterion_id,
                                                    max_eval_batches=max_val_batches,
                                                    nb_epochs=val_nb_epochs,
                                                    # nb_inner_steps=None,
                                                    taylor_order=0, 
                                                    verbose=False)
                        print(f"        Validation Criterion: {ind_crit:-.8f}", flush=True)
                        val_losses.append(np.array([out_step, ind_crit]))

                        ## Check if val loss is lowest to save the model
                        if ind_crit <= jnp.stack(val_losses)[:,1].min() and save_path:
                            print(f"        Saving best model so far ...")
                            self.learner.save_learner(save_path)
                        ## Restore the learner at the last evaluation step
                        if out_step == nb_outer_steps-1:
                            self.learner.load_learner(save_path)

                loss_epochs_model += loss_model
                loss_epochs_ctx += loss_ctx
                nb_batches += 1


        wall_time = time.time() - start_time
        time_in_hmsecs = seconds_to_hours(wall_time)
        print("\nTotal gradient descent training time: %d hours %d mins %d secs" %time_in_hmsecs)

        self.losses_model.append(jnp.vstack(losses_model))
        self.losses_ctx.append(jnp.vstack(losses_ctx))

        if val_dataloader is not None:
            if not hasattr(self, 'val_losses'):
                self.val_losses = []
            self.val_losses.append(jnp.vstack(val_losses))

        self.opt_state_model = opt_state_model
        if val_dataloader is None:
            self.learner.model = model

        ## DO NOT TRUST. Just for visualisation purposes
        self.opt_ctx_state = opt_state_ctx
        self.learner.contexts = contexts

        # Save the model and results
        if save_path:
            self.save_trainer(save_path)



    def meta_test(self, 
                   dataloader, ## Either a full dataloader or a tuple of batches
                   nb_steps=10, 
                   taylor_order=0,
                   optimizer=None, 
                   print_error_every=(10, 10), 
                   max_adapt_batches=None,
                   val_dataloader=None,
                   max_ret_env_states=None,
                   stochastic=True,
                   verbose=True,
                   save_path=False, 
                   key=None) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, Any]]:
        """Adapt the model to new environments (in bulk) using the provided dataset. """

        key = key if key is not None else self.key

        nb_epochs = nb_steps
        assert nb_epochs > 0, "Number of epochs must be greater than 0."

        if stochastic==False:      ## Use all the adaptation environments 
            loss_fn = self.learner.loss_fn_full
        else:
            loss_fn = self.learner.loss_fn
        # model = self.learner.model

        if val_dataloader is None:
            val_dataloader = dataloader

        if isinstance(print_error_every, int):
            print_error_every = (print_error_every, print_error_every)
        print_every_batch, print_every_epoch = print_error_every

        ## This is useful if we want to disable the taylor expansion
        model = self.learner.reset_model(taylor_order, verbose=verbose)

        if optimizer is None:       ## To continue a previous adaptation
            if hasattr(self, 'opt_ctx'):
                if verbose:
                    print("Using any previously defined optimizer for adapation")
                opt = self.opt_ctx
            else:
                raise ValueError("No optimizer provided for adaptation, and none previously defined")
        else:
            opt = optimizer
            self.losses_adapt = []

        if not hasattr(self, 'losses_adapt'):
            self.losses_adapt = []

        if verbose:
            print(f"\n=== Beginning Meta-Testing ... ===")
            print(f"    Number of examples in a batch along envs: {dataloader.batch_size}")
            print(f"    Maximum number of batches (along envs): {dataloader.num_batches}")

        # if dataloader.num_batches != 1:
        #     raise ValueError("The dataloader must be a single batch of environments for meta-testing with NCF")
        # else:
        #     nb_envs_in_batch = dataloader.batch_size
        #     nb_batches = 1

        if isinstance(dataloader, DataLoader):
            nb_batches = dataloader.nb_batches
        else:
            nb_batches = len(dataloader)    ## A tuple of batches

        if max_ret_env_states is None:
            max_ret_env_states = self.learner.loss_contributors
        elif max_ret_env_states < 0:
            max_ret_env_states = val_dataloader.batch_size

        if max_adapt_batches is None or max_adapt_batches<1 or max_adapt_batches>dataloader.num_batches:
            max_adapt_batches = nb_batches
        else:
            if verbose and not self.learner.reuse_contexts:
                print(f"    Adapting on {max_adapt_batches} batches")

        #################### Shortcut to not recreate contexts (only use this for single batch cases)
        if self.learner.reuse_contexts and not dataloader.dataset.adaptation and dataloader.num_batches==1:
            if verbose:
                print(f"    Reusing contexts for adaptation on the single bach")

            contexts = self.learner.contexts
            batch = next(iter(val_dataloader))
            weightings = jnp.ones(dataloader.batch_size) / dataloader.batch_size

            loss, aux_data = loss_fn(model, contexts, batch, weightings, key)
            state_data = self.learner.batch_predict(model, contexts, batch, max_envs=max_ret_env_states)

            self.learner.contexts_latest = contexts
            return jnp.stack(aux_data, axis=1), contexts, state_data
        ####################

        def prox_loss_fn(contexts, model, batch, weightings, key):
            loss, aux_data = loss_fn(model, contexts, batch, weightings, key)
            return loss, aux_data

        @eqx.filter_jit
        def adapt_step(model, contexts, batch, weightings, opt_state, key):
            # print('     ### Compiling function "adapt_step" for the contexts ...  ')

            (loss, aux_data), grads = eqx.filter_value_and_grad(prox_loss_fn, has_aux=True)(contexts, model, batch, weightings, key)

            updates, opt_state = opt.update(grads, opt_state)
            contexts = eqx.apply_updates(contexts, updates)

            return model, contexts, opt_state, loss, aux_data

        start_time = time.time()

        losses = []
        state_data = [[], [], []]
        loss_key, _ = jax.random.split(key)

        torch.manual_seed(loss_key[0])  # Ensure the same shuffling order
        # np.random.seed(loss_key[0])

        for env_batch, (batch, val_batch) in enumerate(zip(dataloader, val_dataloader)):
            if env_batch >= max_adapt_batches:
                break

            nb_envs_in_batch = batch[1].shape[0]
            # # weightings = jnp.ones(nb_envs_in_batch) / nb_envs_in_batch
            if hasattr(self.learner, 'all_env_losses'):
                all_env_losses = self.learner.all_env_losses
            else:
                if self.learner.loss_filling == "NF-iW":
                    all_env_losses = 1e-2*jnp.ones(nb_envs_in_batch)
                else:
                    all_env_losses = 10*jnp.ones(nb_envs_in_batch)
            # min_loss_weight = 1e-3

            contexts = self.learner.reset_contexts(nb_envs_in_batch)
            opt_state_ctx = opt.init(eqx.filter(contexts, eqx.is_array))

            losses_epoch = []

            for epoch in range(nb_epochs):

                loss_key, _ = jax.random.split(loss_key)

                model, contexts, opt_state_ctx, loss_ctx, (term1, term2, term3, loss_contrs) = adapt_step(model, contexts, batch, all_env_losses, opt_state_ctx, loss_key)

                # all_env_losses = all_env_losses.at[loss_contrs].set(loss_ctx)
                # all_env_losses = jnp.clip(all_env_losses, min=all_env_losses.min()+min_loss_weight, 
                #                             max=all_env_losses.min()+min_loss_weight+10*min_loss_weight)
                losses.append(loss_ctx)

                mean_loss_terms = [jnp.mean(term) for term in (term1, term2, term3)]
                losses_epoch.append(jnp.stack([loss_ctx]+mean_loss_terms))

                if epoch == nb_epochs-1:
                    ## Use the contexts and the val_batch to predict Y_hat
                    state_data_ = self.learner.batch_predict(model, contexts, val_batch, max_envs=max_ret_env_states)
                    [state_data[i].append(state_data_[i]) for i in range(3)]

                if verbose and (epoch%print_every_epoch==0 or epoch<=3 or epoch==nb_epochs-1):
                    print(f"Epoch: {epoch:-3d}      Batch: {env_batch:-3d}      Loss: {losses[-1]:-.8f}     ContextsNorm: {jnp.mean(term2):-.8f}", flush=True, end="\n")

            losses_epochs = jnp.stack(losses_epoch, axis=0)

        wall_time = time.time() - start_time
        time_in_hmsecs = seconds_to_hours(wall_time)
        if verbose:
            print("\nTotal gradient descent training time: %d hours %d mins %d secs" %time_in_hmsecs)

        losses = jnp.vstack(losses)
        self.losses_adapt.append(losses)

        ## DO NOT TRUST. Just for visualisation purposes
        if isinstance(dataloader, DataLoader) and dataloader.dataset.adaptation:
            self.learner.contexts_adapt = contexts
        else:      ## Dealing with a list or generator of batches
            self.learner.contexts_latest = contexts

        if save_path:
            self.save_adapted_trainer(save_path)

        # state_data = tuple(jnp.concat(state_data[i], axis=0) for i in range(3))
        # state_data = tuple(jax.tree_map(lambda x: jnp.concat(x, axis=0), state_data))

        # state_data = [x[0] for x in state_data]
        # state_data = jax.tree_map(lambda x: jnp.concat(x, axis=0), state_data)
        state_data = tuple([state_data[i][0] for i in range(3)])

        return losses_epochs, contexts, state_data




























class CAVIATrainer(Trainer):
    def __init__(self, learner:Learner, optimisers, key=None):
        """ Trainer class for the CAVIA algorithm """
        super().__init__(learner, optimisers, key)


    def meta_train(self,
                    dataloader: DataLoader, 
                    nb_outer_steps,
                    nb_inner_steps=10,
                    print_error_every=(1, 1), 
                    save_path=False, 
                    save_checkpoints=False,
                    max_train_batches=None,
                    val_dataloader=None, 
                    val_criterion_id=None, 
                    max_val_batches=None,
                    validate_every=1,
                    key=None):
        """ Train the model using the MAML/CAVIA gradient descent algorithm """

        key = key if key is not None else self.key

        model = self.learner.model
        opt_state_model = self.opt_state_model

        ## 
        if save_checkpoints:
            backup_folder = save_path+"checkpoints/"
            if not os.path.exists(backup_folder):
                os.makedirs(backup_folder)

        def inner_train_step(model, contexts, batch, weightings, opt_state, key):
            print(f'     ### (Re)Compiling function: {inner_train_step.__name__} ...  ')

            nb_envs = contexts.params.shape[0]

            env_loss_fn_ = lambda ctx, model, batch, ctxs, key: self.learner.env_loss_fn(model, batch, ctx, ctxs, key)

            ctx_grad_fn = eqx.filter_value_and_grad(env_loss_fn_, has_aux=True)

            # @eqx.filter_jit
            def step(contexts, model, batch, opt_state, key):

                keys = jax.random.split(key, num=nb_envs)

                (loss, aux_data), grads = eqx.filter_vmap(ctx_grad_fn, in_axes=(0, None, 0, None, 0))(contexts.params, model, batch, contexts.params, keys)

                ### ===== Optimizer approach
                grads_pytree = eqx.tree_at(lambda ptree: ptree.params, contexts, grads)
                updates, opt_state = self.opt_ctx.update(eqx.filter(grads_pytree, eqx.is_array), opt_state)
                contexts = eqx.apply_updates(contexts, updates)
                ### =====

                # #### ===== Simple update rule approach
                # new_params = contexts.params - 0.1*grads
                # contexts = eqx.tree_at(lambda ptree: ptree.params, contexts, new_params)
                # #### =====

                return contexts, opt_state, loss, aux_data

            keys = jax.random.split(key, num=nb_inner_steps)

            ####### Use the simple update rule  #######
            # for i in range(nb_inner_steps):
            #     contexts, opt_state, loss, aux_data = step(contexts, model, batch, opt_state, keys[i])
            ##########################################


            # ####### Use the scan algorithm  #######
            def body_func(carry, key):
                contexts, opt_state = carry
                contexts = eqx.combine(contexts, contexts_stat)

                contexts, opt_state, _, aux_data = step(contexts, model, batch, opt_state, key)

                contexts, _ = eqx.partition(contexts, eqx.is_array)
                return (contexts, opt_state), aux_data

            contexts_dyn, contexts_stat = eqx.partition(contexts, eqx.is_array)
            init_carry = (contexts_dyn, opt_state)
            (contexts_dyn, opt_state), aux_datas = jax.lax.scan(body_func, init_carry, keys)
            contexts = eqx.combine(contexts_dyn, contexts_stat)

            aux_data = [jnp.mean(term) for term in aux_datas]
            # ##########################################

            meta_loss = self.learner.loss_fn(model, contexts, batch, weightings, key)[0]

            return meta_loss, (contexts, opt_state, None, aux_data)



        @eqx.filter_jit
        def outer_train_step(model, contexts, batch, weightings, opt_states, key):
            print(f'     ### (Re)Compiling function: {outer_train_step.__name__} ...  ')

            opt_state_model, opt_state_ctx = opt_states

            (loss, aux_data), grads = eqx.filter_value_and_grad(inner_train_step, has_aux=True)(model, contexts, batch, weightings, opt_state_ctx, key)

            updates, opt_state_model = self.opt_model.update(grads, opt_state_model)
            model = eqx.apply_updates(model, updates)

            new_contexts = aux_data[0]
            opt_states = (opt_state_model, aux_data[1])
            other_loss_terms = aux_data[-1]

            return model, new_contexts, opt_states, loss, other_loss_terms

        if val_dataloader is not None:
            tester = VisualTester(self, key=key)
        validate_every = validate_every if validate_every > 0 else 1

        print(f"\n\n=== Beginning Meta-Training ... ===")
        print(f"    Number of examples in a batch: {dataloader.batch_size}")
        print(f"    Total number of batches : {dataloader.num_batches}")
        print(f"    Number of outer steps: {nb_outer_steps}")
        print(f"    Numbers of inner steps : {nb_inner_steps}")

        if max_train_batches is None or max_train_batches<1 or max_train_batches>dataloader.num_batches:
            max_train_batches = dataloader.num_batches
        print(f"    Training on {max_train_batches} batches")
        if val_dataloader is not None:
            if max_val_batches is None or max_val_batches<1 or max_val_batches>val_dataloader.num_batches:
                max_val_batches = val_dataloader.num_batches
            print(f"    Validating on {max_val_batches} batches")

        if isinstance(print_error_every, int):
            print_error_every = (print_error_every, print_error_every)
        print_every_epoch, print_every_batch = print_error_every

        start_time = time.time()

        losses = []

        if val_dataloader is not None:
            val_losses = []

        loss_key, _ = jax.random.split(key)

        step = 0

        nb_epochs = nb_outer_steps
        for epoch in range(nb_epochs):

            loss_epoch = 0.
            nb_batches = 0

            for env_batch, batch in enumerate(dataloader):
                if env_batch >= max_train_batches:
                    break
                start_time_step = time.perf_counter()

                nb_envs_in_batch = batch[0].shape[0]
                weightings = jnp.ones(nb_envs_in_batch) / nb_envs_in_batch

                ## Reset the context and the optimizer
                contexts = self.learner.reset_contexts(nb_envs_in_batch)
                opt_state_ctx = self.opt_ctx.init(eqx.filter(contexts, eqx.is_array))

                loss_key, _ = jax.random.split(loss_key)
                opt_states = (opt_state_model, opt_state_ctx)
                model, contexts, opt_states, loss, (_, term2, _, _) = outer_train_step(model, contexts, batch, weightings, opt_states, loss_key)

                opt_state_model, _ = opt_states

                loss_epoch += loss
                nb_batches += 1
                step += 1

                losses.append(loss)

                # print("All loss terms: ", term1, term2, term3)

                if epoch%print_every_epoch==0 or epoch==nb_epochs-1:
                    if env_batch%print_every_batch==0 or env_batch==max_train_batches-1:
                        # print(f"Epoch: {epoch:-3d}      Batch: {env_batch:-3d}    Loss: {losses[-1]:-.8f}     ContextsNorm: {jnp.mean(term2):-.8f}      Time/Step(s): {time.perf_counter()-start_time_step:-.4f}        Current time (hms)", flush=True, end="\n")
                        print(f"{time.strftime('%H:%M:%S')}   Epoch: {epoch:-3d}      Batch: {env_batch:-3d}    Loss: {losses[-1]:-.8f}     ContextsNorm: {jnp.mean(term2):-.8f}      Time/Step(s): {time.perf_counter()-start_time_step:-.4f}", flush=True, end="\n")

                        # alpha = model.taylor_weight[0]
                        # print(f"Current unnormalised weight of the taylor expansion: {alpha:-.8f}       NormalisedWeight: {jax.nn.sigmoid(model.taylor_scale*alpha):-.8f}", flush=True, end="\r")
                        # print()

                        if save_checkpoints and epoch==nb_epochs-1:
                            ## Save the context's numpy array with the suffix of the current batch*epoch
                            context_save_path = backup_folder+f"contexts_epoch{epoch:04d}_batch{env_batch:06d}.npy"
                            np.save(context_save_path, contexts.params)

                            ## Save the model as well
                            eqx.tree_serialise_leaves(backup_folder+"model.eqx", model)

            # if epoch==nb_epochs-1 and hasattr(self.learner.model, 'taylor_weight'):
            #     alpha = model.taylor_weight[0]
            #     print(f"Current unnormalised weight of the taylor expansion: {alpha:-.8f}       NormalisedWeight: {jax.nn.sigmoid(model.taylor_scale*alpha):-.8f}", flush=True, end="\n")
            #     print()

            if val_dataloader is not None and (epoch!=0 and (epoch%validate_every==0 or epoch==nb_epochs-1)):
                self.learner.model = model
                self.learner.contexts = contexts

                ind_crit,_ = tester.evaluate(dataloader,
                                            criterion_id=val_criterion_id,
                                            max_adapt_batches=max_val_batches,
                                            nb_steps=nb_inner_steps,
                                            val_dataloader=val_dataloader,
                                            taylor_order=0, 
                                            verbose=False)
                print(f"        Validation Criterion: {ind_crit:-.8f}", flush=True, end="\n")
                val_losses.append(np.array([step, ind_crit]))

                # ## TODO Make a visualisation and save (like Zintgraff)
                # train_XY = dataloader.sample_environments(key, 0, 1)
                # val_XY = val_dataloader.sample_environments(key, 0, 1)
                # batch = (batch for batch in [train_XY, val_XY])
                # tester.visualizeTrainVal(batch, save_path=save_path, key=key)

                ## Check if val loss is lowest to save the model
                if ind_crit <= jnp.stack(val_losses)[:,1].min() and save_path:
                    print(f"        Saving best model so far ...", end="\n", flush=True)
                    self.learner.save_learner(save_path)
                ## Restore the learner at the last evaluation step
                if epoch == nb_epochs-1:
                    self.learner.load_learner(save_path)

            loss_epoch /= nb_batches


        wall_time = time.time() - start_time
        time_in_hmsecs = seconds_to_hours(wall_time)
        print("\nTotal gradient descent training time: %d hours %d mins %d secs" %time_in_hmsecs)

        self.losses_model.append(jnp.vstack(losses))
        self.losses_ctx.append(jnp.vstack(losses))          ## TODO: Wrong, just for quick prototyping !

        if val_dataloader is not None:
            if not hasattr(self, 'val_losses'):
                self.val_losses = []
            self.val_losses.append(jnp.vstack(val_losses))

        self.opt_state_model = opt_state_model
        if val_dataloader is None:
            self.learner.model = model

        ## DO NOT TRUST. Mostly for visualisation purposes
        self.opt_ctx_state = opt_state_ctx
        self.learner.contexts = contexts

        # Save the model and results
        if save_path:
            self.save_trainer(save_path)


    def meta_test(self, 
                   dataloader,
                   nb_steps=10,        ## Number of inner gradient update steps
                   taylor_order=0,
                   optimizer=None, 
                   print_error_every=(1, 1), 
                   max_adapt_batches=None,
                   val_dataloader=None,
                   max_ret_env_states=None,
                   stochastic=True,
                   verbose=True,
                   save_path=False, 
                   key=None) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, Any]]:
        """Adapt the model to new environments (in bulk) using the provided dataset. """

        key = key if key is not None else self.key

        nb_inner_steps = nb_steps
        if val_dataloader is None:
            val_dataloader = dataloader

        env_loss_fn = self.learner.env_loss_fn

        ## This is useful if we want to disable the taylor expansion
        model = self.learner.reset_model(taylor_order, verbose=verbose)

        if optimizer is None:       ## To continue a previous adaptation
            if hasattr(self, 'opt_ctx'):
                if verbose:
                    print("Using any previrouly defined optimizer for adapation")
                opt = self.opt_ctx
            else:
                raise ValueError("No optimizer provided for adaptation, and none previously defined")
        else:
            opt = optimizer
            self.losses_adapt = []

        @eqx.filter_jit
        def adapt_step_cavia(model, contexts, batch, weightings, opt_state, key):
            print(f'     ### (Re)Compiling function: {adapt_step_cavia.__name__} ...  ')

            nb_envs, context_size = contexts.params.shape

            env_loss_fn_ = lambda ctx, model, batch, ctxs, key: env_loss_fn(model, batch, ctx, ctxs, key)

            ctx_grad_fn = eqx.filter_value_and_grad(env_loss_fn_, has_aux=True)
            keys = jax.random.split(key, num=nb_envs)
            (loss, aux_data), grads = eqx.filter_vmap(ctx_grad_fn, in_axes=(0, None, 0, None, 0))(contexts.params, model, batch, contexts.params, keys)

            #### ===== Optimizer approach
            grads_pytree = eqx.tree_at(lambda ptree: ptree.params, contexts, grads)
            updates, opt_state = opt.update(eqx.filter(grads_pytree, eqx.is_array), opt_state)
            contexts = eqx.apply_updates(contexts, updates)
            #### =====

            # #### ===== Simple update rule approach
            # new_params = contexts.params - 0.1*grads
            # contexts = eqx.tree_at(lambda ptree: ptree.params, contexts, new_params)
            # #### =====

            return model, contexts, opt_state, jnp.mean(loss), aux_data

        if isinstance(dataloader, DataLoader):
            nb_batches = dataloader.nb_batches
        else:
            nb_batches = len(dataloader)    ## A tuple of batches

        if max_ret_env_states is None:
            max_ret_env_states = self.learner.loss_contributors

        if verbose:
            print(f"\n=== Beginning Meta-Testing ... ===")
            print(f"    Number of environment batches: {nb_batches}")
            print(f"    Number of envs train steps per batch: {nb_inner_steps}")
            print(f"    Total number of training steps: {nb_batches*nb_inner_steps}")
        if max_adapt_batches is None or max_adapt_batches<1 or max_adapt_batches>nb_batches:
            max_adapt_batches = nb_batches
        else:
            if verbose:
                print(f"    Adapting on {max_adapt_batches} batches")


        #################### Shortcut to not recreate contexts (only use this for single batch cases)
        if self.learner.reuse_contexts and not dataloader.dataset.adaptation and dataloader.num_batches==1:
            if verbose:
                print(f"    Reusing contexts for adaptation on the single bach")

            contexts = self.learner.contexts
            batch = next(iter(val_dataloader))
            weightings = jnp.ones(dataloader.batch_size) / dataloader.batch_size

            loss, aux_data = self.learner.loss_fn_full(model, contexts, batch, weightings, key)
            state_data = self.learner.batch_predict(model, contexts, batch, max_envs=max_ret_env_states)

            return jnp.stack(aux_data, axis=1), contexts, state_data
        ####################

        if isinstance(print_error_every, int):
            print_error_every = (print_error_every, print_error_every)
        print_every_epoch, print_every_batch = print_error_every

        start_time = time.time()

        losses = []
        loss_key, _ = jax.random.split(key)
        state_data = [[], [], []]
        # all_contexts = []

        torch.manual_seed(key[0])  # Ensure the same shuffling order
        # for env_batch, batch in enumerate(dataloader):
        for env_batch, (batch, val_batch) in enumerate(zip(dataloader, val_dataloader)):
            if env_batch >= max_adapt_batches:
                break

            nb_envs_in_batch = batch[0].shape[0]
            weightings = jnp.ones(nb_envs_in_batch) / nb_envs_in_batch

            contexts = self.learner.reset_contexts(nb_envs_in_batch)
            opt_state = opt.init(contexts)

            for inner_step in range(nb_inner_steps):
                loss_key, _ = jax.random.split(loss_key)

                # model, contexts, opt_state, loss, aux_losses = adapt_step_proxi(model, contexts, batch, weightings, opt_state, opt, self.learner.loss_fn, loss_key)

                model, contexts, opt_state, loss, aux_losses = adapt_step_cavia(model, contexts, batch, weightings, opt_state, loss_key)

                mean_loss_terms = [jnp.mean(term) for term in aux_losses]
                losses.append(jnp.stack([loss]+mean_loss_terms))

            if verbose and (env_batch%print_every_epoch==0 or env_batch<=3 or env_batch==max_adapt_batches-1):
                print(f"    Batch: {env_batch:-3d}     Loss: {loss:-.8f}        OtherNorms: {jnp.stack(mean_loss_terms)}", flush=True, end="\r")

            ## Use the contexts and the val_batch to predict Y_hat
            state_data_ = self.learner.batch_predict(model, contexts, val_batch, max_envs=max_ret_env_states)
            [state_data[i].append(state_data_[i]) for i in range(3)]
            # all_contexts.append(contexts)

        wall_time = time.time() - start_time
        time_in_hmsecs = seconds_to_hours(wall_time)
        if verbose:
            print("\nTotal gradient descent adaptation time: %d hours %d mins %d secs" %time_in_hmsecs)

        losses = jnp.vstack(losses)
        if not hasattr(self, 'losses_adapt'):
            self.losses_adapt = []
        self.losses_adapt.append(losses)

        ## DO NOT TRUST. Mostly for visualisation purposes
        if isinstance(dataloader, DataLoader) and dataloader.dataset.adaptation:
            self.learner.contexts_adapt = contexts
        else:      ## Dealing with a list or generator of batches
            self.learner.contexts_latest = contexts

        if save_path:
            self.save_adapted_trainer(save_path)

        # ## Use the contexts and the batch to predict Y_hat
        # for batch in val_dataloader:
        #     pass        ## Batch is the last batch from val_dataloader
        # state_data = self.learner.batch_predict(model, contexts, batch)

        state_data = tuple(jnp.concat(state_data[i], axis=0) for i in range(3))
        # all_contexts = jnp.concatenate(all_contexts, axis=0)

        return losses, contexts, state_data











      