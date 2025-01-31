from ._utils import *
from math import factorial
from jax.experimental.jet import jet



class Learner:
    def __init__(self, 
                 model, 
                 env_loss_fn, 
                 context_size, 
                 context_pool_size, 
                 pool_filling="NF", 
                 contexts=None, 
                 reuse_contexts=False,
                 loss_filling="NF", 
                 model_reg="l2",
                 context_reg="l1",
                 self_reweighting=True,
                 loss_contributors=-1,
                 key=None):
        if key is None:
            raise ValueError("You must provide a key for the learner.")
        self.key = key

        self.model = model
        self.context_size = context_size
        self.context_pool_size = context_pool_size
        self.pool_filling = pool_filling
        self.reuse_contexts = reuse_contexts
        self.loss_contributors = loss_contributors

        self.model_reg = model_reg
        self.context_reg = context_reg
        self.loss_filling = loss_filling

        self.self_reweighting = self_reweighting

        if contexts is not None:
            self.contexts = contexts
        else:
            print("    No context template provides, using arrays ...")
            self.contexts = ArrayContextParams(nb_envs=1, context_size=context_size)

        ## A loss function for the gates
        def loss_fn_gates(model, ctxs, key):
            print("    Compiling gating loss function - coefficient of variation ...")
            batched_gates = eqx.filter_vmap(model.gating_function)(ctxs.params)

            importances = jnp.mean(batched_gates, axis=0)
            coef_var = jnp.var(importances) / jnp.mean(importances)**2

            return coef_var, (batched_gates, )

        self.loss_fn_gates = loss_fn_gates

        def env_loss_fn_(model, batch, ctx, ctxs, key):
            """ Wrapping the loss function before vectorizing it below """
            X, Y = batch

            if self.pool_filling=="RA":         ## Randomly fill the context pool
                ind = jax.random.permutation(key, ctxs.shape[0])[:self.context_pool_size]
                ctx_pool = ctxs[ind, :]
            elif self.pool_filling=="NF":       ## Fill the context with the nearest first
                dists = jnp.mean(jnp.abs(ctxs-ctx), axis=1)
                # dists = jnp.mean((ctxs-ctx)**2, axis=1)     ## TODO test with L2 norm
                ind = jnp.argsort(dists)[:self.context_pool_size]
                ctx_pool = ctxs[ind, :]
            elif self.pool_filling=="NF*":      ## Same as NF, but excluding the current context
                dists = jnp.mean(jnp.abs(ctxs-ctx), axis=1)
                ind = jnp.argsort(dists)[1:self.context_pool_size+1]
                ctx_pool = ctxs[ind, :]
            elif self.pool_filling=="SF":       ## Smallest contexts first
                dists = jnp.mean(jnp.abs(ctxs), axis=1)
                ind = jnp.argsort(dists)[:self.context_pool_size]
                ctx_pool = ctxs[ind, :]
            else:
                raise ValueError("Invalid pool filling strategy provided. Use one of 'RA', 'NF', 'NF*', 'SF'.")

            Y_hat = jax.vmap(model, in_axes=(None, None, 0))(X, ctx, ctx_pool)
            Y_new = jnp.broadcast_to(Y, Y_hat.shape)

            return env_loss_fn(model, ctx, Y_new, Y_hat)

        ## Full loss function, using all environments
        def loss_fn_full(model, contexts, batch, weightings, key):
            keys = jax.random.split(key, num=contexts.params.shape[0])

            losses, (term1, terms2, terms3) = jax.vmap(env_loss_fn_, in_axes=(None, 0, 0, None, 0))(model, batch, contexts.params, contexts.params, keys)
            base_loss = jnp.mean(losses)

            return base_loss, (term1, terms2, terms3, np.arange(contexts.params.shape[0]))

        if loss_contributors > 0:
            print(f"\nUsing {loss_contributors} environments to estimate the global training loss function ...")

            def select_indices(loss_filling, contexts, prev_losses, key):
                """ Select the indices of the contexts to use for the loss function, based on the pool-filling strategy """
                if loss_filling=="RA":         ## Randomly pick contributors to the loss function
                    indices = jax.random.permutation(key, contexts.params.shape[0])[:loss_contributors]
                elif loss_filling=="FO":       ## Pick the first environments, based on their loss (no randomness at all)
                    indices = jnp.arange(loss_contributors)
                elif loss_filling=="NF":       ## Pick one at random and then the nearest to it
                    rnd_env = jax.random.randint(key, (1,), 0, contexts.params.shape[0])[0]
                    dists = jnp.mean(jnp.abs(contexts.params-contexts.params[rnd_env]), axis=1)
                    # dists = jnp.mean((contexts.params-contexts.params[rnd_env])**2, axis=1)    ## TODO test with L2 norm
                    indices = jnp.argsort(dists)[:loss_contributors]
                elif loss_filling=="NF-W":       ## Weighted. We Pick one of the environments we want to focus on
                    probas = prev_losses / jnp.sum(prev_losses)
                    rnd_env = jax.random.choice(key, a=contexts.params.shape[0], shape=(1,), p=probas**1)[0]
                    dists = jnp.mean(jnp.abs(contexts.params-contexts.params[rnd_env]), axis=1)
                    indices = jnp.argsort(dists)[:loss_contributors]
                elif loss_filling=="NF-iW":       ## inversely Weighted.
                    inv_losses = 1/prev_losses
                    probas = inv_losses / jnp.sum(inv_losses)
                    # jax.debug.print("These are the probabilities:  {}  ", probas)
                    rnd_env = jax.random.choice(key, a=contexts.params.shape[0], shape=(1,), p=probas)[0]
                    dists = jnp.mean(jnp.abs(contexts.params-contexts.params[rnd_env]), axis=1)
                    indices = jnp.argsort(dists)[:loss_contributors]
                    # jax.debug.print("These are the indices:  {}  ", indices)
                elif loss_filling=="NF-B":       ## Biggest lost is picked up !
                    rnd_env = jnp.argmax(prev_losses)
                    dists = jnp.mean(jnp.abs(contexts.params-contexts.params[rnd_env]), axis=1)
                    indices = jnp.argsort(dists)[:loss_contributors]
                else:
                    raise ValueError("Invalid loss filling strategy provided. Use one of 'RA', 'NF'.")

                return indices


            def loss_fn(model, contexts, batch, prev_losses, key):
                keys = jax.random.split(key, num=loss_contributors)
                indices = select_indices(self.loss_filling, contexts, prev_losses, key)

                random_contexts = contexts.params[indices, :]

                ## the full batch is now a pytree, the input is a tuple itself
                random_batch = jax.tree_map(lambda x: x[indices], batch)

                losses, (term1, terms2, terms3) = jax.vmap(env_loss_fn_, in_axes=(None, 0, 0, None, 0))(model, random_batch, random_contexts, random_contexts, keys)

                ## Let's assign weights based on the loss values
                if self.self_reweighting:
                    weightings = jax.nn.softmax(losses / 1.0)
                else:
                    weightings = jnp.ones(indices.shape) / indices.shape[0]
                base_loss = jnp.sum(weightings * losses)

                return base_loss, (term1, terms2, terms3, indices)



            ## Loss function for each expert treated individually
            def env_loss_fn_multitask_(model, batch, ctx, ctxs, key):
                """ Wrapping the env loss function without CSM, for each expert individualy """
                X, Y = batch

                new_model = self.reset_model_expert(model, model.vectorfield.neuralnet.experts[0])    ## Reset the expert model without CSM

                expert_losses = []
                nb_experts = len(model.vectorfield.neuralnet.experts)
                if model.vectorfield.neuralnet.split_context:
                    ctx_pieces = jnp.split(ctx, nb_experts, axis=0)

                for i, expert in enumerate(model.vectorfield.neuralnet.experts):
                    if model.vectorfield.neuralnet.split_context:
                        ctx_i = ctx_pieces[i]
                    else:
                        ctx_i = ctx

                    ## Surgery on new_model to replace the expert
                    new_model = eqx.tree_at(lambda m: m.vectorfield.neuralnet, new_model, expert)

                    Y_hat = new_model(X, ctx_i, ctx_i)
                    Y_new = jnp.broadcast_to(Y, Y_hat.shape)

                    loss, _ = env_loss_fn(expert, ctx_i, Y_hat, Y_new)
                    expert_losses.append(loss)

                expert_losses = jnp.array(expert_losses)

                return jnp.mean(expert_losses), (expert_losses, )


            @eqx.filter_jit
            def loss_fn_multitask(model, contexts, batch, key):
                """ This loss computes the loss function for each expert invidually, and then combines them """
                print("     ### Compiling function 'loss_fn_multitask' for the experts  ...")

                ## Let's use all the environments for each expert
                indices = jnp.arange(contexts.params.shape[0])
                random_contexts = contexts.params[indices, :]

                ## the full batch is now a pytree, the input is a tuple itself
                random_batch = jax.tree_map(lambda x: x[indices], batch)

                # keys = keys[indices]
                keys = jax.random.split(key, num=indices.shape[0])

                losses, (expert_losses, ) = jax.vmap(env_loss_fn_multitask_, in_axes=(None, 0, 0, None, 0))(model, random_batch, random_contexts, random_contexts, keys)

                weightings = jnp.arange(indices.shape[0]) / indices.shape[0]
                mean_loss = jnp.sum(weightings * losses)

                return mean_loss, (expert_losses, indices)      ## Expert losses of shape (nb_experts, nb_envs)

        else:
            print("    Using all environments to estimate the global training loss function ...")
            loss_fn = loss_fn_full

        self.loss_fn = loss_fn                  ## Meta loss function
        self.loss_fn_full = loss_fn_full        ## Base loss function in full
        self.env_loss_fn = env_loss_fn_         ## Base loss function

        self.env_loss_fn_multitask = env_loss_fn_multitask_
        self.loss_fn_multitask = loss_fn_multitask


    def save_learner(self, path):
        assert path[-1] == "/", "ERROR: Invalid path provided. The path must end with /"
        eqx.tree_serialise_leaves(path+"model.eqx", self.model)
        if hasattr(self, "contexts"):
            eqx.tree_serialise_leaves(path+"contexts.eqx", self.contexts)

    def load_learner(self, path):
        assert path[-1] == "/", "ERROR: Invalidn parovided. The path must end with /"
        self.model = eqx.tree_deserialise_leaves(path+"model.eqx", self.model)
        if os.path.exists(path+"contexts.eqx") and hasattr(self, "contexts"):
            self.contexts = eqx.tree_deserialise_leaves(path+"contexts.eqx", self.contexts)



    def reset_model(self, taylor_order, verbose=True):
        if hasattr(self.model, "taylor_order") and taylor_order==self.model.taylor_order:
            model = self.model
        else:
            if verbose:
                print(f"    Creating a new model with taylor order {taylor_order} ...")
            if isinstance(self.model, NeuralContextFlow):
                model = NeuralContextFlow(neuralnet=self.model.neuralnet, 
                                            taylor_order=taylor_order)
            elif isinstance(self.model, NeuralNeuralContextFlow):
                if taylor_order != 0:
                    model = NeuralNeuralContextFlow(neuralnet=self.model.neuralnet, 
                                                    flownet=self.model.flownet)
                else:
                    model = NeuralNeuralContextFlow(neuralnet=self.model.neuralnet, 
                                                    flownet=None)
            elif isinstance(self.model, NeuralContextFlowAdaptiveTaylor):
                model = NeuralContextFlow(neuralnet=self.model.neuralnet, 
                                            taylor_order=taylor_order,
                                            taylor_scale=self.model.taylor_scale,
                                            taylor_weight_init=self.model.taylor_weight[0])
            elif isinstance(self.model, NeuralODE):
                model = NeuralODE(neuralnet=self.model.vectorfield.neuralnet, 
                                    taylor_order=taylor_order,
                                    taylor_ad_mode=self.model.taylor_ad_mode, 
                                    ivp_args=self.model.ivp_args,
                                    t_eval=self.model.t_eval)
            elif isinstance(self.model, NeuralCDE):
                model = NeuralCDE(neuralnet=self.model.vectorfield.neuralnet, 
                                    taylor_order=taylor_order,
                                    taylor_ad_mode=self.model.taylor_ad_mode, 
                                    ivp_args=self.model.ivp_args,
                                    encoder=self.model.encoder,
                                    decoder=self.model.decoder)
            elif isinstance(self.model, TFNeuralODE):
                model = TFNeuralODE(neuralnet=self.model.vectorfield.neuralnet, 
                                    taylor_order=taylor_order,
                                    taylor_ad_mode=self.model.taylor_ad_mode, 
                                    ivp_args=self.model.ivp_args)
            elif isinstance(self.model, BatchedNeuralContextFlow):
                if hasattr(self.model, "taylor_scale"):
                    model = BatchedNeuralContextFlow(neuralnet=self.model.neuralnet, 
                                                    taylor_order=taylor_order,
                                                    taylor_scale=self.model.taylor_scale,
                                                    taylor_weight_init=self.model.taylor_weight[0])
                else:
                    model = BatchedNeuralContextFlow(neuralnet=self.model.neuralnet, 
                                                    taylor_order=taylor_order)
            elif isinstance(self.model, DirectMapping):
                model = DirectMapping(neuralnet=self.model.vectorfield.neuralnet,
                                    taylor_order=self.model.taylor_order)
            else:
                raise ValueError("The model type is not supported")
        return model



    def reset_model_expert(self, model, expert):
        """ Reset the model to use a specific expert """
        if isinstance(model, NeuralODE):        ## TODO also check that its neural net is a MoE
            new_model = NeuralODE(neuralnet=expert, 
                            taylor_order=0,
                            taylor_ad_mode=model.taylor_ad_mode, 
                            ivp_args=model.ivp_args,
                            t_eval=model.t_eval)
        elif isinstance(model, NeuralCDE):
            new_model = NeuralCDE(neuralnet=expert, 
                            taylor_order=0,
                            taylor_ad_mode=model.taylor_ad_mode, 
                            ivp_args=model.ivp_args,
                            encoder=model.encoder,
                            decoder=model.decoder)
        elif isinstance(model, TFNeuralODE):
            new_model = TFNeuralODE(neuralnet=expert, 
                            taylor_order=0,
                            taylor_ad_mode=model.taylor_ad_mode, 
                            ivp_args=model.ivp_args)
        elif isinstance(model, DirectMapping):
            new_model = DirectMapping(neuralnet=expert, taylor_order=0)
        else:
            raise ValueError(f"The model type {type(model)} is not supported")
        return new_model

    def reset_contexts(self, nb_envs):
        if isinstance(self.contexts, InfDimContextParams):
            if hasattr(self.model, "vectorfield"):
                mlp_utils = self.model.vectorfield.neuralnet.ctx_utils[3]
            else:
                mlp_utils = self.model.neuralnet.ctx_utils[3]
            # contexts = InfDimContextParams(nb_envs=nb_envs, 
            #                         context_size=self.context_size,
            #                         hidden_size=mlp_utils[1],
            #                         depth=mlp_utils[2], 
            #                         key=None)
            input_dim, output_dim, hidden_size, depth, activation = mlp_utils
            key = self.contexts.key
            if key is not None:
                key, _ = jax.random.split(key)
            contexts = InfDimContextParams(nb_envs=nb_envs, 
                                    input_dim=input_dim,
                                    output_dim=output_dim,
                                    hidden_size=hidden_size,
                                    depth=depth, 
                                    activation=activation,
                                    key=key)
        elif isinstance(self.contexts, ArrayContextParams):
            key = self.contexts.key
            if key is not None:
                key, _ = jax.random.split(key)
            contexts = ArrayContextParams(nb_envs=nb_envs, 
                                        context_size=self.context_size,
                                        key=key)
        elif isinstance(self.contexts, GaussianContextParams):
            contexts = GaussianContextParams(nb_envs=nb_envs, 
                                        nb_gaussians_per_env=self.context_size//GAUSSIAN_ATTRIBUTE_COUNT_2D,
                                        img_shape=self.contexts.img_shape,
                                        key=self.contexts.key)
        elif isinstance(self.contexts, ConvContextParams):
            input_chans, output_chans, hidden_chans, kernel_size, depth, activation = self.model.neuralnet.ctx_utils[3]
            key = self.contexts.key
            if key is not None:
                key, _ = jax.random.split(key)
            contexts = ConvContextParams(nb_envs=nb_envs,
                                        input_chans=input_chans,
                                        output_chans=output_chans,
                                        hidden_chans=hidden_chans,
                                        kernel_size=kernel_size,
                                        depth=depth,
                                        activation=activation,
                                        key=key)
        else:
            raise ValueError(f"The context type {type(self.contexts)} is not supported")

        return contexts


    # @eqx.filter_jit
    def batch_predict(self, model, contexts, batch, max_envs=-1):
        """ Predict Y_hat for a batch issued from a dataloader
            CSM may or may not be deleted from the model; 
            as this function ensures the deactivation of CSM"""
        ## Predict in in a single batched call if possible, or a maximum sub-batches to avoid OOM

        X, Y = batch
        batched_model = eqx.filter_vmap(model, in_axes=(0, 0, 0))

        if max_envs==-1 or max_envs>=Y.shape[0] or self.loss_contributors==-1:
            Y_hat = batched_model(X, contexts.params, contexts.params)

        elif max_envs == None:
            sub_batch_size = self.loss_contributors
            print(f"    Too many environments to predict in a single batch, predicting in {sub_batch_size} environments ...")
            X_list = []
            Y_list = []
            Y_hat = []
            for i in range(0, Y.shape[0], sub_batch_size):
                contexts_ = contexts.params[i:i+sub_batch_size]
                Y_hat.append(batched_model(X[i:i+sub_batch_size], contexts_, contexts_))

                X_list.append(X[i:i+sub_batch_size])
                Y_list.append(Y[i:i+sub_batch_size])

                # break   ## TODO 1 sub-batch is enough ?

            Y_hat = jnp.concatenate(Y_hat, axis=0)
            X = jnp.concatenate(X_list, axis=0)
            Y = jnp.concatenate(Y_list, axis=0)

        else:
            contexts_ = contexts.params[:max_envs]
            # Y_hat = batched_model(X[:max_envs], contexts_, contexts_)
            # X = X[:max_envs]

            if isinstance(X, tuple) or isinstance(X, list):
                # X = (X[0], jnp.broadcast_to(X[1], (X[1].shape[0], X[0].shape[1], X[1].shape[1])))
                # X = (X[0], jnp.repeat(X[1], X[0].shape[1], axis=1))
                X = jax.tree_map(lambda x: x[:max_envs], X)
            else:
                X = X[:max_envs]

            Y_hat = batched_model(X, contexts_, contexts_)
            Y = Y[:max_envs]

        return X, Y, Y_hat


    # @eqx.filter_jit
    def batch_predict_multi(self, model, contexts, batch, max_envs=-1, uq_train_contexts=-1):
        """ Predict multiple Y_hats for a batch issued from a dataloader
            CSM should be active in the model;
            max_envs=6 means do not predict more than 6 environments, even if we have more in the batch
            uq_train_contexts is the number of training contexts to use for uncertainty quantification later on
            Upon return, the first result in Y_hat is the prediction for the context itself
            """

        X, Y = batch
        batched_model = eqx.filter_vmap(model, in_axes=(0, 0, 0))

        if uq_train_contexts != -1:
            train_contexts = self.contexts
            assert uq_train_contexts <= train_contexts.params.shape[0], "The number of UQ contexts must be less than the number of training contexts."
            assert uq_train_contexts > 1, "The number of UQ contexts must be greater than 1."
            ## Select the max_envs closest to each of the given contexts for prediction
            neighbors = []
            for e in range(contexts.params.shape[0]):
                dists = jnp.mean(jnp.abs(train_contexts.params-contexts.params[e]), axis=1)
                indices = jnp.argsort(dists)[:uq_train_contexts-1]      ## -1 because we will append the context itself
                # indices = jnp.argsort(dists)[-uq_train_contexts+1:]   ## TODO UQ is much too pronounced if we take the farthest 
                neigh_e = jnp.concat((contexts.params[e:e+1], train_contexts.params[indices]))
                neighbors.append(neigh_e)
        else:
            ## Reuse the given contexts as the neighbors (rearange so that 0 is the context itself)
            neighbors = []
            for e in range(contexts.params.shape[0]):
                neigh_e = jnp.concatenate((contexts.params[e:e+1], contexts.params[:e], contexts.params[e+1:]))
                neighbors.append(neigh_e)
        neighbors = jnp.stack(neighbors, axis=0)

        ### Now the prediction of a maximum of max_envs environments
        if max_envs==-1 or max_envs>=X.shape[0] or self.loss_contributors==-1:
            Y_hat = []
            for e in range(contexts.params.shape[0]):
                X_ctx = jnp.broadcast_to(X[e:e+1], (neighbors[e].shape[0], *X.shape[1:]))
                ctxs = jnp.broadcast_to(contexts.params[e:e+1], neighbors[e].shape)
                Y_hat.append(batched_model(X_ctx, ctxs, neighbors[e]))
        else:
            X = X[:max_envs]
            Y = Y[:max_envs]
            contexts_ = contexts.params[:max_envs]

            Y_hat = []
            for e in range(contexts_.shape[0]):
                X_ctx = jnp.broadcast_to(X[e:e+1], (neighbors[e].shape[0], *X.shape[1:]))
                ctxs = jnp.broadcast_to(contexts_[e:e+1], neighbors[e].shape)
                Y_hat.append(batched_model(X_ctx, ctxs, neighbors[e]))

        return X, Y, jnp.stack(Y_hat, axis=0)





class MLP(eqx.Module):
    """ An MLP """
    layers: jnp.ndarray

    def __init__(self, in_size, out_size, hidden_size, depth, activation, key=None):
        keys = jax.random.split(key, num=depth+1)

        self.layers = []

        for i in range(depth):
            if i==0:
                layer = eqx.nn.Linear(in_size, hidden_size, use_bias=True, key=keys[i])
            elif i==depth-1:
                layer = eqx.nn.Linear(hidden_size, out_size, use_bias=True, key=keys[i])
            else:
                layer = eqx.nn.Linear(hidden_size, hidden_size, use_bias=True, key=keys[i])

            self.layers.append(layer)

            if i != depth-1:
                self.layers.append(activation)

    def __call__(self, x):
        """ Returns y such that y = MLP(x) """
        y = x
        for layer in self.layers:
            y = layer(y)
        return y


class ConvNet(eqx.Module):
    """ An MLP """
    layers: jnp.ndarray

    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size, depth, activation, key=None):
        keys = jax.random.split(key, num=depth+1)

        self.layers = []

        for i in range(depth):
            if i==0:
                layer = eqx.nn.Conv2d(in_channels, hidden_channels, kernel_size, padding='SAME', key=keys[i])
            elif i==depth-1:
                layer = eqx.nn.Conv2d(hidden_channels, out_channels, kernel_size, padding='SAME', key=keys[i])
            else:
                layer = eqx.nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding='SAME', key=keys[i])

            self.layers.append(layer)

            if i != depth-1:
                self.layers.append(activation)

    def __call__(self, x):
        """ Returns y such that y = ConvNet(x) """
        y = x
        for layer in self.layers:
            y = layer(y)
        return y





class ArrayContextParams(eqx.Module):
    """ A context initialised with gaussian """
    params: jnp.ndarray
    eff_context_size: int
    key: jnp.ndarray

    def __init__(self, nb_envs, context_size, key=None):
        if key is None:
            self.params = jnp.zeros((nb_envs, context_size))
        else:
            self.params = jax.random.normal(key, (nb_envs, context_size))
        self.eff_context_size = context_size
        self.key = key

    def __call__(self):
        return self.params


class GaussianContextParams(eqx.Module):
    """ A context initialised with gaussian """
    params: jnp.ndarray
    eff_context_size: int
    key: jnp.ndarray        ## If we want the gaussian to be always initialised the same
    img_shape: tuple

    def __init__(self, nb_envs, nb_gaussians_per_env, img_shape=None, key=None):
        self.eff_context_size = nb_gaussians_per_env*GAUSSIAN_ATTRIBUTE_COUNT_2D
        self.key = key
        self.img_shape = img_shape

        if key is None:
            self.params = jnp.zeros((nb_envs, self.eff_context_size))
        else:
            if img_shape is None:
                raise ValueError("You must provide the intended rendered image shape to properly initialise the Gaussians.")
            gaussians = init_gaussians(key, img_shape, nb_envs*nb_gaussians_per_env,)
            self.params = jnp.reshape(gaussians, (nb_envs, self.eff_context_size))

    def __call__(self):
        # return jnp.reshape(self.params, (-1, 9))        ## Returns the gaussians
        return self.params                                ## Returns the flattened gaussians


class InfDimContextParams(eqx.Module):
    params: list
    ctx_utils: any
    eff_context_size: int     ## The effective/actual size of a context vector (flattened neural network)
    key: jnp.ndarray

    def __init__(self, nb_envs, input_dim, output_dim, hidden_size, depth, activation=jax.nn.softplus, key=None):
        if key is None:
            self.key = None
            keys = jax.random.split(jax.random.PRNGKey(0), nb_envs)
        else:
            self.key = key
            keys = jax.random.split(key, nb_envs)

        all_contexts = [MLP(input_dim, output_dim, hidden_size, depth, activation, key=keys[i]) for i in range(nb_envs)]

        mlp_utils = (input_dim, output_dim, hidden_size, depth, activation)

        ex_params, ex_static = eqx.partition(all_contexts[0], eqx.is_array)
        ex_ravel, ex_shapes, ex_treedef = flatten_pytree(ex_params)
        self.ctx_utils = (ex_shapes, ex_treedef, ex_static, mlp_utils)

        all_params_1D = [flatten_pytree(eqx.filter(context, eqx.is_array))[0] for context in all_contexts]

        self.eff_context_size = sum(x.size for x in jax.tree_util.tree_leaves(ex_params) if x is not None)

        if key is None:
            self.params = jnp.zeros_like(jnp.stack(all_params_1D, axis=0))
        else:
            self.params = jnp.stack(all_params_1D, axis=0)



class ConvContextParams(eqx.Module):
    params: list
    ctx_utils: any
    eff_context_size: int     ## The effective/actual size of a context vector (flattened neural network)
    key: jnp.ndarray

    def __init__(self, nb_envs, input_chans, output_chans, hidden_chans, kernel_size, depth, activation=jax.nn.relu, key=None):

        if key is None:
            self.key = None
            keys = jax.random.split(jax.random.PRNGKey(0), nb_envs)
        else:
            self.key = key
            keys = jax.random.split(key, nb_envs)

        all_contexts = [ConvNet(input_chans, output_chans, hidden_chans, kernel_size, depth, activation, key=keys[i]) for i in range(nb_envs)]

        mlp_utils = (input_chans, output_chans, hidden_chans, kernel_size, depth, activation)

        ex_params, ex_static = eqx.partition(all_contexts[0], eqx.is_array)
        ex_ravel, ex_shapes, ex_treedef = flatten_pytree(ex_params)
        self.ctx_utils = (ex_shapes, ex_treedef, ex_static, mlp_utils)

        all_params_1D = [flatten_pytree(eqx.filter(context, eqx.is_array))[0] for context in all_contexts]

        self.eff_context_size = sum(x.size for x in jax.tree_util.tree_leaves(ex_params) if x is not None)

        if key is None:
            self.params = jnp.zeros_like(jnp.stack(all_params_1D, axis=0))
        else:
            self.params = jnp.stack(all_params_1D, axis=0)




class NeuralContextFlow(eqx.Module):
    neuralnet: eqx.Module
    taylor_order: int
    taylor_weight: jnp.ndarray
    taylor_scale: int

    def __init__(self, neuralnet, taylor_order, taylor_weight_init=0., taylor_scale=100):
        ############# NCF without the possibility to ignore Taylor expansion #############
        self.neuralnet = neuralnet
        self.taylor_order = taylor_order

        ## Taylor weight and scale are only included for backward compatibility
        self.taylor_weight = jnp.array([taylor_weight_init]).squeeze()
        self.taylor_scale = taylor_scale


    def __call__(self, xs, ctx, ctx_):

        def point_predict(x):

            vf = lambda xi: self.neuralnet(x, xi)

            if self.taylor_order==0:
                return vf(ctx)

            elif self.taylor_order==1:
                gradvf = lambda xi_: eqx.filter_jvp(vf, (xi_,), (ctx-xi_,))[1]
                return vf(ctx_) + 1.0*gradvf(ctx_)

            elif self.taylor_order==2:
                gradvf = lambda xi_: eqx.filter_jvp(vf, (xi_,), (ctx-xi_,))[1]
                scd_order_term = eqx.filter_jvp(gradvf, (ctx_,), (ctx-ctx_,))[1]
                return vf(ctx_) + 1.5*gradvf(ctx_) + 0.5*scd_order_term

            else:
                h0 = ctx_
                h1 = ctx-ctx_
                h2 = jnp.zeros_like(h0)

                hs = [h1, h2]
                coeffs = [1, 0.5]
                for order in range(2+1, self.taylor_order+1):
                    hs.append(jnp.zeros_like(h0))
                    coeffs.append(1 / factorial(order))

                f0, fs = jet(vf, (h0,), (hs,))
                taylor_exp = f0 + jnp.sum(jnp.stack(fs, axis=-1) * jnp.array(coeffs)[None,:], axis=-1)

                return taylor_exp

        ys = eqx.filter_vmap(point_predict)(xs)

        return ys



class NeuralNeuralContextFlow(eqx.Module):
    neuralnet: eqx.Module
    flownet: eqx.Module

    def __init__(self, neuralnet, flownet=None):
        ############# NCF with a flow network instead of Taylor expansion #############
        self.neuralnet = neuralnet
        self.flownet = flownet

    def __call__(self, xs, ctx, ctx_):

        def point_predict(x):

            vf = lambda xi: self.neuralnet(x, xi)

            if self.flownet is None:
                return vf(ctx)

            else:
                out_main = vf(ctx_)
                correction = self.flownet(out_main, ctx_, vf(ctx), ctx)
                # return vf(ctx_) + correction      ## TODO use different variations of the input/outputs to the flow network
                return out_main + correction

        ys = eqx.filter_vmap(point_predict)(xs)

        return ys






class NeuralContextFlowAdaptiveTaylor(eqx.Module):
    neuralnet: eqx.Module
    taylor_order: int
    taylor_weight: jnp.ndarray
    taylor_scale: int

    def __init__(self, neuralnet, taylor_order, taylor_weight_init=0., taylor_scale=100):
        """ Neural Context Flow with an additional parameter to select the weight of the Taylor expansion """

        self.neuralnet = neuralnet

        self.taylor_order = taylor_order
        self.taylor_weight = jnp.array([taylor_weight_init])
        self.taylor_scale = taylor_scale


    def __call__(self, xs, ctx, ctx_):

        def point_predict(x):

            ############# With possibility to ignore Taylor expansion #############
            vf = lambda xi: self.neuralnet(x, xi)
            alpha = jax.nn.sigmoid(self.taylor_scale*self.taylor_weight[0])

            if self.taylor_order==0:
                return (alpha)*vf(ctx)       ## Could be (1.-alpha)*vf(ctx), but problem when resetting the model with different alpha

            elif self.taylor_order==1:
                gradvf = lambda xi_: eqx.filter_jvp(vf, (xi_,), (ctx-xi_,))[1]
                taylor_exp = vf(ctx_) + 1.0*gradvf(ctx_)

                return (1.-alpha)*vf(ctx) + (alpha)*taylor_exp

            elif self.taylor_order==2:
                gradvf = lambda xi_: eqx.filter_jvp(vf, (xi_,), (ctx-xi_,))[1]
                scd_order_term = eqx.filter_jvp(gradvf, (ctx_,), (ctx-ctx_,))[1]
                taylor_exp = vf(ctx_) + 1.5*gradvf(ctx_) + 0.5*scd_order_term

                return (1.-alpha)*vf(ctx) + (alpha)*taylor_exp

            else:
                # raise NotImplementedError("Higher order terms are not implemented yet.")

                h0 = ctx_
                h1 = ctx-ctx_
                h2 = jnp.zeros_like(h0)

                hs = [h1, h2]
                coeffs = [1, 0.5]
                for order in range(2+1, self.taylor_order+1):
                    hs.append(jnp.zeros_like(h0))
                    coeffs.append(1 / factorial(order))

                f0, fs = jet(vf, (h0,), (hs,))
                taylor_exp = f0 + jnp.sum(jnp.stack(fs, axis=-1) * jnp.array(coeffs)[None,:], axis=-1)

                return (1.-alpha)*vf(ctx) + (alpha)*taylor_exp


        ys = eqx.filter_vmap(point_predict)(xs)

        return ys






class BatchedNeuralContextFlow(eqx.Module):
    neuralnet: eqx.Module
    taylor_order: int

    def __init__(self, neuralnet, taylor_order):
        self.neuralnet = neuralnet
        self.taylor_order = taylor_order

    def __call__(self, xs, ctx, ctx_):

        vf = lambda xi: self.neuralnet(xs, xi)

        if self.taylor_order==0:
            return vf(ctx)

        elif self.taylor_order==1:
            gradvf = lambda xi_: eqx.filter_jvp(vf, (xi_,), (ctx-xi_,))[1]
            taylor_exp = vf(ctx_) + 1.0*gradvf(ctx_)

            return taylor_exp

        elif self.taylor_order==2:
            gradvf = lambda xi_: eqx.filter_jvp(vf, (xi_,), (ctx-xi_,))[1]
            scd_order_term = eqx.filter_jvp(gradvf, (ctx_,), (ctx-ctx_,))[1]
            taylor_exp = vf(ctx_) + 1.5*gradvf(ctx_) + 0.5*scd_order_term

            return taylor_exp

        else:
            # raise NotImplementedError("Higher order terms are not implemented yet.")
            h0 = ctx_
            h1 = ctx-ctx_
            h2 = jnp.zeros_like(h0)

            hs = [h1, h2]
            coeffs = [1, 0.5]
            for order in range(2+1, self.taylor_order+1):
                hs.append(jnp.zeros_like(h0))
                coeffs.append(1 / factorial(order))

            f0, fs = jet(vf, (h0,), (hs,))
            taylor_exp = f0 + jnp.sum(jnp.stack(fs, axis=-1) * jnp.array(coeffs)[None,:], axis=-1)

            return taylor_exp




class BatchedNeuralContextFlowAdaptiveTaylor(eqx.Module):
    neuralnet: eqx.Module

    taylor_order: int
    taylor_scale: int
    taylor_weight: jnp.ndarray

    def __init__(self, neuralnet, taylor_order, taylor_weight_init=0., taylor_scale=100):
        self.neuralnet = neuralnet

        self.taylor_order = taylor_order
        self.taylor_weight = jnp.array([taylor_weight_init])        ## We start with 50-50
        self.taylor_scale = taylor_scale                     ## Multiply by this before sigmoid

    def __call__(self, xs, ctx, ctx_):

        vf = lambda xi: self.neuralnet(xs, xi)
        alpha = jax.nn.sigmoid(self.taylor_scale*self.taylor_weight[0])

        if self.taylor_order==0:
            return (1.-alpha)*vf(ctx)

        elif self.taylor_order==1:
            gradvf = lambda xi_: eqx.filter_jvp(vf, (xi_,), (ctx-xi_,))[1]
            taylor_exp = vf(ctx_) + 1.0*gradvf(ctx_)

            return (1.-alpha)*vf(ctx) + (alpha)*taylor_exp

        elif self.taylor_order==2:
            gradvf = lambda xi_: eqx.filter_jvp(vf, (xi_,), (ctx-xi_,))[1]
            scd_order_term = eqx.filter_jvp(gradvf, (ctx_,), (ctx-ctx_,))[1]
            taylor_exp = vf(ctx_) + 1.5*gradvf(ctx_) + 0.5*scd_order_term

            return (1.-alpha)*vf(ctx) + (alpha)*taylor_exp

        else:
            # raise NotImplementedError("Higher order terms are not implemented yet.")
            h0 = ctx_
            h1 = ctx-ctx_
            h2 = jnp.zeros_like(h0)

            hs = [h1, h2]
            coeffs = [1, 0.5]
            for order in range(2+1, self.taylor_order+1):
                hs.append(jnp.zeros_like(h0))
                coeffs.append(1 / factorial(order))

            f0, fs = jet(vf, (h0,), (hs,))
            taylor_exp = f0 + jnp.sum(jnp.stack(fs, axis=-1) * jnp.array(coeffs)[None,:], axis=-1)

            return (1.-alpha)*vf(ctx) + (alpha)*taylor_exp




class SelfModVectorField(eqx.Module):
    """ A vector field with fixed Taylor order """
    neuralnet: eqx.Module
    taylor_order: int
    taylor_ad_mode: str

    def __init__(self, neuralnet, taylor_order, taylor_ad_mode):
        self.neuralnet = neuralnet
        self.taylor_order = taylor_order
        self.taylor_ad_mode = taylor_ad_mode

    def __call__(self, t, x, args):
        ctx, ctx_ = args

        vf = lambda xi: self.neuralnet(t, x, xi)

        if self.taylor_order==0:
            return vf(ctx)

        elif self.taylor_order==1:
            if self.taylor_ad_mode=="forward":
                gradvf = lambda xi_: eqx.filter_jvp(vf, (xi_,), (ctx-xi_,))[1]
                taylor_exp = vf(ctx_) + 1.0*gradvf(ctx_)
            elif self.taylor_ad_mode=="reverse":
                jac = eqx.filter_jacrev(vf)(ctx_)
                taylor_exp = vf(ctx_) + jac @ (ctx-ctx_)
            else:
                raise ValueError("Invalid AD mode provided.")

            return taylor_exp

        elif self.taylor_order==2:
            if self.taylor_ad_mode=="forward":
                gradvf = lambda xi_: eqx.filter_jvp(vf, (xi_,), (ctx-xi_,))[1]
                scd_order_term = eqx.filter_jvp(gradvf, (ctx_,), (ctx-ctx_,))[1]
                taylor_exp = vf(ctx_) + 1.5*gradvf(ctx_) + 0.5*scd_order_term
            elif self.taylor_ad_mode=="reverse":
                print("WARNING: Reverse-mode AD for 2nd order Taylor expansion materialises the Hessian and is unstable for the CAVIA algorithm. Consider reducing the Taylor order or using forward-mode AD.")
                jac = eqx.filter_jacrev(vf)(ctx_)
                hess = eqx.filter_jacrev(eqx.filter_jacrev(vf))(ctx_)
                taylor_exp = vf(ctx_) + jac @ (ctx-ctx_) + 0.5 * (hess @ (ctx-ctx_)) @ (ctx-ctx_)
            else:
                raise ValueError("Invalid AD mode provided.")

            return taylor_exp

        else:
            if self.taylor_ad_mode=="forward":
                h0 = ctx_
                h1 = ctx-ctx_
                h2 = jnp.zeros_like(h0)

                hs = [h1, h2]
                coeffs = [1, 0.5]
                for order in range(2+1, self.taylor_order+1):
                    hs.append(jnp.zeros_like(h0))
                    coeffs.append(1 / factorial(order))

                f0, fs = jet(vf, (h0,), (hs,))
                taylor_exp = f0 + jnp.sum(jnp.stack(fs, axis=-1) * jnp.array(coeffs)[None,:], axis=-1)
            else:
                raise ValueError("Higher order terms are only implemented for forward mode AD.")

            return taylor_exp


class NeuralODE(eqx.Module):
    vectorfield: eqx.Module
    ivp_args: dict
    taylor_order: int
    taylor_ad_mode: str
    t_eval: tuple

    def __init__(self, neuralnet, taylor_order, ivp_args=None, t_eval=None, taylor_ad_mode="forward"):
        self.ivp_args = ivp_args if ivp_args is not None else {}
        self.vectorfield = SelfModVectorField(neuralnet, taylor_order=taylor_order, taylor_ad_mode=taylor_ad_mode)
        self.taylor_order = taylor_order
        self.taylor_ad_mode = taylor_ad_mode
        self.t_eval = t_eval

    def get_t_eval(self, y0):
        """ Determines the appropriate t-eval based on the input y0 """
        if self.t_eval is None:     
            if not self.ivp_args.get("return_traj", False):     ## User only cares for terminal state
                t_eval = jnp.array((0., self.ivp_args.get("T", 1.)))
            else:   ## Users cares for trajectory, but didn't provide it at initialisationt ime. User must now provide t_eval in every call!
                y0, t_eval = y0
        else:       ## User provides t_eval in the constructor
            t_eval = jnp.array(self.t_eval)

        return y0, t_eval


    def __call__(self, xs, ctx, ctx_):

        integrator = self.ivp_args.get("integrator", diffrax.Dopri5())

        # if isinstance(integrator, type(eqx.Module)):
        if not callable(integrator):
            def integrate(y0):
                y0, t_eval = self.get_t_eval(y0)    ## y0 might be a tuple (y0, t_eval)

                sol = diffrax.diffeqsolve(
                        terms=diffrax.ODETerm(self.vectorfield),
                        solver=integrator,
                        args=(ctx, ctx_.squeeze()),
                        t0=t_eval[0],
                        t1=t_eval[-1],
                        dt0=self.ivp_args.get("dt_init", t_eval[1]-t_eval[0]),
                        y0=jnp.concat([y0, jnp.zeros((self.ivp_args.get("y0_pad_size", 0),))], axis=0),
                        stepsize_controller=diffrax.PIDController(rtol=self.ivp_args.get("rtol", 1e-3), 
                                                                    atol=self.ivp_args.get("atol", 1e-6),
                                                                    dtmin=self.ivp_args.get("dt_min", None)),
                        saveat=diffrax.SaveAt(ts=t_eval),
                        adjoint=self.ivp_args.get("adjoint", diffrax.RecursiveCheckpointAdjoint()),
                        max_steps=self.ivp_args.get("max_steps", 4096*1),
                        throw=True,    ## Keep the nans and infs, don't throw and error !
                    )
                # jax.debug.print("SOL {}", sol.ys)
                ys = sol.ys
                clip = self.ivp_args.get("clip_sol", None)
                if clip is not None:
                    ys = jnp.clip(ys, clip[0], clip[1])

                if self.ivp_args.get("return_traj", False):
                    return ys[:, :y0.shape[0]]
                else:
                    return ys[-1, :y0.shape[0]]

        else:   ## Custom-made integrator
            def integrate(y0):
                y0, t_eval = self.get_t_eval(y0)
                ys = integrator(fun=self.vectorfield, 
                                t_span=(t_eval[0], t_eval[-1]), 
                                y0=y0,
                                args=(ctx, ctx_.squeeze()),
                                t_eval=t_eval, 
                                **self.ivp_args
                                )
                if self.ivp_args.get("return_traj", False):
                    return ys
                else:
                    return ys[-1]

        if isinstance(xs, tuple) or isinstance(xs, list):
            xs = (xs[0], jnp.broadcast_to(xs[1][None,:], (xs[0].shape[0], xs[1].shape[0])))

        batched_results = eqx.filter_vmap(integrate)(xs)

        # return jnp.nan_to_num(batched_results, nan=0., posinf=0., neginf=0.)
        return batched_results



class TFSelfModVectorField(eqx.Module):
    """ A vector field for the teacher-forced Neural ODE """
    neuralnet: eqx.Module
    taylor_order: int
    taylor_ad_mode: str

    def __init__(self, neuralnet, taylor_order, taylor_ad_mode):
        self.neuralnet = neuralnet
        self.taylor_order = taylor_order
        self.taylor_ad_mode = taylor_ad_mode

    def interp_signal(self, tau, xs_, t_evals):
        return jnp.interp(tau, t_evals, xs_, left="extrapolate", right="extrapolate").squeeze()

    def __call__(self, t, hat_x, args):
        ctx, ctx_, true_xs, t_evals = args

        # dt = t_evals[1] - t_evals[0]
        x = eqx.filter_vmap(self.interp_signal, in_axes=(None, 0, None))(t, true_xs.T, t_evals).T       ## Big difference with the previous version

        vf = lambda xi: self.neuralnet(t, x, xi)

        if self.taylor_order==0:
            return vf(ctx)

        elif self.taylor_order==1:
            if self.taylor_ad_mode=="forward":
                gradvf = lambda xi_: eqx.filter_jvp(vf, (xi_,), (ctx-xi_,))[1]
                taylor_exp = vf(ctx_) + 1.0*gradvf(ctx_)
            elif self.taylor_ad_mode=="reverse":
                jac = eqx.filter_jacrev(vf)(ctx_)
                taylor_exp = vf(ctx_) + jac @ (ctx-ctx_)
            else:
                raise ValueError("Invalid AD mode provided.")

            return taylor_exp

        elif self.taylor_order==2:
            if self.taylor_ad_mode=="forward":
                gradvf = lambda xi_: eqx.filter_jvp(vf, (xi_,), (ctx-xi_,))[1]
                scd_order_term = eqx.filter_jvp(gradvf, (ctx_,), (ctx-ctx_,))[1]
                taylor_exp = vf(ctx_) + 1.5*gradvf(ctx_) + 0.5*scd_order_term
            elif self.taylor_ad_mode=="reverse":
                print("WARNING: Reverse-mode AD for 2nd order Taylor expansion materialises the Hessian and is unstable for the CAVIA algorithm. Consider reducing the Taylor order or using forward-mode AD.")
                jac = eqx.filter_jacrev(vf)(ctx_)
                hess = eqx.filter_jacrev(eqx.filter_jacrev(vf))(ctx_)
                taylor_exp = vf(ctx_) + jac @ (ctx-ctx_) + 0.5 * (hess @ (ctx-ctx_)) @ (ctx-ctx_)
            else:
                raise ValueError("Invalid AD mode provided.")

            return taylor_exp

        else:
            if self.taylor_ad_mode=="forward":
                h0 = ctx_
                h1 = ctx-ctx_
                h2 = jnp.zeros_like(h0)

                hs = [h1, h2]
                coeffs = [1, 0.5]
                for order in range(2+1, self.taylor_order+1):
                    hs.append(jnp.zeros_like(h0))
                    coeffs.append(1 / factorial(order))

                f0, fs = jet(vf, (h0,), (hs,))
                taylor_exp = f0 + jnp.sum(jnp.stack(fs, axis=-1) * jnp.array(coeffs)[None,:], axis=-1)
            else:
                raise ValueError("Higher order terms are only implemented for forward mode AD.")

            return taylor_exp



class TFNeuralODE(eqx.Module):
    """ A teacher-forced Neural ODE: the vector field ignores the current state and uses the (interpolated) target state 
    TODO: Add the possibility to use the current state as well in a linear combination, as in Generalized Teacher-Forcing
    """
    vectorfield: eqx.Module
    ivp_args: dict
    taylor_order: int
    taylor_ad_mode: str

    def __init__(self, neuralnet, taylor_order, ivp_args=None, taylor_ad_mode="forward"):
        self.ivp_args = ivp_args if ivp_args is not None else {}
        self.vectorfield = TFSelfModVectorField(neuralnet, taylor_order=taylor_order, taylor_ad_mode=taylor_ad_mode)
        self.taylor_order = taylor_order
        self.taylor_ad_mode = taylor_ad_mode

    def __call__(self, xts, ctx, ctx_):

        integrator = self.ivp_args.get("integrator", diffrax.Dopri5())

        if not callable(integrator):
            def integrate(ys, ts):
                y0, t_eval = ys[0], ts

                sol = diffrax.diffeqsolve(
                        terms=diffrax.ODETerm(self.vectorfield),
                        solver=integrator,
                        args=(ctx, ctx_.squeeze(), ys, t_eval),
                        t0=t_eval[0],
                        t1=t_eval[-1],
                        dt0=self.ivp_args.get("dt_init", t_eval[1]-t_eval[0]),
                        y0=jnp.concat([y0, jnp.zeros((self.ivp_args.get("y0_pad_size", 0),))], axis=0),
                        stepsize_controller=diffrax.PIDController(rtol=self.ivp_args.get("rtol", 1e-3), 
                                                                    atol=self.ivp_args.get("atol", 1e-6),
                                                                    dtmin=self.ivp_args.get("dt_min", None)),
                        saveat=diffrax.SaveAt(ts=t_eval),
                        adjoint=self.ivp_args.get("adjoint", diffrax.RecursiveCheckpointAdjoint()),
                        max_steps=self.ivp_args.get("max_steps", 4096*1),
                        throw=True,    ## Keep the nans and infs, don't throw and error ?
                    )
                ys = sol.ys
                clip = self.ivp_args.get("clip_sol", None)
                if clip is not None:
                    ys = jnp.clip(ys, clip[0], clip[1])

                if self.ivp_args.get("return_traj", False):
                    return ys[:, :y0.shape[0]]
                else:
                    return ys[-1, :y0.shape[0]]

        else:   ## Custom-made integrator
            def integrate(ys, ts):
                y0, t_eval = ys[0], ts
                ys = integrator(fun=self.vectorfield, 
                                t_span=(t_eval[0], t_eval[-1]), 
                                y0=y0,
                                args=(ctx, ctx_.squeeze(), ys, t_eval),
                                t_eval=t_eval, 
                                **self.ivp_args
                                )
                if self.ivp_args.get("return_traj", False):
                    return ys
                else:
                    return ys[-1]

        xs = xts[0]
        ts = jnp.broadcast_to(xts[1][None,:], (xts[0].shape[0], xts[1].shape[0]))     ## Broadcast along trajectories in the same environment

        batched_results = eqx.filter_vmap(integrate)(xs, ts)

        # return jnp.nan_to_num(batched_results, nan=0., posinf=0., neginf=0.)
        return batched_results


class NeuralCDE(eqx.Module):
    vectorfield: eqx.Module
    ivp_args: dict
    taylor_order: int
    taylor_ad_mode: str

    encoder: eqx.Module
    decoder: eqx.Module

    def __init__(self, neuralnet, taylor_order, ivp_args=None, taylor_ad_mode="forward", encoder=None, decoder=None):
        self.ivp_args = ivp_args if ivp_args is not None else {}
        self.vectorfield = SelfModVectorField(neuralnet, taylor_order=taylor_order, taylor_ad_mode=taylor_ad_mode)
        self.taylor_order = taylor_order
        self.taylor_ad_mode = taylor_ad_mode

        self.encoder = encoder
        self.decoder = decoder

    def __call__(self, xts, ctx, ctx_):
        """ Forward call of the Neural CDE """

        def vectorfield(t, x, args):
            y = self.vectorfield(t, x, args)
            return jnp.reshape(jax.nn.tanh(y), (x.shape[0], -1))

        def integrate(z0, xs, t_eval):
            coeffs = diffrax.backward_hermite_coefficients(t_eval, xs)
            control = diffrax.CubicInterpolation(t_eval, coeffs)

            sol = diffrax.diffeqsolve(
                    # diffrax.ControlTerm(self.vectorfield, control).to_ode(),
                    diffrax.ControlTerm(vectorfield, control).to_ode(),
                    diffrax.Tsit5(),
                    args=(ctx, ctx_.squeeze()),
                    t0=t_eval[0],
                    t1=t_eval[-1],
                    dt0=t_eval[1]-t_eval[0],
                    y0=z0,
                    stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
                    saveat=diffrax.SaveAt(ts=t_eval),
                    adjoint=diffrax.RecursiveCheckpointAdjoint(),
                    max_steps=4096*5
                )

            return sol.ys

        if not isinstance(xts, tuple) and not isinstance(xts, list):
            return ValueError("The Neural CDE expects both trajctories and times as inputs.")

        xs = xts[0]
        ts = jnp.broadcast_to(xts[1][None,:], (xts[0].shape[0], xts[1].shape[0]))     ## Broadcast along trajectories in the same environment

        z0s = eqx.filter_vmap(self.encoder)(xs[:, 0, ...])        ## Shape: (batch, latent_size)

        ## Add time as a channel to all_xs
        # all_ts = jnp.ones((xs.shape[0], xs.shape[1], 1)) * ts[None, :, None]
        xts = jnp.concatenate((ts[...,None], xs), axis=-1)

        zs = eqx.filter_vmap(integrate)(z0s, xts, ts)        ## Shape: (batch, T, latent_size)
        x_recons = eqx.filter_vmap(eqx.filter_vmap(self.decoder))(zs)        ## Shape: (batch, T, data_size)

        return x_recons


class DirectMapping(eqx.Module):
    """ Fall Back Model in case neither NeuralODE, TFNeuralODE, NeuralCDE is used.
    Presented like a NeuralODE (with a vf) but it's just a FeedForward NN
    """
    vectorfield: list
    taylor_order: int
    def __init__(self, neuralnet, taylor_order):
        self.vectorfield = BatchedNeuralContextFlow(neuralnet, taylor_order=taylor_order)
        self.taylor_order = taylor_order
    def __call__(self, xts, ctx, ctx_):
        xs, ts = xts
        new_ts = jnp.broadcast_to(ts[None,:], (xs.shape[0], *ts.shape))     ## Expand/Repead ts to match the xs
        return eqx.filter_vmap(self.vectorfield, in_axes=(0, None, None))((xs, new_ts), ctx, ctx_.squeeze())


class Swish(eqx.Module):
    """ Swish activation function """
    beta: jnp.ndarray
    def __init__(self, key=None):
        self.beta = jax.random.uniform(key, shape=(1,), minval=0.01, maxval=1.0)
    def __call__(self, x):
        return x * jax.nn.sigmoid(self.beta * x)



class NeuroModulatedSwish(eqx.Module):
    """ NMN neuro-modulation layer with swish base activation function: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0227922 """
    beta: jnp.ndarray
    w_s: jnp.ndarray
    w_b: jnp.ndarray

    def __init__(self, latent_size, key=None):
        self.beta = jax.random.uniform(key, shape=(1,), minval=0.1, maxval=1.0)
        self.w_s = jnp.ones((latent_size, 1))
        self.w_b = jnp.zeros((latent_size, 1))

    def __call__(self, x, ctx):
        y = ctx.T @ (x * self.w_s + self.w_b)
        return y * jax.nn.sigmoid(self.beta * y)







class Upsample2D(eqx.Module):
    """ Upsample 2D image by a factor: https://docs.kidger.site/equinox/examples/unet/ """
    factor: int
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, y):
        C, H, W = y.shape
        y = jnp.reshape(y, [C, H, 1, W, 1])
        y = jnp.tile(y, [1, 1, self.factor, 1, self.factor])
        return jnp.reshape(y, [C, H * self.factor, W * self.factor])


class VAEDecoder(eqx.Module):
    """ Decoder with dense layers and deconvolutions"""
    img_size: list
    kernel_size: list
    latent_dim: int

    layers: list

    def __init__(self, img_size, kernel_size, latent_dim, key):
        self.img_size = img_size
        self.kernel_size = kernel_size
        self.latent_dim = latent_dim

        layer_keys = jax.random.split(key, 4)
        H, W, C = self.img_size

        self.layers = [
            eqx.nn.Linear(latent_dim, 48, key=layer_keys[0]),
            eqx.nn.PReLU(init_alpha=0.),
            eqx.nn.Linear(48, 12*H*W//(4*4), key=layer_keys[1]),
            eqx.nn.PReLU(init_alpha=0.),
            lambda x: x.reshape((12, H//4, W//4)),
            Upsample2D(factor=2),
            eqx.nn.ConvTranspose2d(12, 8, kernel_size, padding="SAME", key=layer_keys[2]),
            eqx.nn.PReLU(init_alpha=0.),
            Upsample2D(factor=2),
            eqx.nn.ConvTranspose2d(8, C, kernel_size, padding="SAME", key=layer_keys[3]),
            jax.nn.sigmoid
        ]

    def __call__(self, z):
        x = z
        for layer in self.layers:
            x = layer(x)
        return x

class Decoder(eqx.Module):
    """ Decoder with dense layers and deconvolutions"""
    img_size: list
    kernel_size: list
    latent_dim: int

    layers: list

    def __init__(self, img_size, kernel_size, latent_dim, key):
        self.img_size = img_size
        self.kernel_size = kernel_size
        self.latent_dim = latent_dim

        layer_keys = jax.random.split(key, 4)
        H, W, C = self.img_size

        self.layers = [
            eqx.nn.Linear(latent_dim, 1024, key=layer_keys[0]),
            eqx.nn.PReLU(init_alpha=0.),
            eqx.nn.Linear(1024, 64*H*W//(4*4), key=layer_keys[1]),
            eqx.nn.PReLU(init_alpha=0.),
            lambda x: x.reshape((64, H//4, W//4)),
            Upsample2D(factor=2),
            eqx.nn.ConvTranspose2d(64, 16, kernel_size, padding="SAME", key=layer_keys[2]),
            eqx.nn.PReLU(init_alpha=0.),
            Upsample2D(factor=2),
            eqx.nn.ConvTranspose2d(16, C, kernel_size, padding="SAME", key=layer_keys[3]),
            jax.nn.sigmoid
        ]

    def __call__(self, z):
        x = z
        for layer in self.layers:
            x = layer(x)
        return x



class FuncContextParams(eqx.Module):
    params: list
    img_size: list
    kernel_size: list
    latent_dim: int
    context_size: int

    ctx_utils: any

    def __init__(self, nb_envs, key=None):

        keys = jax.random.split(key, num=nb_envs)

        # all_contexts = [Decoder(img_size=[32, 32, 3], kernel_size=[3, 3], latent_dim=18, key=keys[i]) for i in range(nb_envs)]

        ## Load the decoders from 240101-193230-VAE/decoder.eqx
        all_contexts = [eqx.tree_deserialise_leaves("runs/240101-193230-VAE/decoder.eqx", Decoder(img_size=[32, 32, 3], kernel_size=[3, 3], latent_dim=18, key=keys[i])) for i in range(nb_envs)]

        self.img_size = all_contexts[0].img_size
        self.kernel_size = all_contexts[0].kernel_size
        self.latent_dim = all_contexts[0].latent_dim

        ex_params, ex_static = eqx.partition(all_contexts[0], eqx.is_array)
        ex_ravel, ex_shapes, ex_treedef = flatten_pytree(ex_params)
        # self.ctx_utils = (ex_shapes, ex_treedef, ex_static)
        non_empty_shapes = []
        for shape in ex_shapes:
            if shape == ():
                non_empty_shapes.append((1,))
            else:
                non_empty_shapes.append(shape)
        self.ctx_utils = (non_empty_shapes, ex_treedef, ex_static)

        all_params_1D = [flatten_pytree(eqx.filter(context, eqx.is_array))[0] for context in all_contexts]
        self.context_size = all_params_1D[0].shape[0]

        self.params = jnp.stack(all_params_1D, axis=0)
        # self.params = jnp.zeros_like(jnp.stack(all_params_1D, axis=0))


    def __call__(self, z):
        def unravel_and_call(ctx, z):
            context = jax.flatten_util.unravel_pytree(ctx, self.treedef)
            return context(z)
        return jax.vmap(unravel_and_call)(self.params, z)






########### Implementation of a Vnet model ###########

class DownsamplingLayer(eqx.Module):
    layer: eqx.Module
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, *, key):
        self.layer = eqx.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding="SAME", key=key)
    
    def __call__(self, x):
        return self.layer(x)

class UpsamplingLayer(eqx.Module):
    layer: eqx.Module

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, *, key):
        self.layer = eqx.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding="SAME", key=key)

    def __call__(self, x):
        return self.layer(x)

class DoubleConv(eqx.Module):
    layer_1: eqx.Module
    layer_2: eqx.Module
    activation: callable
    norm_layer: eqx.Module
    dropout_rate: float

    def __init__(self, in_channels, out_channels, kernel_size=3, activation=jax.nn.relu, batch_norm=False, dropout_rate=0., *, key):
        k1, k2 = jax.random.split(key, 2)

        self.layer_1 = eqx.nn.Conv2d(in_channels, out_channels, kernel_size, padding='SAME', key=k1)
        self.layer_2 = eqx.nn.Conv2d(out_channels, out_channels, kernel_size, padding='SAME', key=k2)
        self.activation = activation
        if batch_norm:
            self.norm_layer = eqx.nn.BatchNorm(input_size=out_channels)
        else:
            self.norm_layer = None
        self.dropout_rate = dropout_rate

    def __call__(self, x):
        x = self.layer_1(x)
        x = self.activation(x)
        x = self.layer_2(x)
        x = self.activation(x)
        if self.norm_layer:
            x = self.norm_layer(x)
        if self.dropout_rate > 0.:
            x = eqx.nn.Dropout(self.dropout_rate)(x)
        return x



class VNet(eqx.Module):
    input_shape: tuple
    output_shape: tuple
    levels: int
    depth: int
    kernel_size: int
    activation: callable
    final_activation: callable
    batch_norm: bool
    dropout_rate: float

    ## Learnable params
    left_doubleconvs: dict
    right_doubleconvs: dict
    downsamplings: dict
    upsamplings: dict
    final_conv: eqx.Module


    def __init__(self, input_shape, output_shape, levels=5, depth=32, kernel_size=5, activation=jax.nn.relu, final_activation=jax.nn.sigmoid, batch_norm=True, dropout_rate=0., *, key):

        l_key, r_key, d_key, u_key, f_key = jax.random.split(key, 5)

        self.input_shape = input_shape      ## C, H, W
        self.output_shape = output_shape    ## C, H, W
        self.levels = levels
        self.depth = depth                  ## Number of filters in the first layer
        self.kernel_size = kernel_size
        self.activation = activation
        self.final_activation = final_activation
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate

        self.left_doubleconvs = {}
        self.right_doubleconvs = {}
        self.downsamplings = {}
        self.upsamplings = {}
        self.final_conv = eqx.nn.Conv2d(depth, output_shape[0], 1, padding="SAME", key=f_key)


        ## NOTE! The convolutions are not changing the number of channels, the downsampling and upsampling layers are

        d_keys = jax.random.split(d_key, levels-1)
        l_keys = jax.random.split(l_key, levels)

        self.left_doubleconvs[0] = DoubleConv(input_shape[0], depth, kernel_size, activation, batch_norm, dropout_rate, key=l_keys[0])
        for i in range(1, levels):
            self.downsamplings[i] = DownsamplingLayer(self.depth*2**(i-1), self.depth*2**(i), key=d_keys[i-1])
            self.left_doubleconvs[i] = DoubleConv(self.depth*2**(i), self.depth*2**(i), kernel_size, activation, batch_norm, dropout_rate, key=l_keys[i])

        u_keys = jax.random.split(u_key, levels-1)
        r_keys = jax.random.split(r_key, levels-1)

        for i in range(self.levels-2, -1, -1):
            self.upsamplings[i] = UpsamplingLayer(self.depth*2**(i+1), self.depth*2**i, key=u_keys[i])
            self.right_doubleconvs[i] = DoubleConv(self.depth*2**(i), self.depth*2**i, kernel_size, activation, batch_norm, dropout_rate, key=r_keys[i])


    def __call__(self, inputs):
        left = {}
        left[0] = self.left_doubleconvs[0](inputs)
        # print("     - left[0].shape =", left[0].shape)
        for i in range(1, self.levels):
            down = self.downsamplings[i](left[i-1])
            conv = self.left_doubleconvs[i](down)
            left[i] = down + conv
            # if i<self.levels-1:
            #     print(f"     - left[{i}].shape = ", left[i].shape)

        central = left[self.levels-1]
        # print(f"     - central.shape = ", central.shape)

        right = central
        for i in range(self.levels-2, -1,-1):
            up = self.upsamplings[i](right)
            add = left[i] + up
            conv = self.right_doubleconvs[i](add)
            right = up + conv
            # print(f"     - right[{i}].shape =", right.shape)

        return self.final_activation(self.final_conv(right))

############################################################################################################



















# ## Define model and loss function for the learner
class Expert_NCF(eqx.Module):
    """ Expert Neural Context Flow. Can be one that uses array contextts, or functional contexts """
    layers_ctx: list
    activations_ctx: list
    layers_data: list
    activations_data: list
    layers_main: list
    activations_main: list

    ctx_utils:any
    depth_ctx:int
    depth_data:int
    depth_main:int
    intermediate_size:int

    shift_context:bool
    ctx_shift: jnp.ndarray

    def __init__(self, 
                 data_size, 
                 width_main, 
                 depth_main, 
                 depth_data, 
                 depth_ctx, 
                 context_size, 
                 intermediate_size, 
                 ctx_utils=None, 
                 activation="swish", 
                 shift_context=False, 
                 key=None):
        self.ctx_utils = ctx_utils      ## TODO If not none, then use InfDim NCF 

        self.depth_data = depth_data
        self.depth_main = depth_main
        self.depth_ctx = depth_ctx

        ## Activation functions directly from jax
        builtin_fns = {"relu":jax.nn.relu, "tanh":jax.nn.tanh, 'softplus':jax.nn.softplus}

        self.intermediate_size = width_main//2 if intermediate_size is None else intermediate_size

        ## Setup a context processing network with gradual increase/decrease in width
        keys_ctx = jax.random.split(key, num=depth_ctx+2)
        phi = np.linspace(0, 1, depth_ctx+2)
        widths_ctx = [int(context_size * (np.exp(p * np.log(intermediate_size/context_size)))) for p in phi]
        widths_ctx = [context_size] + [int(np.ceil(h/2)*2) for h in widths_ctx[1:-1]] + [intermediate_size]    ## Nearest multiple of 2
        self.activations_ctx = [Swish(key=k) if activation=="swish" else builtin_fns[activation] for k in keys_ctx[:depth_ctx]]
        self.layers_ctx = [eqx.nn.Linear(widths_ctx[i-1], widths_ctx[i], key=keys_ctx[i]) for i in range(1, depth_ctx+2)]

        ## Setup a data processing network with gradual increase/decrease in width
        keys_data = jax.random.split(keys_ctx[-1], num=depth_data+2)
        phi = np.linspace(0, 1, depth_data+2)
        widths_data = [int(data_size * (np.exp(p * np.log(intermediate_size/data_size)))) for p in phi]
        widths_data = [data_size] + [int(np.ceil(h/2)*2) for h in widths_data[1:-1]] + [intermediate_size]
        self.activations_data = [Swish(key=k) if activation=="swish" else builtin_fns[activation] for k in keys_data[:depth_data]]
        self.layers_data = [eqx.nn.Linear(widths_data[i-1], widths_data[i], key=keys_data[i]) for i in range(1, depth_data+2)]

        ## Setup a main processing network with gradualfixed width
        keys_main = jax.random.split(keys_data[-1], num=depth_main+2)
        self.activations_main = [Swish(key=k) if activation=="swish" else builtin_fns[activation] for k in keys_main[:depth_main]]
        self.layers_main = [eqx.nn.Linear(2*intermediate_size, width_main, key=keys_main[1])]
        self.layers_main += [eqx.nn.Linear(width_main, width_main, key=keys_main[i+1]) for i in range(1, depth_main)]
        self.layers_main += [eqx.nn.Linear(width_main, data_size, key=keys_main[depth_main+1])]

        ## Context shift
        self.shift_context = shift_context
        self.ctx_shift = jnp.array([0.])

    def __call__(self, t, y, ctx):

        ## Shift and process the context
        if self.shift_context:
            ctx = ctx + self.ctx_shift
        for layer, activation in zip(self.layers_ctx[:-1], self.activations_ctx):
            ctx = activation(layer(ctx))
        ctx = self.layers_ctx[-1](ctx)

        ## Process the input data
        for layer, activation in zip(self.layers_data[:-1], self.activations_data):
            y = activation(layer(y))
        y = self.layers_data[-1](y)

        ## Merge and process the data and context
        y = jnp.concatenate([y, ctx], axis=0)
        for layer, activation in zip(self.layers_main[:-1], self.activations_main):
            y = activation(layer(y))
        y = self.layers_main[-1](y)

        return y



class CoDARootMLP(eqx.Module):
    network: list
    root_utils: any
    network_size: int     ## The effective/actual size of a root network (flattened neural network)

    def __init__(self, input_dim, output_dim, hidden_size, depth, activation=jax.nn.softplus, key=None):
        key = key if key is not None else jax.random.PRNGKey(0)
        self.network = MLP(input_dim, output_dim, hidden_size, depth, activation, key=key)
        
        props = (input_dim, output_dim, hidden_size, depth, activation)
        params, static = eqx.partition(self.network, eqx.is_array)
        _, shapes, treedef = flatten_pytree(params)
        self.root_utils = (shapes, treedef, static, props)

        self.network_size = sum(x.size for x in jax.tree_util.tree_leaves(params) if x is not None)

    def __call__(self, x):
        return self.network(x)


# ## Define model and loss function for the learner
class Expert_CoDA(eqx.Module):
    """ Expert CoDA, following Kirchmeyer et al. 2022 """
    root_weights: jnp.ndarray
    hyperlayer: list
    root_utils: list

    shift_context: bool
    ctx_shift: jnp.ndarray

    def __init__(self, 
                 data_size, 
                 width, 
                 depth, 
                 context_size, 
                 activation="swish",
                 shift_context=False,
                 key=None):

        keys = jax.random.split(key, num=3)
        builtin_fns = {"relu":jax.nn.relu, "tanh":jax.nn.tanh, 'softplus':jax.nn.softplus}
        act_fn = Swish(key=keys[0]) if activation=="swish" else builtin_fns[activation]

        root = CoDARootMLP(data_size, data_size, width, depth, act_fn, key=keys[1])
        self.root_utils = root.root_utils
        root_params, static = eqx.partition(root.network, eqx.is_array)
        self.root_weights = flatten_pytree(root_params)[0]

        in_hyper, out_hyper = context_size, root.network_size
        self.hyperlayer = eqx.nn.Linear(in_hyper, out_hyper, use_bias=False, key=keys[2])
        ## Initialise these weights to zero
        # self.hyperlayer = jax.tree.map(lambda x: jnp.zeros_like(x), self.hyperlayer)

        self.shift_context = shift_context
        self.ctx_shift = jnp.array([0.])

    def __call__(self, t, y, ctx):

        if self.shift_context:
            ctx = ctx + self.ctx_shift

        delta_arr = self.hyperlayer(ctx)
        final_arr = self.root_weights + delta_arr

        shapes, treedef, static, _ = self.root_utils
        params = unflatten_pytree(final_arr, shapes, treedef)
        root_fun = eqx.combine(params, static)

        return root_fun(y)










class GEPSRootMLP(eqx.Module):
    network: list

    def __init__(self, input_dim, output_dim, hidden_size, depth, activation=jax.nn.softplus, key=None):
        key = key if key is not None else jax.random.PRNGKey(0)
        self.network = MLP(input_dim, output_dim, hidden_size, depth, activation, key=key)

    def __call__(self, x):
        return self.network(x)


# ## Define model and loss function for the learner
class Expert_GEPS(eqx.Module):
    """ Expert GEPS, following Kassaï-Koupaï et al. 2024 """
    root_weights: eqx.Module
    left_weights: eqx.Module        ## A matices
    right_weights: eqx.Module       ## B matrices

    shift_context: bool
    ctx_shift: jnp.ndarray

    def __init__(self, 
                 data_size, 
                 width, 
                 depth, 
                 context_size, 
                 activation="swish",
                 shift_context=False,
                 key=None):

        keys = jax.random.split(key, num=3)
        builtin_fns = {"relu":jax.nn.relu, "tanh":jax.nn.tanh, 'softplus':jax.nn.softplus}
        act_fn = Swish(key=keys[0]) if activation=="swish" else builtin_fns[activation]

        root = GEPSRootMLP(data_size, data_size, width, depth, act_fn, key=keys[1])
        root_params, root_static = eqx.partition(root, eqx.is_array)
        _, root_shapes, root_treedef = flatten_pytree(root_params)
        self.root_weights = root

        r = context_size
        def generate_weights(leaf, key, side="left"):
            """ We form A, B, c such that A@c@B is the weight matrix """

            if leaf is not None:
                if leaf.ndim == 2:
                    d_out, d_in = leaf.shape
                    A = xavier_uniform(key, (d_out, r))
                    B = xavier_uniform(key, (r, d_in))
                    if side=="left":
                        return A
                    elif side=="right":
                        return B
                    else:
                        raise ValueError("Side not recognised")
                elif leaf.ndim == 1:
                    d_out = leaf.shape[0]
                    A = xavier_uniform(key, (d_out, r))
                    B = None
                    if side=="left":
                        return A
                    elif side=="right":
                        return B
                    else:
                        raise ValueError("Side not recognised")

        flat_keys = jax.random.split(keys[2], num=root_treedef.num_leaves)
        root_keys = jax.tree.unflatten(root_treedef, flat_keys)

        self.left_weights = jax.tree.map(partial(generate_weights, side="left"), root_params, root_keys)
        self.right_weights = jax.tree.map(partial(generate_weights, side="right"), root_params, root_keys)

        self.shift_context = shift_context
        self.ctx_shift = jnp.array([0.])

    def __call__(self, t, y, ctx):

        if self.shift_context:
            ctx = ctx + self.ctx_shift

        def multiplication_fn(W, A, B):
            if W.ndim == 2:
                return W + A @ jnp.diag(ctx) @ B
            elif W.ndim == 1:
                return W + A @ ctx

        root_weights_d, root_weights_s = eqx.partition(self.root_weights, eqx.is_array)
        final_params = jax.tree.map(multiplication_fn, root_weights_d, self.left_weights, self.right_weights)
        root_fun = eqx.combine(final_params, root_weights_s)

        return root_fun(y)






# ## Define model and loss function for the learner
class MixER(eqx.Module):
    """ MixER Vector Field. Used for State to Sequence modeling """
    experts: list
    gate:jnp.ndarray

    n_experts: int
    split_context: bool
    meta_learner: str
    use_gate_bias: bool
    gate_update_strategy: str
    is_moe: bool

    def __init__(self, 
                 key, 
                 nb_experts=1,
                 meta_learner="NCF",
                 split_context=False, 
                 same_expert_init=True, 
                 use_gate_bias=True, 
                 gate_update_strategy="least_squares", 
                 **expert_params):

        self.split_context = split_context

        ## The context is now split into tiny chunks for each expert
        context_size = expert_params["context_size"]
        if self.split_context:
            eff_context_size = context_size//nb_experts
        else:
            eff_context_size = context_size

        ## Replace the context_size in the expert_params
        expert_params["context_size"] = eff_context_size

        keys = [key]*nb_experts if same_expert_init else jax.random.split(key, num=nb_experts+1)

        self.meta_learner = meta_learner
        if self.meta_learner == "NCF":
            self.experts = [Expert_NCF(**expert_params, key=keys[i]) for i in range(nb_experts)]
        elif self.meta_learner == "CoDA":
            self.experts = [Expert_CoDA(**expert_params, key=keys[i]) for i in range(nb_experts)]
            pass
        elif self.meta_learner == "GEPS":
            self.experts = [Expert_GEPS(**expert_params, key=keys[i]) for i in range(nb_experts)]
            pass
        else:
            raise ValueError("Meta-learner not recognised !")

        self.use_gate_bias = use_gate_bias
        gate_in_size = context_size+1 if use_gate_bias else context_size
        lim = 1 / np.sqrt(gate_in_size)
        self.gate = jax.random.uniform(keys[-1], (gate_in_size, nb_experts), minval=-lim, maxval=lim)

        self.n_experts = nb_experts
        self.is_moe = True     ## Helps the framework distinguish MoE models
        self.gate_update_strategy = gate_update_strategy

    def gating_function(self, ctx):
        """ Gating function for the MoE model """
        gate_input = jnp.concatenate([ctx, jnp.ones((1,))], axis=0) if self.use_gate_bias else ctx

        if self.gate_update_strategy == "gradient_descent":
            G = self.gate.T @ gate_input
        else:
            G = jax.lax.stop_gradient(self.gate.T) @ gate_input

        return jax.nn.softmax(G)


    def __call__(self, t, y, ctx):
        if self.split_context:
            ctx_pieces = jnp.split(ctx, self.n_experts, axis=0)

        G = self.gating_function(ctx)
        max_G = jnp.max(G)
        dy = jnp.zeros_like(y)
        for i in range(self.n_experts):
            if self.split_context:
                ctx_i = ctx_pieces[i]
            else:
                ctx_i = ctx

            contribution = jax.lax.cond(G[i]>max_G-1e-6, 
                                        lambda in_dat: self.experts[i](*in_dat), 
                                        lambda in_dat: jnp.zeros_like(in_dat[1]), 
                                        (t, y, ctx_i))
            dy += contribution

        return dy





















class RootRNN(eqx.Module):
    root_utils: any
    network_size: int

    def __init__(self, data_size, latent_size, hidden_size, key=None):
        """ Shallow piece-wise linear RNN from Mannuel Brenner et al. 2024. Encoders and Decoders are identiy functions (for now) """
        super().__init__()
        D, L, M = data_size, hidden_size, latent_size

        ## Example leanable params for the hier-plSLRNN
        keys = jax.random.split(key, 7)
        A = xavier_uniform(keys[0], (M, M))
        W1 = xavier_uniform(keys[1], (M, L))
        W2 = xavier_uniform(keys[2], (L, M))
        h2 = xavier_uniform(keys[3], (L,))
        h1 = xavier_uniform(keys[4], (M,))
        alpha = jnp.array([0.1])

        props = (data_size, latent_size, hidden_size, None, None)
        params = (A, W1, W2, h1, h2, alpha)

        _, shapes, treedef = flatten_pytree(params)
        self.root_utils = (shapes, treedef, props)
        self.network_size = sum(x.size for x in jax.tree_util.tree_leaves(params) if x is not None)

    def __call__(self, xs_gt, params):
        """ Predict based on the observation 
        x_gt: (T, D) 
        params: (A, W1, W2, h1, h2, alpha)
        """
        A, W1, W2, h1, h2, alpha = params
        z0 = jnp.zeros(xs_gt.shape[1])

        def f(z, x_gt):
            z_curr = alpha*z + (1-alpha)*x_gt     ## Teacher-Forcing
            z_next = A@z_curr + W1@jax.nn.relu(W2@z_curr + h2) + h1
            return z_next, z_next

        _, zs = jax.lax.scan(f, z0, xs_gt)

        return zs


# ## Define model and loss function for the learner
class Expert_HierShPLRNN(eqx.Module):
    root_network: eqx.Module
    hyperlayer: list

    data_size: int
    latent_size: int

    shift_context:bool
    ctx_shift: jnp.ndarray
    tf_alpha_min: float

    def __init__(self, data_size, latent_size, hidden_size, context_size, shift_context=None, ctx_utils=None, tf_alpha_min=1.0, key=None):
        self.data_size = data_size
        self.latent_size = latent_size

        self.root_network = RootRNN(data_size, data_size, hidden_size, key=key)

        in_hyper, out_hyper = context_size, self.root_network.network_size
        self.hyperlayer = eqx.nn.Linear(in_hyper, out_hyper, key=key, use_bias=False)

        self.shift_context = shift_context
        self.ctx_shift = jnp.array([0.])
        self.tf_alpha_min = tf_alpha_min

    def __call__(self, xts, ctx):
        if self.shift_context:
            ctx = ctx + self.ctx_shift

        subject_weights = self.hyperlayer(ctx)
        subject_weights = subject_weights.at[-1].max(self.tf_alpha_min)    ## Clip the alpha value

        shapes, treedef, _ = self.root_network.root_utils
        subject_params = unflatten_pytree(subject_weights, shapes, treedef)

        return self.root_network(xts[0], subject_params)

# ## Define model and loss function for the learner
class MixER_S2S(eqx.Module):
    """ MixER, but to sequence to sequence models """
    experts: list
    gate:jnp.ndarray

    n_experts: int
    split_context: bool
    meta_learner: str
    use_gate_bias: bool
    gate_update_strategy: str
    is_moe: bool

    def __init__(self, 
                 key, 
                 nb_experts=1,
                 meta_learner="hier-shPLRNN",
                 split_context=False, 
                 same_expert_init=True, 
                 use_gate_bias=True, 
                 gate_update_strategy="least_squares", 
                 **expert_params):

        self.split_context = split_context

        ## The context is now split into tiny chunks for each expert
        context_size = expert_params["context_size"]
        if self.split_context:
            eff_context_size = context_size//nb_experts
        else:
            eff_context_size = context_size

        ## Replace the context_size in the expert_params
        expert_params["context_size"] = eff_context_size

        keys = [key]*nb_experts if same_expert_init else jax.random.split(key, num=nb_experts+1)

        self.meta_learner = meta_learner
        if self.meta_learner == "hier-shPLRNN":
            self.experts = [Expert_HierShPLRNN(**expert_params, key=keys[i]) for i in range(nb_experts)]
            pass
        else:
            raise ValueError("Meta-learner not recognised !")

        self.use_gate_bias = use_gate_bias
        gate_in_size = context_size+1 if use_gate_bias else context_size
        lim = 1 / np.sqrt(gate_in_size)
        self.gate = jax.random.uniform(keys[-1], (gate_in_size, nb_experts), minval=-lim, maxval=lim)

        self.n_experts = nb_experts
        self.is_moe = True     ## Helps the framework distinguish MoE models
        self.gate_update_strategy = gate_update_strategy

    def gating_function(self, ctx):
        """ Gating function for the MoE model """
        gate_input = jnp.concatenate([ctx, jnp.ones((1,))], axis=0) if self.use_gate_bias else ctx

        if self.gate_update_strategy == "gradient_descent":
            G = self.gate.T @ gate_input
        else:
            G = jax.lax.stop_gradient(self.gate.T) @ gate_input

        return jax.nn.softmax(G)

    def __call__(self, xts, ctx):
        if self.split_context:
            ctx_pieces = jnp.split(ctx, self.n_experts, axis=0)

        G = self.gating_function(ctx)
        max_G = jnp.max(G)
        ys = jnp.zeros(xts[0].shape)
        for i in range(self.n_experts):
            if self.split_context:
                ctx_i = ctx_pieces[i]
            else:
                ctx_i = ctx

            contribution = jax.lax.cond(G[i]>max_G-1e-6, 
                                        lambda in_dat: self.experts[i](*in_dat), 
                                        lambda in_dat: jnp.zeros(in_dat[0][0].shape), 
                                        (xts, ctx_i))
            ys += contribution

        return ys
