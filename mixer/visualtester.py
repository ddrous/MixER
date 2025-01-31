
from abc import abstractmethod
from types import SimpleNamespace
from mixer.dataloader import DataLoader, CelebADataLoader, NumpyLoader
from ._utils import *



class VisualTester:
    def __init__(self, trainer, key=None):
        if key is None:
            raise ValueError("Key must be provided for reproducibility.")
        self.key = key

        self.trainer = trainer

    @abstractmethod
    def evaluate(self, 
                 dataloader, 
                 nb_steps=500,
                #  nb_inner_steps=10,
                 print_error_every=(100, 100),
                 loss_criterion=None, 
                 criterion_id=0, 
                 max_adapt_batches=-1, 
                 max_ret_env_states=-1,
                 stochastic=True,
                 taylor_order=0, 
                 val_dataloader=None,
                 verbose=False):
        """
        Adapt and compute test metrics on the adaptation dataloader.
         - loss_criterion if the one used for training is not satisfactory.
         - criterion_id is the index of the desired criterion from the loss auxiliaries
        """

        ## Adapt and extract the losses for each batch of environment
        losses, _, state_data = self.trainer.meta_test(dataloader, 
                                            nb_steps=nb_steps,
                                            # nb_inner_steps=nb_inner_steps, 
                                            max_adapt_batches=max_adapt_batches,
                                            print_error_every=print_error_every,
                                            taylor_order=taylor_order, 
                                            val_dataloader=val_dataloader,
                                            max_ret_env_states=max_ret_env_states,
                                            stochastic=stochastic,
                                            verbose=verbose)

        # ## losses: (nb_inner_steps, nb_criterions)
        # ## state_data: (X, Y, Y_hat)
        # ## Y, Y_hat: (envs, trajs_per_envs, steps_per_traj, data_size)

        ## If adaptation and reuse_contexts, then set the context_adapt addtribute
        if dataloader.dataset.adaptation and self.trainer.learner.reuse_contexts:
            self.trainer.learner.contexts_adapt = self.trainer.learner.contexts_latest

        ## Compute the confidence intervals on the losses
        _, Y, Y_hat = state_data
        # TODO fix this cleanly plz - print("State data is: let's analysse it ...", jax.tree_map(lambda x: x.shape, state_data))
        if loss_criterion is None:
            axis = (-1, -2, -3) if len(Y.shape)>3 else (-1, -2)
            loss_criterion = lambda y, y_hat: jnp.mean((y - y_hat)**2, axis=axis)
        test_means = jax.vmap(loss_criterion, in_axes=(0, 0))(Y, Y_hat)
        test_mean, test_std = jnp.mean(test_means), jnp.std(test_means)
        # test_mean, test_std = jnp.median(test_means), jnp.std(test_means)   ## TODO: careful with MEDIAN

        ## Gather a metric from the training losses
        losses_means = losses[-1, :]
        # losses_means = jnp.mean(losses, axis=0)
        # losses_means = jnp.min(losses, axis=0)

        ## TODO Add the environment-wide UQ from NCF aware testing
        aux_losses = None

        train_mean = losses_means[criterion_id+1]
        if verbose:
            # print("\n==  Meta-Evaluation ... ==")
            print(f"\n    Test loss value: {test_mean:.3e} ± {test_std:.3e}")
            print(f"    Train loss value for criterion {criterion_id}: {train_mean:.3e}")

        # return mean_loss, None
        return test_mean, (test_means, test_std, train_mean, aux_losses)


    @abstractmethod
    def visualize_train_val(self, dataloader, few_shot_loader, save_path=False, environment=None, key=None):
        """ Visualize two samples and their predictions: one from training and the other from validation """
        ## The dataloader muct be a generator of length 2. One containing training data and the second validation data.
        pass

    @abstractmethod
    def visualize_artefacts(self, adaptation=False, save_path=False, key=None, ylim=None):
        """ Visualize the artefacts of the model : loss, and context dimensions """
        key = key if key != None else self.key

        ## Context dimensions to plot: 3 along x, 3 along y
        ctx_x_key, ctx_y_key = jax.random.split(key, num=2)
        ctx_dims_x = jax.random.randint(ctx_x_key, (3,), 1, self.trainer.learner.context_size)-1
        ctx_dims_y = jax.random.randint(ctx_y_key, (3,), 0, self.trainer.learner.context_size-1)+1

        print("\n==  Begining artefacts visualisation ... ==")
        print("    Visualized context dimensions along x:", ctx_dims_x)
        print("    Visualized context dimensions along y:", ctx_dims_y)

        fig, ax = plt.subplot_mosaic('DDD;EFG', figsize=(4*3, 3.7*2))

        losses_model = np.vstack(self.trainer.losses_model)
        losses_ctx = np.vstack(self.trainer.losses_ctx)

        if hasattr(self.trainer.learner, 'contexts'):
            xis = self.trainer.learner.contexts.params
        else:
            print("No contexts found. Using zeros.")
            xis = jnp.zeros((10, self.trainer.learner.context_size))

        if adaptation == True:  ## Overwrite the above if adaptation
            losses_model = np.vstack(self.trainer.losses_adapt)
            losses_ctx = np.vstack(self.trainer.losses_adapt)
            if hasattr(self.trainer.learner, 'contexts_adapt'):
                xis = self.trainer.learner.contexts_adapt.params
            elif hasattr(self.trainer.learner, 'contexts_latest'):
                print("No adaptation contexts found. Using latest found.")
                xis = self.trainer.learner.contexts_latest.params
            else:
                print("No contexts found. Using zeros.")
                xis = jnp.zeros((10, self.trainer.learner.context_size))

        mke = np.ceil(losses_model.shape[0]/100).astype(int)
        mks = 2

        label_model = "Model Loss" if adaptation == False else "Model Loss Adapt"
        ax['D'].plot(losses_model[:,0], label=label_model, color="grey", linewidth=3, alpha=1.0)
        label_ctx = "Context Loss" if adaptation == False else "Context Loss Adapt"
        ax['D'].plot(losses_ctx[:,0], "x-", markevery=mke, markersize=mks, label=label_ctx, color="grey", linewidth=1, alpha=0.5)

        if adaptation==False and hasattr(self.trainer, 'val_losses') and len(self.trainer.val_losses)>0:
            val_losses = np.vstack(self.trainer.val_losses)
            ax['D'].plot(val_losses[:,0], val_losses[:,1], "y.", label="Validation Loss", linewidth=3, alpha=0.5)

        ax['D'].set_xlabel("Iterations")
        ax['D'].set_title("Loss Terms")
        ax['D'].set_yscale('log')
        ax['D'].legend()
        if ylim is not None:
            ax['D'].set_ylim(ylim)

        colors = ['dodgerblue', 'crimson', 'darkgreen', 'purple', 'brown']
        ax['E'].scatter(xis[:,ctx_dims_x[0]], xis[:,ctx_dims_y[0]], s=30, c=colors[0], marker='X')
        ax['F'].scatter(xis[:,ctx_dims_x[1]], xis[:,ctx_dims_y[1]], s=50, c=colors[1], marker='o')
        ax['G'].scatter(xis[:,ctx_dims_x[2]], xis[:,ctx_dims_y[2]], s=60, c=colors[2], marker='+')

        ax['E'].set_title(f'dim {ctx_dims_y[0]} vs dim {ctx_dims_x[0]}')
        ax['E'].set_xlabel(f'dim {ctx_dims_x[0]}')
        ax['E'].set_ylabel(f'dim {ctx_dims_y[0]}')

        ax['F'].set_title(f'dim {ctx_dims_y[1]} vs dim {ctx_dims_x[1]}')
        ax['F'].set_xlabel(f'dim {ctx_dims_x[1]}')
        ax['F'].set_ylabel(f'dim {ctx_dims_y[1]}')

        ax['G'].set_title(f'dim {ctx_dims_y[2]} vs dim {ctx_dims_x[2]}')
        ax['G'].set_xlabel(f'dim {ctx_dims_x[2]}')
        ax['G'].set_ylabel(f'dim {ctx_dims_y[2]}')

        plt.suptitle(f"Losses and Context Vectors", fontsize=20)

        plt.tight_layout()
        # plt.show();
        plt.draw();

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print("Saving artefacts in:", save_path, flush=True);



    @abstractmethod
    def visualize_context_clusters(self, perplexities=(15, 5), save_path=False, key=None):
        """ Visualize the context clusters of the model with a dimensionality reduction technique """
        key = key if key != None else self.key

        xis_train = self.trainer.learner.contexts
        if hasattr(self.trainer.learner, 'contexts_adapt'):
            xis_adapt = self.trainer.learner.contexts_adapt
        # elif hasattr(self.trainer.learner, 'contexts_latest'):
        #     xis_adapt = self.trainer.learner.contexts_latest
        # else:
        #     print("No contexts found. Using zeros.")
        #     xis_adapt = jnp.zeros((10, self.trainer.learner.context_size))
        else:
            xis_adapt = SimpleNamespace(params=jnp.zeros((0, self.trainer.learner.context_size)))

        print("\n==  Begining context clusters visualisation ... ==")

        fig, ax = plt.subplot_mosaic('ABC;DEF', figsize=(4*3, 3.7*2))

        print("    Visualising the parameters (either vectors or function weights)")

        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=3, perplexity=perplexities[0], random_state=int(key[0]))
        X = jnp.concatenate([xis_train.params, xis_adapt.params], axis=0)
        if X.shape[1] > 2:
            X_embedded = reducer.fit_transform(X)
        else:
            X_embedded = X

        nb_train_envs = xis_train.params.shape[0]
        nb_total_envs = X.shape[0]

        dim0s = [0, 0, 1]
        dim1s = [1, 2, 2]
        plot_ids = ['A', 'B', 'C']
        markers = ['X', 's', '+', 'o']
        mkss = [30, 30, 30, 20]
        colors = ['dodgerblue', 'orange', 'limegreen', 'purple']

        for j in range(3):
            dim0, dim1, p_id = dim0s[j], dim1s[j], plot_ids[j]

            ax[p_id].scatter(X_embedded[:nb_train_envs, dim0], X_embedded[:nb_train_envs, dim1], label='Training', marker=markers[j], s=mkss[j], color=colors[j])
            ax[p_id].scatter(X_embedded[nb_train_envs:, dim0], X_embedded[nb_train_envs:, dim1], label='Adaptation', marker=markers[-1], s=mkss[-1], color=colors[-1])

            if nb_total_envs <= 20: ## Otherwise it gets too cluttered
                for i in range(nb_train_envs):
                    ax[p_id].text(X_embedded[i, dim0]-10, X_embedded[i, dim1]+5, str(i), fontsize=10, ha='center', va='center')
                for i in range(nb_train_envs, nb_total_envs):
                    ax[p_id].text(X_embedded[i, dim0]+10, X_embedded[i, dim1]+10, str(i-nb_train_envs), fontsize=10, ha='left', va='top')

            # ax[p_id].set_xlabel(f't-SNE ${dim0}$')
            # ax[p_id].set_ylabel(f't-SNE ${dim1}$')
            ax[p_id].set_title(f"t-SNE ${dim0}$ vs t-SNE ${dim1}$" if X.shape[1] > 2 else f"dim ${dim0}$ vs dim ${dim1}$")
            ax[p_id].legend(fontsize=8, loc='upper right')

        print("    Visualising predictions from the contexts (reconstituted as functions)")


        model = self.trainer.learner.model
        if hasattr(model, 'vectorfield'):
            model_ = model.vectorfield.neuralnet
        else:
            model_ = model.neuralnet
        
        ts = jnp.array([0, 1, 10])[:, None]
        if hasattr(model_, 'ctx_utils') and model_.ctx_utils is not None:
            ctx_utils = model_.ctx_utils
            @eqx.filter_vmap
            def ctx_arr_to_fun_eval(ctx_arr):
                ctx_shapes, ctx_treedef, ctx_static, _ = ctx_utils
                ctx_params = unflatten_pytree(ctx_arr, ctx_shapes, ctx_treedef)
                ctx_fun = eqx.combine(ctx_params, ctx_static)
                rets = jnp.stack([ctx_fun(ts[j]) for j in range(3)], axis=1)
                return rets
            train_dat = ctx_arr_to_fun_eval(xis_train.params)
            adapt_dat = ctx_arr_to_fun_eval(xis_adapt.params)
        else:
            print("WARNING: No context utilities found, meaning inf dim context are not used. Using randoms.")
            train_dat = jax.random.normal(key, (nb_train_envs, 8, 3))
            adapt_dat = jax.random.normal(key, (nb_total_envs-nb_train_envs, 8, 3))

        plot_ids = ['D', 'E', 'F']
        colors = ['royalblue', 'orangered', 'darkgreen', 'darkviolet']
        for j, (t, p_id) in enumerate(zip(ts[:,0].tolist(), plot_ids)):
            X = jnp.concatenate([train_dat[..., j], adapt_dat[..., j],], axis=0)
            if perplexities[1] > 0 and X.shape[1] > 2:
                reducer = TSNE(n_components=2, perplexity=perplexities[1], random_state=int(key[1]))
                X_embedded = reducer.fit_transform(X)
                dim0, dim1 = 0, 1
            else:   ## We don't want the reduce anything
                if j==0: print(f"   WARNING: invalid perplexity {perplexities[1]}. Skipping reduction.")
                X_embedded = X
                key, _ = jax.random.split(key)
                dim0, dim1 = jax.random.randint(key, (2,), 0, X.shape[1])
                if dim0==dim1: dim1 = (dim1+1) % X.shape[1]

            if hasattr(model_, 'ctx_utils') and model_.ctx_utils is not None:
                ax[p_id].scatter(X_embedded[:nb_train_envs, dim0], X_embedded[:nb_train_envs, dim1], label='Training', marker=markers[j], s=mkss[j], color=colors[j])
                ax[p_id].scatter(X_embedded[nb_train_envs:, dim0], X_embedded[nb_train_envs:, dim1], label='Adaptation', marker=markers[-1], s=mkss[-1], color=colors[-1])

            # if nb_total_envs <= 20:
            #     for i in range(nb_train_envs):
            #         ax[p_id].text(X_embedded[i, dim0], X_embedded[i, dim1], str(i), fontsize=10, ha='center', va='center')
            #     for i in range(nb_train_envs, nb_total_envs):
            #         ax[p_id].text(X_embedded[i, dim0]+20, X_embedded[i, dim1]+20, str(i-9), fontsize=10, ha='left', va='top')

            if perplexities[1] > 0 and X.shape[1] > 2:
                ax[p_id].set_xlabel(f't-SNE ${dim0}$', fontsize=8)
                ax[p_id].set_ylabel(f't-SNE ${dim1}$', fontsize=8)
            else:
                ax[p_id].set_xlabel(f'dim ${dim0}$', fontsize=10)
                ax[p_id].set_ylabel(f'dim ${dim1}$', fontsize=10)

            ax[p_id].set_title(f"$t$={t}", fontsize=14)
            ax[p_id].legend(fontsize=8, loc='upper right')

        plt.suptitle(f"Context Embeddings (top, 3D) - Evaluations (bottom, 2D)", fontsize=20)

        plt.tight_layout()
        # plt.show();
        plt.draw();

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print("Saving context clusters in:", save_path, flush=True);






#%%

class CelebAVisualTester(VisualTester):
    def __init__(self, trainer, key=None):
        super().__init__(trainer, key)

    def visualize_few_shots(self, 
                        few_shots_loader:DataLoader, 
                        all_shots_loader:DataLoader, 
                        nb_steps=10,
                        save_path=False, 
                        key=None):
        key = key if key != None else self.key

        print("\n==  Begining in-domain CelebA visualisation ... ==")

        ## The contexts are not obtained from a quick adaptation process (hidden in meta-test)
        if isinstance(all_shots_loader, CelebADataLoader):
            e = jax.random.randint(key, (1,), 0, few_shots_loader.nb_batches)[0]
            X, Y = all_shots_loader.sample_environments(key, e, 1)
        elif isinstance(all_shots_loader, NumpyLoader):
            e = jax.random.randint(key, (1,), 0, len(few_shots_loader.dataset))[0]
            X, Y = all_shots_loader.dataset.set_seed_sample_pixels(key[0], e)
            X, Y = X[None, ...], Y[None, ...]
        else:
            raise ValueError("Invalid dataloader class instance provided.")

        if isinstance(few_shots_loader, CelebADataLoader):
            img_size = few_shots_loader.img_size
            X_few_shots, Y_few_shots = few_shots_loader.sample_environments(key, e, 1)
        elif isinstance(few_shots_loader, NumpyLoader):
            img_size = few_shots_loader.dataset.img_size
            X_few_shots, Y_few_shots = few_shots_loader.dataset.set_seed_sample_pixels(key[0], e)
            X_few_shots, Y_few_shots = X_few_shots[None, ...], Y_few_shots[None, ...]
        else:
            raise ValueError("Invalid dataloader class instance provided.")

        print("    Environment (batch) id:", e)

        _, _, (X, Y, Y_hat) = self.trainer.meta_test(dataloader=[(X_few_shots, Y_few_shots)], 
                                                     nb_steps=nb_steps, 
                                                     max_ret_env_states=1,
                                                     val_dataloader=[(X, Y)],
                                                     verbose=False)
        X_hat, Y_true, Y_hat = X[0], Y[0], Y_hat[0]
        X_few_shots, Y_few_shots = X_few_shots[0], Y_few_shots[0]

        fig, ax = plt.subplot_mosaic('ABC', figsize=(4*3, 3.7*1))

        def make_image(xy_coords, rgb_pixels):
            img = np.zeros(img_size)
            x_coords = (xy_coords[:, 0] * img_size[0]).astype(int)
            y_coords = (xy_coords[:, 1] * img_size[1]).astype(int)
            img[x_coords, y_coords, :] = np.clip(rgb_pixels, 0., 1.)
            return img

        true_img = make_image(X_hat, Y_true)
        ax['A'].imshow(true_img)
        ax['A'].set_title('True', fontsize=14)

        few_shoot_img = make_image(X_few_shots, Y_few_shots)
        ax['B'].imshow(few_shoot_img)
        ax['B'].set_title('Few-shots', fontsize=14)

        pred_img = make_image(X_hat, Y_hat)
        ax['C'].imshow(pred_img)
        ax['C'].set_title('Predicted', fontsize=14)


        plt.suptitle(f"Sample Predictions", fontsize=20)

        plt.tight_layout()
        # plt.show();
        plt.draw();

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print("Saving visualization in:", save_path, flush=True);




    def visualize_few_shots_multi(self, 
                                few_shots_loader:DataLoader, 
                                all_shots_loader:DataLoader, 
                                nb_steps=10,
                                num_envs=6,
                                save_path=False, 
                                key=None):
        key = key if key != None else self.key

        print("\n==  Begining in-domain CelebA visualisation ... ==")

        ## The contexts are not obtained from a quick adaptation process (hidden in meta-test)
        if isinstance(all_shots_loader, CelebADataLoader):
            e = jax.random.randint(key, (1,), 0, few_shots_loader.nb_batches)[0]
            X, Y = all_shots_loader.sample_environments(key, e, num_envs)
            print("    Environment (batch) id:", e)
        elif isinstance(all_shots_loader, NumpyLoader):
            keys = jax.random.split(key, num=num_envs)
            batches = [all_shots_loader.dataset.set_seed_sample_pixels(keys[e, 0], e) for e in range(num_envs)]
            # batches = [all_shots_loader.dataset.__getitem__(e) for e in range(num_envs)]
            X = jnp.stack([b[0] for b in batches])
            Y = jnp.stack([b[1] for b in batches])
            print("    Environment ids:", range(num_envs))
        else:
            raise ValueError("Invalid dataloader class instance provided.")

        if isinstance(few_shots_loader, CelebADataLoader):
            img_size = few_shots_loader.img_size
            X_few_shots, Y_few_shots = few_shots_loader.sample_environments(key, e, num_envs)
        elif isinstance(few_shots_loader, NumpyLoader):
            img_size = few_shots_loader.dataset.img_size
            keys = jax.random.split(key, num=num_envs)
            batches = [few_shots_loader.dataset.set_seed_sample_pixels(keys[e, 0], e) for e in range(num_envs)]
            # batches = [few_shots_loader.dataset.__getitem__(e) for e in range(num_envs)]
            X_few_shots = jnp.stack([b[0] for b in batches])
            Y_few_shots = jnp.stack([b[1] for b in batches])
        else:
            raise ValueError("Invalid dataloader class instance provided.")

        _, _, (X_hat, _, Y_hat) = self.trainer.meta_test(dataloader=[(X_few_shots, Y_few_shots)], 
                                                     nb_steps=nb_steps, 
                                                     max_ret_env_states=num_envs,
                                                     val_dataloader=[(X, Y)],
                                                     verbose=False)

        fig, ax = plt.subplots(num_envs, 3, figsize=(4*3, 3.7*num_envs))

        def make_image(xy_coords, rgb_pixels):
            img = np.zeros(img_size)
            x_coords = (xy_coords[:, 0] * img_size[0]).astype(int)
            y_coords = (xy_coords[:, 1] * img_size[1]).astype(int)
            img[x_coords, y_coords, :] = np.clip(rgb_pixels, 0., 1.)
            return img

        for e in range(num_envs):
            true_img = make_image(X[e], Y[e])
            ax[e, 0].imshow(true_img)

            few_shoot_img = make_image(X_few_shots[e], Y_few_shots[e])
            ax[e, 1].imshow(few_shoot_img)

            pred_img = make_image(X_hat[e], Y_hat[e])
            ax[e, 2].imshow(pred_img)

            ## Remove axis
            ax[e, 0].set_xticks([])
            ax[e, 0].set_yticks([])
            ax[e, 1].set_xticks([])
            ax[e, 1].set_yticks([])
            ax[e, 2].set_xticks([])
            ax[e, 2].set_yticks([])

            if e==0:
                ax[e, 0].set_title('True', fontsize=20)
                ax[e, 1].set_title('Few-shots', fontsize=20)
                ax[e, 2].set_title('Predicted', fontsize=20)

        plt.suptitle(f"Sample Predictions", fontsize=30, y=1.003)

        plt.tight_layout()
        # plt.show();
        plt.draw();

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print("Saving visualization in:", save_path, flush=True);








    def visualize_few_shots_multi_uq(self, 
                                few_shots_loader:DataLoader, 
                                all_shots_loader:DataLoader, 
                                nb_steps=10,
                                num_envs=6,
                                taylor_order=2,
                                uq_train_contexts=10,
                                interp_method='linear',
                                save_path=False, 
                                key=None):
        key = key if key != None else self.key

        print("\n==  Begining in-domain CelebA visualisation ... ==")

        ## The contexts are not obtained from a quick adaptation process (hidden in meta-test)
        if isinstance(all_shots_loader, CelebADataLoader):
            e = jax.random.randint(key, (1,), 0, few_shots_loader.nb_batches)[0]
            X, Y = all_shots_loader.sample_environments(key, e, num_envs)
            print("    Environment (batch) id:", e)
        elif isinstance(all_shots_loader, NumpyLoader):
            keys = jax.random.split(key, num=num_envs)
            batches = [all_shots_loader.dataset.set_seed_sample_pixels(keys[e, 0], e) for e in range(num_envs)]
            # batches = [all_shots_loader.dataset.__getitem__(e) for e in range(num_envs)]
            X = jnp.stack([b[0] for b in batches])
            Y = jnp.stack([b[1] for b in batches])
            print("    Environment ids:", range(num_envs))
        else:
            raise ValueError("Invalid dataloader class instance provided.")

        if isinstance(few_shots_loader, CelebADataLoader):
            img_size = few_shots_loader.img_size
            X_few_shots, Y_few_shots = few_shots_loader.sample_environments(key, e, num_envs)
        elif isinstance(few_shots_loader, NumpyLoader):
            img_size = few_shots_loader.dataset.img_size
            keys = jax.random.split(key, num=num_envs)
            batches = [few_shots_loader.dataset.set_seed_sample_pixels(keys[e, 0], e) for e in range(num_envs)]
            # batches = [few_shots_loader.dataset.__getitem__(e) for e in range(num_envs)]
            X_few_shots = jnp.stack([b[0] for b in batches])
            Y_few_shots = jnp.stack([b[1] for b in batches])
        else:
            raise ValueError("Invalid dataloader class instance provided.")

        _, _, _ = self.trainer.meta_test(dataloader=[(X_few_shots, Y_few_shots)], 
                                        nb_steps=nb_steps, 
                                        max_ret_env_states=1,
                                        verbose=False)
        contexts = self.trainer.learner.contexts_latest     ## A bit dangerous, but it's fine for now !

        ## Reset the model to taylor_oder
        model = self.trainer.learner.reset_model(taylor_order, verbose=True)

        ## Do a batch predict multi
        # X, Y, Y_hat = self.trainer.learner.batch_predict_multi(model, contexts, (X, Y), max_envs=num_envs)
        X, Y, Y_hat = self.trainer.learner.batch_predict_multi(model, 
                                                               contexts, 
                                                               (X, Y), 
                                                               max_envs=num_envs, 
                                                               uq_train_contexts=uq_train_contexts)

        X_hat, Y_true, Y_hat = X, Y, Y_hat

        fig, ax = plt.subplots(num_envs, 5, figsize=(4*5, 3.7*num_envs))
        if num_envs==1: ax = ax[None, ...]

        for e in range(num_envs):
            true_img = make_image(X_hat[e], Y_true[e], img_size)
            ax[e, 0].imshow(true_img)

            few_shoot_img = make_image(X_few_shots[e], Y_few_shots[e], img_size)
            ax[e, 1].imshow(few_shoot_img)

            # pred_img = make_image(X_hat[e], Y_hat[e, e])
            pred_img = make_image(X_hat[e], Y_hat[e, 0], img_size)    ## The perfectest image is always the first
            ax[e, 2].imshow(pred_img)

            uncertainty = make_image(X_hat[e], jnp.std(Y_hat[e], axis=0), img_size)
            # ax[e, 3].imshow(uncertainty, cmap="grey")
            ax[e, 3].imshow(uncertainty)

            interpolation = interpolate_2D_image(np.asarray(X_few_shots[e]), np.asarray(Y_few_shots[e]), img_size, method=interp_method)
            ax[e, 4].imshow(interpolation)

            ## Remove ticks
            ax[e, 0].set_xticks([])
            ax[e, 0].set_yticks([])
            ax[e, 1].set_xticks([])
            ax[e, 1].set_yticks([])
            ax[e, 2].set_xticks([])
            ax[e, 2].set_yticks([])
            ax[e, 3].set_xticks([])
            ax[e, 3].set_yticks([])
            ax[e, 4].set_xticks([])
            ax[e, 4].set_yticks([])

            if e==0:
                ax[e, 0].set_title('True', fontsize=20)
                ax[e, 1].set_title('Few-shots', fontsize=20)
                ax[e, 2].set_title('Predicted', fontsize=20)
                ax[e, 3].set_title('Uncertainty', fontsize=20)
                ax[e, 4].set_title(interp_method.capitalize()+" Int.", fontsize=20)

        plt.suptitle(f"Sample Predictions", fontsize=30, y=1.003)

        plt.tight_layout()
        # plt.show();
        plt.draw();

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print("Saving visualization in:", save_path, flush=True);










class SineVisualTester(VisualTester):
    def __init__(self, trainer, key=None):
        super().__init__(trainer, key)


class DynamicsVisualTester(VisualTester):
    def __init__(self, trainer, key=None):
        super().__init__(trainer, key)


    def visualize_dynamics(self, 
                           data_loader, 
                           traj,
                           dims=(0,1), 
                           nb_envs=-1,
                           envs=None,
                           share_axes=True,
                           save_path=False, 
                           key=None):

        """ Visualize the dynamics of the model on a single trajectory from all environments. envs is the actual envs ids to plot, ignored if nb_envs is set. """

        key = key if key != None else jax.random.PRNGKey(time.time_ns())
        traj = traj if traj is not None else jax.random.randint(key, (1,), 0, data_loader.dataset.num_shots)[0]

        t_test = data_loader.dataset.t_eval
        batch = next(iter(data_loader))

        if nb_envs ==-1 and envs is None:
            raise ValueError("Please provide either the number of environments to plot or the actual envs ids.")
        if nb_envs > 0 and envs is not None:
            raise ValueError("Please provide either the number of environments to plot or the actual envs ids, not both.")
        # total_envs = data_loader.dataset.total_envs
        total_envs = batch[1].shape[0]

        if envs is None:
            nb_envs = nb_envs if (nb_envs > 0 and nb_envs < total_envs) else total_envs
        else:
            nb_envs = len(envs) if len(envs) < total_envs else total_envs
            if np.max(envs) >= total_envs:
                raise ValueError("One of the provided envs ids is out of bounds.")

        if data_loader.dataset.adaptation == False:
            print("\n==  Begining in-domain dynamics visualisation ... ==")
        else:
            print("\n==  Begining out-of-distribution dynamics visualisation ... ==")
        print("    Trajectory id:", traj)
        print("    Visualized dimensions:", dims)

        ## TODO check is learner.reset_model is True, otherwise, relearn the contexts
        ## Dynamics models are handled in a single batch, so the saved contexts can be reused
        if data_loader.dataset.adaptation == False:
            print("    Using contexts from meta-training.")
            contexts = self.trainer.learner.contexts
        else:
            if hasattr(self.trainer.learner, 'contexts_adapt'):
                print("    Using contexts from meta-testing.")
                contexts = self.trainer.learner.contexts_adapt
            elif hasattr(self.trainer.learner, 'contexts_latest'):
                print("WARNING: No specific adaptation contexts found. Using latest found.")
                contexts = self.trainer.learner.contexts_latest
            else:
                raise ValueError("No contexts found for adaptation. Please adapt the model first.")

        # model = self.trainer.learner.model
        model = self.trainer.learner.reset_model(taylor_order=0, verbose=False)
        _, X, X_hat = self.trainer.learner.batch_predict(model, contexts, batch)

        # print("Are THERE anu NANs in X_hat?", jnp.any(jnp.isnan(X_hat)))

        ## Select nb_envs indices at random from the total_envs
        if envs is None:
            plot_envs = jax.random.choice(key, total_envs, (nb_envs,), replace=False)
        else:
            plot_envs = jnp.array(envs)

        # X_hat = X_hat[:,traj,...]
        # X = X[:,traj,...]

        X_hat = X_hat[plot_envs, traj, ...]
        X = X[plot_envs, traj, ...]
        t_test = t_test[plot_envs] if t_test.ndim > 1 else t_test

        # fig, ax = plt.subplots(nb_envs, 2, figsize=(5*2, 3*nb_envs), sharex=False, sharey=False)
        # fig, ax = plt.subplots(nb_envs, 2, figsize=(4*2, 2*nb_envs), sharex=False, sharey=False, gridspec_kw = {'wspace':0.17, 'hspace':0.15})
        fig, ax = plt.subplots(nb_envs, 2, figsize=(4*2, 2*nb_envs), sharex=False, sharey=False)
        if nb_envs ==1:
            ax = ax[None, ...]

        mks = 4
        dim0, dim1 = dims

        xlim_0 = np.min([np.min(X[...,dim0]), np.min(X_hat[...,dim0])])
        xlim_1 = np.max([np.max(X[...,dim0]), np.max(X_hat[...,dim0])])
        ylim_0 = np.min([np.min(X[...,dim1]), np.min(X_hat[...,dim1])])
        ylim_1 = np.max([np.max(X[...,dim1]), np.max(X_hat[...,dim1])])
        eps = 0.1

        for e in range(nb_envs):
            t_plot = t_test[e] if t_test.ndim > 1 else t_test

            ax[e, 0].plot(t_plot, X_hat[e, :, dim0], "o", c="royalblue", label=f"$\\hat{{x}}_{{{dim0}}}$ (Pred)", markersize=mks)
            ax[e, 0].plot(t_plot, X[e, :, dim0], c="deepskyblue", label=f"$x_{{{dim0}}}$ (GT)")

            ax[e, 0].plot(t_plot, X_hat[e, :, dim1], "x", c="purple", label=f"$\\hat{{x}}_{{{dim1}}}$ (Pred)", markersize=mks+1)
            ax[e, 0].plot(t_plot, X[e, :, dim1], c="violet", label=f"$x_{{{dim1}}}$ (GT)")

            ax[e, 1].plot(X_hat[e, :, dim0], X_hat[e, :, dim1], ".", c="teal", label="Pred")
            ax[e, 1].plot(X[e, :, dim0], X[e, :, dim1], c="turquoise", label="GT")

            if e==nb_envs-1: ax[e, 0].set_xlabel("Time")
            elif share_axes==True : ax[e, 0].set_xticklabels([])
            if e==0: ax[e, 0].legend(title=f"Env {plot_envs[e]}", loc='upper right')
            else: ax[e, 0].legend([], title=f"Env {plot_envs[e]}", loc='upper right')
            ax[e, 0].set_ylabel("State")

            if e==nb_envs-1: ax[e, 1].set_xlabel(f"$x_{{{dim0}}}$")
            elif share_axes==True : ax[e, 1].set_xticklabels([])
            if e==0: ax[e, 1].legend(title=f"Env {plot_envs[e]}", loc='upper right')
            else: ax[e, 1].legend([], title=f"Env {plot_envs[e]}", loc='upper right')
            ax[e, 1].set_ylabel(f"$x_{{{dim1}}}$")

            if share_axes==True:
                ax[e, 0].set_ylim(min(xlim_0, ylim_0)-eps, max(xlim_1, ylim_1)+eps)
                ax[e, 1].set_xlim(xlim_0-eps, xlim_1+eps)
                ax[e, 1].set_ylim(ylim_0-eps, ylim_1+eps)

        # plt.subplots_adjust(wspace=0, hspace=0)

        plt.tight_layout()
        plt.suptitle(f"Trajectories and Phase Spaces for Trajectories Id {traj}", fontsize=16, y=1.001)
        # plt.suptitle(f"Trajectories and Phase Spaces for Trajectory Id {traj}", fontsize=16)

        plt.draw();

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print("Testing finished. Figure saved in:", save_path);


  