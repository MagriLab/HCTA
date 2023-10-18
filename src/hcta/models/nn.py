import pathlib as Path

import numpy as np
import tensorflow as tf

from ..utils.train import print_status_bar, sample


# @todo:
# check for the passed network structures (number of inputs and outputs)
class BaseNN:
    def __init__(self, model):
        """Encompassing neural network structure
        model: tf model
        rijke: rijke model whose properties are used in the physics-informed training
               (defaults to None)
        """
        self.model = model
        # Instantiate an optimizer.
        self.optimizer = tf.keras.optimizers.Adam()

        # Instantiate a loss function.
        self.loss_fn = tf.keras.losses.mean_squared_error

    @tf.function
    def evaluate(self, input, output, training, observe="both"):
        # predict using the model
        output_pred = self.predict(input, training=training)

        # Compute mse loss for the prediction (data-driven loss)
        # split the variables
        p_pred = output_pred[:, 0]
        u_pred = output_pred[:, 1]
        p = output[:, 0]
        u = output[:, 1]
        # we change the data-driven loss depending on which variable(s)
        # have been observed, even though we predict for both
        if observe == "both":  # if both pressure and velocity are observed
            dd_loss_value = tf.reduce_mean(self.loss_fn(output, output_pred))
        elif observe == "p":  # if only pressure is observed
            dd_loss_value = tf.reduce_mean(self.loss_fn(p[:, None], p_pred[:, None]))
        elif observe == "u":  # if only velocity is observed
            dd_loss_value = tf.reduce_mean(self.loss_fn(u[:, None], u_pred[:, None]))

        # calculate the total loss including the regularisation losses in the layers
        loss_value = tf.add_n([dd_loss_value] + self.model.losses)
        return loss_value, output_pred

    # @codereview: put evaluate_pi so people know they should implement it

    @tf.function
    def train_step(self, input, output, lr, observe="both"):
        with tf.GradientTape() as tape:
            loss_value, output_pred = self.evaluate(
                input, output, training=True, observe=observe
            )
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.learning_rate = lr
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss_value, output_pred

    @tf.function
    def train_step_PI(self, input, output, input_batch_size, lr, observe="both"):
        with tf.GradientTape() as tape:
            loss_value, output_pred = self.evaluate_PI(
                input, output, input_batch_size, training=True, observe=observe
            )
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.learning_rate = lr
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss_value, output_pred

    def train(
        self,
        log_dir,
        train_dataset,
        val_interp_dataset,
        val_extrap_dataset,
        epochs,
        save_epochs,
        print_epoch_mod,
        lr_schedule,
        lambda_dd=1,
        lambda_m=0,
        lambda_e=0,
        sampled_batch_size=32,
        sampled_domain={"x": (0, 1), "t_train": (0, 4), "t_val": (4, 8)},
        observe="both",
    ):
        """
        Trains any model of BaseNN
        Args:
            log_dir: where to save the model weights
            train_dataset: training tf dataset
            val_interp_dataset, val_extrap_dataset: validation tf datasets for interpolation and extrapolation
            epochs: number of epochs
            save_epochs: which epochs to save
            print_epochs_mod: epoch frequency at which to print the metrics
            lr_schedule: learning rate schedule
            lambda_dd: weighting of the data-driven loss
            lambda_m: weighting of the momentum loss
            lambda_e: weighting of the energy loss
            sampled_batch_size: batch size of the sampled input points for physics loss evaluation
            sampled_domain: domain of the sampling for physics loss evaluation
        Returns:
            history: History dictionary that contains the evolution of metrics
        """
        # Check if the training is physics-informed or not
        self.lambda_dd = lambda_dd
        self.lambda_m = lambda_m
        self.lambda_e = lambda_e
        # Set pi flag to true if using pi losses
        # @codereview: one line, can we get rid of this flag
        if (self.lambda_m != 0) or (self.lambda_e != 0):
            pi_flag = True
        else:
            pi_flag = False

        # Prepare the metrics
        train_loss = tf.keras.metrics.Mean("train_loss", dtype=tf.float32)
        train_metric = tf.keras.metrics.MeanSquaredError("train_metric")
        train_metric_true = tf.keras.metrics.MeanSquaredError("train_metric_true")

        val_interp_loss = tf.keras.metrics.Mean("val_interp_loss", dtype=tf.float32)
        val_interp_metric = tf.keras.metrics.MeanSquaredError("val_interp_metric")
        val_interp_metric_true = tf.keras.metrics.MeanSquaredError("val_interp_true")

        val_extrap_loss = tf.keras.metrics.Mean("val_extrap_loss", dtype=tf.float32)
        val_extrap_metric = tf.keras.metrics.MeanSquaredError("val_extrap_metric")
        val_extrap_metric_true = tf.keras.metrics.MeanSquaredError("val_extrap_true")

        # Create history dict
        # @codereview: can use list instead of numpy array
        history = {
            "train_loss": np.array([]),
            "train_metric": np.array([]),
            "train_metric_true": np.array([]),
            "val_interp_loss": np.array([]),
            "val_interp_metric": np.array([]),
            "val_interp_metric_true": np.array([]),
            "val_extrap_loss": np.array([]),
            "val_extrap_metric": np.array([]),
            "val_extrap_metric_true": np.array([]),
        }

        print("Starting training.")
        min_val_loss = np.infty
        cum_step = 0
        for epoch in range(epochs):
            # Iterate over the batches of the dataset.
            for input_batch, output_batch, output_batch_true in train_dataset:
                cum_step += 1
                lr = lr_schedule(cum_step)

                # @codereview: just one train function, overlap between train_step functions
                if pi_flag:
                    input_batch_size = len(input_batch)  # size of the input batch
                    sampled_input_batch = tf.constant(
                        sample(
                            x_domain=sampled_domain["x"],
                            t_domain=sampled_domain["t_train"],
                            batch_size=sampled_batch_size,
                        ),
                        dtype=tf.float32,
                    )
                    stacked_input_batch = tf.concat(
                        (input_batch, sampled_input_batch), axis=0
                    )
                    loss_value, output_batch_pred = self.train_step_PI(
                        stacked_input_batch, output_batch, input_batch_size, lr, observe
                    )
                else:
                    loss_value, output_batch_pred = self.train_step(
                        input_batch, output_batch, lr, observe
                    )

                train_loss.update_state(loss_value)
                train_metric.update_state(output_batch, output_batch_pred)
                train_metric_true.update_state(output_batch_true, output_batch_pred)

            # Record the loss at each epoch in the history dict
            history["train_loss"] = np.append(
                history["train_loss"], train_loss.result()
            )
            history["train_metric"] = np.append(
                history["train_metric"], train_metric.result()
            )
            history["train_metric_true"] = np.append(
                history["train_metric_true"], train_metric_true.result()
            )
            if val_interp_dataset is not None:
                # Run a validation loop at the end of each epoch to assess interpolation performance.
                for (
                    input_val_interp_batch,
                    output_val_interp_batch,
                    output_val_interp_batch_true,
                ) in val_interp_dataset:
                    if pi_flag:
                        input_val_interp_batch_size = len(
                            input_val_interp_batch
                        )  # size of the input val batch
                        sampled_input_val_interp_batch = tf.constant(
                            sample(
                                x_domain=sampled_domain["x"],
                                t_domain=sampled_domain["t_train"],
                                batch_size=sampled_batch_size,
                            ),
                            dtype=tf.float32,
                        )
                        stacked_input_val_interp_batch = tf.concat(
                            (input_val_interp_batch, sampled_input_val_interp_batch),
                            axis=0,
                        )
                        (
                            val_interp_loss_value,
                            output_val_interp_batch_pred,
                        ) = self.evaluate_PI(
                            stacked_input_val_interp_batch,
                            output_val_interp_batch,
                            input_val_interp_batch_size,
                            training=False,
                            observe=observe,
                        )
                    else:
                        (
                            val_interp_loss_value,
                            output_val_interp_batch_pred,
                        ) = self.evaluate(
                            input_val_interp_batch,
                            output_val_interp_batch,
                            training=False,
                            observe=observe,
                        )
                    # Update val metrics
                    val_interp_loss.update_state(val_interp_loss_value)
                    val_interp_metric.update_state(
                        output_val_interp_batch, output_val_interp_batch_pred
                    )
                    val_interp_metric_true.update_state(
                        output_val_interp_batch_true, output_val_interp_batch_pred
                    )

            # Record the loss at each epoch in the history dict
            history["val_interp_loss"] = np.append(
                history["val_interp_loss"], val_interp_loss.result()
            )
            history["val_interp_metric"] = np.append(
                history["val_interp_metric"], val_interp_metric.result()
            )
            history["val_interp_metric_true"] = np.append(
                history["val_interp_metric_true"], val_interp_metric_true.result()
            )

            # Run a validation loop at the end of each epoch to assess extrapolation performance.
            for (
                input_val_extrap_batch,
                output_val_extrap_batch,
                output_val_extrap_batch_true,
            ) in val_extrap_dataset:
                if pi_flag:
                    input_val_extrap_batch_size = len(
                        input_val_extrap_batch
                    )  # size of the input val batch
                    sampled_input_val_extrap_batch = tf.constant(
                        sample(
                            x_domain=sampled_domain["x"],
                            t_domain=sampled_domain["t_val"],
                            batch_size=sampled_batch_size,
                        ),
                        dtype=tf.float32,
                    )
                    stacked_input_val_extrap_batch = tf.concat(
                        (input_val_extrap_batch, sampled_input_val_extrap_batch), axis=0
                    )
                    (
                        val_extrap_loss_value,
                        output_val_extrap_batch_pred,
                    ) = self.evaluate_PI(
                        stacked_input_val_extrap_batch,
                        output_val_extrap_batch,
                        input_val_extrap_batch_size,
                        training=False,
                        observe=observe,
                    )
                else:
                    (
                        val_extrap_loss_value,
                        output_val_extrap_batch_pred,
                    ) = self.evaluate(
                        input_val_extrap_batch,
                        output_val_extrap_batch,
                        training=False,
                        observe=observe,
                    )
                # Update val metrics
                val_extrap_loss.update_state(val_extrap_loss_value)
                val_extrap_metric.update_state(
                    output_val_extrap_batch, output_val_extrap_batch_pred
                )
                val_extrap_metric_true.update_state(
                    output_val_extrap_batch_true, output_val_extrap_batch_pred
                )

            # Record the loss at each epoch in the history dict
            history["val_extrap_loss"] = np.append(
                history["val_extrap_loss"], val_extrap_loss.result()
            )
            history["val_extrap_metric"] = np.append(
                history["val_extrap_metric"], val_extrap_metric.result()
            )
            history["val_extrap_metric_true"] = np.append(
                history["val_extrap_metric_true"], val_extrap_metric_true.result()
            )

            # Save model with min val loss
            min_val_loss_prev = min_val_loss
            # if a validation of interpolation dataset is given,
            # then save the best model according to that loss
            # otherwise use the validation of extrapolation loss
            if val_interp_dataset is not None:
                min_val_loss = min(min_val_loss, val_interp_loss.result())
            else:
                min_val_loss = min(min_val_loss, val_extrap_loss.result())

            if log_dir is not None:
                if min_val_loss < min_val_loss_prev:
                    self.model.save_weights(log_dir / "models" / "best_model")

            # Print status
            if epoch % print_epoch_mod == 0:
                print_status_bar(
                    epoch + 1,
                    epochs,
                    [train_loss, val_interp_loss, val_extrap_loss],
                    [train_metric, val_interp_metric, val_extrap_metric],
                )

            # Save the model at certain epochs
            if log_dir is not None:
                if epoch + 1 in save_epochs:
                    self.model.save_weights(
                        log_dir / "models" / f"model_epochs_{epoch+1}"
                    )

            # Reset metrics at the end of each epoch
            train_loss.reset_states()
            val_interp_loss.reset_states()
            val_extrap_loss.reset_states()
            train_metric.reset_states()
            train_metric_true.reset_states()
            val_interp_metric.reset_states()
            val_interp_metric_true.reset_states()
            val_extrap_metric.reset_states()
            val_extrap_metric_true.reset_states()
        return history

    def predict_P_U(self, input_output_dict, split_data_dict):
        # predict on the full train data (before splitting into train and validation for interpolation sets)
        output_train_pred = self.predict(
            input_output_dict["input_train_full"], training=False
        ).numpy()
        # predict on the validation for extrapolation set
        output_val_pred = self.predict(
            input_output_dict["input_val_extrap"], training=False
        ).numpy()
        # predict on the test set
        output_test_pred = self.predict(
            input_output_dict["input_test"], training=False
        ).numpy()
        # reshape to get the predictions on a grid
        pred_dict = {}
        pred_dict["P_train"] = output_train_pred[:, 0].reshape(
            split_data_dict["P_train"].shape[0], split_data_dict["P_train"].shape[1]
        )
        pred_dict["U_train"] = output_train_pred[:, 1].reshape(
            split_data_dict["U_train"].shape[0], split_data_dict["U_train"].shape[1]
        )
        pred_dict["P_val"] = output_val_pred[:, 0].reshape(
            split_data_dict["P_val"].shape[0], split_data_dict["P_val"].shape[1]
        )
        pred_dict["U_val"] = output_val_pred[:, 1].reshape(
            split_data_dict["U_val"].shape[0], split_data_dict["U_val"].shape[1]
        )
        pred_dict["P_test"] = output_test_pred[:, 0].reshape(
            split_data_dict["P_test"].shape[0], split_data_dict["P_test"].shape[1]
        )
        pred_dict["U_test"] = output_test_pred[:, 1].reshape(
            split_data_dict["U_test"].shape[0], split_data_dict["U_test"].shape[1]
        )
        return pred_dict
