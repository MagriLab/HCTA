import tensorflow as tf

from .nn import BaseNN


class ForwardRijkeNN(BaseNN):
    def __init__(self, model, rijke=None, dx=None):
        """Feed-forward neural network structure
        model: tf model
        rijke: rijke model whose properties are used in the physics-informed training
               (defaults to None)
        dx: spatial grid spacing used in the physics-informed training
            (defaults to None)
        """
        super().__init__(model)
        self.rjk = rijke
        if self.rjk is not None:
            self.jpi = tf.constant(self.rjk.jpi, dtype=tf.float32)
            self.sinjpixf = tf.constant(self.rjk.sinjpixf, dtype=tf.float32)
            self.cosjpixf = tf.constant(self.rjk.cosjpixf, dtype=tf.float32)
            self.zeta = tf.constant(self.rjk.zeta, dtype=tf.float32)
        self.dx = dx

    @tf.function
    def predict(self, input, training):
        """Predict pressure and velocity from time and space."""
        output_pred = self.model(input, training=training)
        return output_pred

    @tf.function
    def evaluate_PI(self, input, output, input_batch_size, training, observe="both"):
        """Evaluate the physics-informed loss.
        L_PI = lambda_DD * L_DataDriven + lambda_M * L_Momentum + lambda_E * L_Energy

        Momentum equation: du/dt+dp/dx = 0
        Energy equation: dp/dt+du/dx+zeta*p-\dot{q}\delta(x-x_f) = 0

        Note: if data has been standardised, these equations may need to be adapted.
        """
        input_size = input.shape[0]
        # use gradient tape for automatic differentiation of the network
        with tf.GradientTape() as loss_tape:
            loss_tape.watch(input)
            output_pred = self.model(input, training=training)
        # get the jacobian of output (p,u) with respect to inputs (x,t)
        dydx = loss_tape.batch_jacobian(output_pred, input)
        dpdx = dydx[:, 0, 0]
        dpdt = dydx[:, 0, 1]
        dudx = dydx[:, 1, 0]
        dudt = dydx[:, 1, 1]

        # Momentum equation
        # EVALUATE RESIDUAL FROM MOMENTUM EQUATION
        momentum_eqn = dudt + dpdx
        pi_momentum_loss = tf.reduce_mean(tf.square(momentum_eqn))

        # Energy equation
        # prepare input at t = t-tau
        input_tau = input[:, 1] - self.rjk.tau
        input_tau = input_tau[:, None]
        # prepare input at x = x_f
        input_f_tau = tf.concat(
            [self.rjk.x_f * tf.ones((input_size, 1)), input_tau], axis=1
        )
        # predict u(x = x_f, t = t-tau) using the model
        u_f_tau = self.model(input_f_tau, training=False)[:, 1]

        # compute heat release rate
        if self.rjk.heat_law == "kings":
            q_dot = self.rjk.beta * (tf.sqrt(tf.abs(1 + u_f_tau)) - 1)
        elif self.rjk.heat_law == "sigmoid":
            q_dot = self.rjk.beta / (1 + tf.exp(-u_f_tau))

        # heat_release = \sum_{j = 1}^N_g 2*\dot{q}*sin(j\pi x_f)*sin(j\pi x)
        sinjpix = tf.sin(tf.tensordot(input[:, 0], self.jpi, axes=0))
        sum_jx = tf.tensordot(sinjpix, self.sinjpixf, axes=1)
        heat_release = 2 * q_dot * sum_jx

        if self.rjk.damping == "constant":
            # Damping term
            damping = self.rjk.c_1 * output_pred[:, 0]
        elif self.rjk.damping == "modal":
            # Damping
            x_grid = tf.linspace(0, 1, 3 * self.rjk.N_g + 1)
            x_grid = tf.cast(x_grid, dtype=tf.float32)
            t_grid = input[:, 1]

            # create input data to evaluate on the whole x grid at each t in the batch
            input_grid = tf.meshgrid(x_grid, t_grid)
            input_x_grid = tf.reshape(input_grid[0], [-1])
            input_t_grid = tf.reshape(input_grid[1], [-1])
            input_x_grid = input_x_grid[:, None]
            input_t_grid = input_t_grid[:, None]
            input_grid_flat = tf.concat([input_x_grid, input_t_grid], axis=1)

            # predict pressure
            P_grid_flat = self.model(input_grid_flat, training=False)[:, 0]
            P_grid_real = tf.reshape(P_grid_flat, [len(t_grid), len(x_grid)])
            # create the pressure over x = [0,2) so that when we take the FFT
            # we can have pi as the first Fourier frequency (resolution)
            P_grid_real = tf.concat(
                [P_grid_real, -tf.reverse(P_grid_real[:, 1:-1], axis=[1])], axis=1
            )
            # the indices in the reverse are from 1 because we don't want to include 2
            # to -1 because we don't want to repeat 1
            P_grid_im = tf.zeros(P_grid_real.shape)
            P_grid_complex = tf.complex(P_grid_real, P_grid_im)  # convert to complex

            # transform pressure to frequency domain via fft
            P_grid_fft = tf.signal.fft(P_grid_complex)

            # find the damping modes
            zeta_im = tf.zeros(self.zeta.shape, dtype=tf.float32)
            zeta = tf.complex(self.zeta, zeta_im)

            P_grid_fft_pos = P_grid_fft[
                :, 1 : self.rjk.N_g + 1
            ]  # take only the positive frequencies and only up to the N_g

            # apply damping
            # convolution in spatial domain is multiplication in the frequency domain
            conv1 = zeta * P_grid_fft_pos
            # find the reverse for the negative frequencies
            conv2 = tf.math.conj(tf.reverse(conv1, axis=[1]))
            conv = tf.concat(
                (
                    conv1,
                    conv2,
                ),
                axis=1,
            )

            # now we will do the inverse fft
            # get the frequencies (positive and negative)
            jpi_full = tf.concat([self.jpi, tf.reverse(-self.jpi, axis=[0])], axis=0)

            # multiply with the inputs
            jpi_full_x_im = tf.tensordot(
                input[:, 0], jpi_full, axes=0
            )  # this is the inside of the exponential
            jpi_full_x_real = tf.zeros(jpi_full_x_im.shape, dtype=tf.float32)
            jpi_full_x = tf.complex(jpi_full_x_real, jpi_full_x_im)

            # take the exponential
            exp_im_jpix = tf.exp(jpi_full_x)

            # multiply the exponential term with the damping values in frequency domain
            conv_exp = conv * exp_im_jpix

            # sum to get the real damping value
            damping = tf.reduce_sum(conv_exp, axis=1)

            # divide by the total number of frequencies
            damping = 1 / (6 * self.rjk.N_g) * damping
            # N_g times 2 because positive and negative freq,
            # times 3 because we initially divided [0,1) into 3*N_g
            # the fft was performed on a total of 6*N_g points

            damping = tf.math.real(damping)

        # EVALUATE RESIDUAL FROM ENERGY EQUATION
        energy_eqn = dpdt + dudx + damping - heat_release
        pi_energy_loss = tf.reduce_mean(tf.square(energy_eqn))

        # COMPUTE DATA-DRIVEN LOSS
        # input has been stacked with the data point and sampled points only used for physics loss,
        # so output_pred is stacked such that [output_pred_datadriven, output_pred_sampled]
        output_pred_dd = output_pred[:input_batch_size, :]
        # split the variables
        p_pred_dd = output_pred_dd[:, 0]
        u_pred_dd = output_pred_dd[:, 1]
        p = output[:, 0]
        u = output[:, 1]
        # we change the data-driven loss depending on which variable(s)
        # have been observed, even though we predict for both
        if observe == "both":  # if both pressure and velocity are observed
            dd_loss = tf.reduce_mean(self.loss_fn(output, output_pred_dd))
        elif observe == "p":  # if only pressure is observed
            dd_loss = tf.reduce_mean(self.loss_fn(p[:, None], p_pred_dd[:, None]))
        elif observe == "u":  # if only velocity is observed
            dd_loss = tf.reduce_mean(self.loss_fn(u[:, None], u_pred_dd[:, None]))

        # find the total loss by summing data-driven, physics-informed and regularization losses
        loss_value = tf.add_n(
            [self.lambda_dd * dd_loss]
            + [self.lambda_m * pi_momentum_loss]
            + [self.lambda_e * pi_energy_loss]
            + self.model.losses
        )
        return loss_value, output_pred_dd


class ForwardKinematicNN(BaseNN):
    def __init__(
        self,
        model,
        x_f=None,
        rho_up=None,
        rho_down=None,
        u_up=None,
        u_down=None,
        p_up=None,
        p_down=None,
        gamma=None,
    ):
        """Feed-forward neural network structure
        model: tf model
        """
        super().__init__(model)

        # set the mean flow properties
        self.x_f = x_f
        self.rho_down = rho_down
        self.rho_up = rho_up
        self.u_down = u_down
        self.u_up = u_up
        self.p_down = p_down
        self.p_up = p_up
        self.gamma = gamma

    @tf.function
    def predict(self, input, training):
        """Predict pressure and velocity from time and space."""
        output_pred = self.model(input, training=training)
        return output_pred

    @tf.function
    def evaluate_PI(self, input, output, input_batch_size, training, observe):
        """Evaluate the physics-informed loss.
        L_PI = lambda_DD * L_DataDriven + lambda_M * L_Momentum + lambda_E * L_Energy

        Momentum equation: du/dt+u_bar*du/dx+1/rho_bar*dp/dx = 0
        Energy equation: dp/dt+u_bar*dp/dx+gamma*p_bar*du/dx = 0
        """
        # Find up and down indices
        x = input[:, 0]

        # find pressure and velocity
        up_idx = x <= self.x_f  # condition array for the upstream section
        down_idx = x > self.x_f  # condition array for the downstream section

        # use gradient tape for automatic differentiation of the network
        with tf.GradientTape() as loss_tape:
            loss_tape.watch(input)
            output_pred = self.predict(input, training=training)
        # get the jacobian of output (p,u) with respect to inputs (x,t)
        dydx = loss_tape.batch_jacobian(output_pred, input)
        dpdx = dydx[:, 0, 0]
        dpdt = dydx[:, 0, 1]
        dudx = dydx[:, 1, 0]
        dudt = dydx[:, 1, 1]

        # slice for up- and downstream regions (mask)
        dpdx_up = tf.where(up_idx, dpdx, tf.zeros_like(dpdx))
        dpdx_down = tf.where(down_idx, dpdx, tf.zeros_like(dpdx))
        dpdt_up = tf.where(up_idx, dpdt, tf.zeros_like(dpdt))
        dpdt_down = tf.where(down_idx, dpdt, tf.zeros_like(dpdt))
        dudx_up = tf.where(up_idx, dudx, tf.zeros_like(dudx))
        dudx_down = tf.where(down_idx, dudx, tf.zeros_like(dudx))
        dudt_up = tf.where(up_idx, dudt, tf.zeros_like(dudt))
        dudt_down = tf.where(down_idx, dudt, tf.zeros_like(dudt))

        # momentum equation
        momentum_eqn_up = dudt_up + self.u_up * dudx_up + (1 / self.rho_up) * dpdx_up
        momentum_eqn_down = (
            dudt_down + self.u_down * dudx_down + (1 / self.rho_down) * dpdx_down
        )
        momentum_eqn = momentum_eqn_up + momentum_eqn_down
        pi_momentum_loss = tf.reduce_mean(tf.square(momentum_eqn))

        # energy equation
        energy_eqn_up = dpdt_up + self.u_up * dpdx_up + self.gamma * self.p_up * dudx_up
        energy_eqn_down = (
            dpdt_down + self.u_down * dpdx_down + self.gamma * self.p_up * dudx_down
        )
        energy_eqn = energy_eqn_up + energy_eqn_down
        pi_energy_loss = tf.reduce_mean(tf.square(energy_eqn))

        # COMPUTE DATA-DRIVEN LOSS
        # input has been stacked with the data point and sampled points only used for physics loss,
        # so output_pred is stacked such that [output_pred_datadriven, output_pred_sampled]
        output_pred_dd = output_pred[:input_batch_size, :]
        # split the variables
        p_pred_dd = output_pred_dd[:, 0]
        u_pred_dd = output_pred_dd[:, 1]
        p = output[:, 0]
        u = output[:, 1]
        # we change the data-driven loss depending on which variable(s)
        # have been observed, even though we predict for both
        if observe == "both":  # if both pressure and velocity are observed
            dd_loss = tf.reduce_mean(self.loss_fn(output, output_pred_dd))
        elif observe == "p":  # if only pressure is observed
            dd_loss = tf.reduce_mean(self.loss_fn(p[:, None], p_pred_dd[:, None]))
        elif observe == "u":  # if only velocity is observed
            dd_loss = tf.reduce_mean(self.loss_fn(u[:, None], u_pred_dd[:, None]))

        # find the total loss by summing data-driven, physics-informed and regularization losses
        loss_value = tf.add_n(
            [self.lambda_dd * dd_loss]
            + [self.lambda_m * pi_momentum_loss]
            + [self.lambda_e * pi_energy_loss]
            + self.model.losses
        )
        return loss_value, output_pred_dd
