import numpy as np
import tensorflow as tf

from ..utils import acoustic_modes as ac
from .nn import BaseNN


class GalerkinRijkeNN(BaseNN):
    def __init__(self, model, N_g, rijke=None):
        """Galerkin feed-forward neural network structure
        model: tf model
        N_g: number of Galerkin modes
        rijke: rijke model whose properties are used in the physics-informed training
               (defaults to None) (instance of a rijke class)
        """
        super().__init__(model)

        self.rjk = rijke
        if self.rjk is None:
            self.N_g = N_g
            j = np.arange(1, self.N_g + 1)
            self.jpi = tf.constant(j * np.pi, dtype=tf.float32)
        else:
            self.N_g = self.rjk.N_g
            print(
                "Warning: Since Rijke system is provided, N_g argument will be overriden by Rijke.N_g."
            )
            self.jpi = tf.constant(self.rjk.jpi, dtype=tf.float32)
            self.sinjpixf = tf.constant(self.rjk.sinjpixf, dtype=tf.float32)
            self.cosjpixf = tf.constant(self.rjk.cosjpixf, dtype=tf.float32)
            self.zeta = tf.constant(
                self.rjk.zeta, dtype=tf.float32
            )  # declare attributes

    @tf.function
    def predict(self, input, training):  # @codereview: input built-in
        """Predict Galerkin variables from time and space.
        Compute pressure and velocity.
        everything in the doc string
        """
        # Split inputs
        x = input[:, 0]  # enum
        t = input[:, 1]
        t = t[:, None]  # nicer new way, reshape, einops

        # @codereview: instead of big arrays, pass in individual arrays

        # predict galerkin coefficients
        y_pred = self.model(t, training=training)
        mu_pred = y_pred[:, 0 : self.N_g]
        eta_pred = y_pred[:, self.N_g : 2 * self.N_g]

        # find pressure and velocity
        # predict galerkin coefficients function
        jpix = tf.tensordot(x, self.jpi, axes=0)
        p_modes = -tf.sin(jpix)
        u_modes = tf.cos(jpix)
        p_pred = tf.reduce_sum(mu_pred * p_modes, axis=1)
        u_pred = tf.reduce_sum(eta_pred * u_modes, axis=1)
        p_pred = p_pred[:, None]
        u_pred = u_pred[:, None]
        output_pred = tf.concat([p_pred, u_pred], axis=1)
        return output_pred

    @tf.function
    def evaluate_PI(self, input, output, input_batch_size, training, observe="both"):
        """Evaluate the physics-informed loss.
        L_PI = lambda_DD * L_DataDriven + lambda_M * L_Momentum + lambda_E * L_Energy

        Momentum equations: \dot{\eta}_j = j\pi\mu_j
        Energy equations: \dot{\mu}_j = -j\pi\eta_j -\zeta_j\mu_j - 2\dot{q}\sin(j\pi x_f)
        """
        # Split inputs
        x = input[:, 0]
        t = input[:, 1]
        t = t[:, None]

        # use gradient tape for automatic differentiation of the network
        with tf.GradientTape() as pi_tape:
            pi_tape.watch(t)
            y_pred = self.model(t, training=training)
        y_dot_pred = pi_tape.batch_jacobian(y_pred, t)

        # @codereview: abstracting this into a function
        mu_pred = y_pred[:, 0 : self.N_g]
        eta_pred = y_pred[:, self.N_g : 2 * self.N_g]
        mu_dot_pred = y_dot_pred[
            :, 0 : self.N_g, 0
        ]  # @codereview: squeeze over dimension
        eta_dot_pred = y_dot_pred[:, self.N_g : 2 * self.N_g, 0]

        # Momentum equation
        # EVALUATE RESIDUAL FROM MOMENTUM EQUATION
        momentum_eqn = eta_dot_pred - self.jpi * mu_pred
        pi_momentum_loss = tf.reduce_mean(tf.square(momentum_eqn))

        # Energy equation
        # predict time-delayed input for the batch points
        y_tau_pred = self.model(t - self.rjk.tau, training=False)
        eta_tau_pred = y_tau_pred[:, self.N_g : 2 * self.N_g]
        # compute time-delayed flame velocity
        u_f_tau = tf.tensordot(eta_tau_pred, self.cosjpixf, axes=1)

        # compute heat release rate, @codereview: else statement raise some error, enums
        if self.rjk.heat_law == "kings":
            q_dot = self.rjk.beta * (tf.sqrt(tf.abs(1 + u_f_tau)) - 1)
        elif self.rjk.heat_law == "sigmoid":
            q_dot = self.rjk.beta / (1 + tf.exp(-u_f_tau))

        heat_release = 2 * tf.tensordot(q_dot, self.sinjpixf, axes=0)

        # EVALUATE RESIDUAL FROM MOMENTUM EQUATION
        energy_eqn = (
            mu_dot_pred + self.jpi * eta_pred + self.zeta * mu_pred + heat_release
        )
        pi_energy_loss = tf.reduce_mean(tf.square(energy_eqn))

        # find pressure and velocity
        p_modes = -tf.sin(tf.tensordot(x, self.jpi, axes=0))
        u_modes = tf.cos(tf.tensordot(x, self.jpi, axes=0))
        p_pred = tf.reduce_sum(mu_pred * p_modes, axis=1)
        u_pred = tf.reduce_sum(eta_pred * u_modes, axis=1)
        p_pred = p_pred[:, None]
        u_pred = u_pred[:, None]
        output_pred = tf.concat([p_pred, u_pred], axis=1)

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
        )  # @codereview: break it down into functions, dd_loss = ..., pi_momentum_loss = ...
        # print(dd_loss)
        # print(pi_momentum_loss)
        # print(pi_energy_loss)
        return loss_value, output_pred_dd


class GalerkinKinematicNN(BaseNN):
    def __init__(
        self,
        model,
        N_g,
        x_f,
        rho_up=1,
        rho_down=1,
        u_up=None,
        u_down=None,
        p_up=None,
        p_down=None,
        gamma=None,
        use_mean_flow=True,
        use_jump=True,
        use_linear=False,
    ):
        """Galerkin feed-forward neural network structure with piecewise modes for kinematic flame data
        model: tf model
        N_g: number of Galerkin modes
        rho_up, rho_down: mean density in the up- and downstream region, defaults to 1
        u_up, u_down: mean velocity in the up- and downstream region
                    defaults to None, needed only for PI
        p_up, p_down: mean pressure in the up- and downstream region
                    defaults to None, needed only for PI
        gamma: heat capacity ratio
                    defaults to None, needed only for PI
        use_mean_flow: only affects the chosen Galerkin modes,
                    if True then the given mean density values are used to determine piece-wise Galerkin modes
                    if False then the modes are created by assuming no jump of the mean density over the flame
                    defaults to True
        use_jump: whether to use the constant discontinuous modes to model the velocity jump
                defaults to True
        """
        super().__init__(model)

        self.N_g = N_g
        self.x_f = x_f
        self.rho_down = rho_down
        self.rho_up = rho_up
        self.u_down = u_down
        self.u_up = u_up
        # self.c_down = c_down
        # self.c_up = c_up
        self.p_down = p_down
        self.p_up = p_up
        self.gamma = gamma
        # self.A1 = A1
        # self.A2 = A2

        # determine the acoustic frequencies from the dispersion relationship
        self.use_mean_flow = use_mean_flow
        if self.use_mean_flow:
            (
                self.omega,
                self.k_up,
                self.k_down,
                self.upsilon,
                _,
                _,
            ) = ac.solve_dispersion(N_g, rho_up, rho_down, x_f)
        else:
            # (
            #    self.omega,
            #    self.k_up,
            #    self.k_down,
            #    self.upsilon,
            #    _,
            #    _,
            # ) = ac.solve_dispersion(N_g, 1.0, 1.0, x_f)
            self.omega = np.pi * np.arange(1, N_g + 1, dtype=np.float32)
            self.k_up = self.omega
            self.k_down = self.omega
            self.upsilon = np.empty((N_g,), dtype=np.float32)
            self.upsilon[::2] = 1
            self.upsilon[1::2] = -1
        # set the use_jump flag
        self.use_jump = use_jump
        self.use_linear = use_linear

        self.N_eta = self.N_g
        if self.use_jump:
            self.N_eta = self.N_eta + 2

        self.N_mu = self.N_g
        if self.use_linear:
            self.N_mu = self.N_mu + 2

    @tf.function
    def predict(self, input, training):
        """Predict Galerkin variables from time and space.
        Compute pressure and velocity.
        """
        # Split inputs
        x = input[:, 0]
        t = input[:, 1]
        t = t[:, None]

        # predict galerkin coefficients
        y_pred = self.model(t, training=training)
        mu_pred = y_pred[:, 0 : self.N_mu]
        eta_pred = y_pred[:, self.N_mu : self.N_mu + self.N_eta]

        # find pressure and velocity
        up_idx = x <= self.x_f  # condition array for the upstream section
        up_idx = up_idx[:, None]
        up_idx_0 = up_idx
        up_idx = tf.repeat(up_idx, self.N_g, axis=1)  # repeat the condition N_g times
        down_idx = x > self.x_f  # condition array for the downstream section
        down_idx = down_idx[:, None]
        down_idx_0 = down_idx
        down_idx = tf.repeat(
            down_idx, self.N_g, axis=1
        )  # repeat the condition N_g times

        # pressure modes in the up- and downstream
        p_up_modes = -tf.sin(tf.tensordot(x, self.k_up, axes=0))
        p_down_modes = -self.upsilon * tf.sin(
            tf.tensordot((1 - x), self.k_down, axes=0)
        )

        # mask the modes so that only the modes that satisfy the x condition remain,
        # the rest is set to zero
        # note: in tf, can't assign values directly, e.g. p_modes[up_idx,:] = ... not allowed
        # that's why we use this workaround
        p_up_modes = tf.where(up_idx, p_up_modes, tf.zeros((input.shape[0], self.N_g)))
        p_down_modes = tf.where(
            down_idx, p_down_modes, tf.zeros((input.shape[0], self.N_g))
        )
        p_modes = p_up_modes + p_down_modes

        # velocity modes in the up- and downstream
        if self.use_mean_flow:
            u_up_modes = (1 / self.rho_up**0.5) * tf.cos(
                tf.tensordot(x, self.k_up, axes=0)
            )
            u_down_modes = (
                -(1 / self.rho_down**0.5)
                * self.upsilon
                * tf.cos(tf.tensordot((1 - x), self.k_down, axes=0))
            )
        else:
            u_up_modes = 1.0 * tf.cos(tf.tensordot(x, self.k_up, axes=0))
            u_down_modes = (
                -1.0 * self.upsilon * tf.cos(tf.tensordot((1 - x), self.k_down, axes=0))
            )

        # mask the modes so that only the modes that satisfy the x condition remain,
        # the rest is set to zero
        # note: in tf, can't assign values directly, e.g. u_modes[up_idx,:] = ... not allowed
        # that's why we use this workaround
        u_up_modes = tf.where(up_idx, u_up_modes, tf.zeros((input.shape[0], self.N_g)))
        u_down_modes = tf.where(
            down_idx, u_down_modes, tf.zeros((input.shape[0], self.N_g))
        )
        u_modes = u_up_modes + u_down_modes

        if self.use_jump:
            # 2 constant separate modes
            u_mode_0 = tf.ones((input.shape[0], 1))
            u_mode_0_1 = tf.where(up_idx_0, u_mode_0, tf.zeros((input.shape[0], 1)))
            u_mode_0_2 = tf.where(down_idx_0, u_mode_0, tf.zeros((input.shape[0], 1)))
            u_modes = tf.concat([u_mode_0_1, u_mode_0_2, u_modes], axis=1)

        if self.use_linear:
            p_mode_0_1 = tf.ones((input.shape[0], 1))
            p_mode_0_2 = x[:, None]
            p_modes = tf.concat([p_mode_0_1, p_mode_0_2, p_modes], axis=1)

        # compute pressure and velocity from the modes
        p_pred = tf.reduce_sum(mu_pred * p_modes, axis=1)
        u_pred = tf.reduce_sum(eta_pred * u_modes, axis=1)
        p_pred = p_pred[:, None]
        u_pred = u_pred[:, None]
        output_pred = tf.concat([p_pred, u_pred], axis=1)
        return output_pred

    # @ todo: write out the dimensions of the operations for future reference

    @tf.function
    def evaluate_PI(self, input, output, input_batch_size, training, observe="both"):
        """Evaluate the physics-informed loss.
        L_PI = lambda_DD * L_DataDriven + lambda_M * L_Momentum + lambda_E * L_Energy

        Momentum equation: du/dt+u_bar*du/dx+1/rho_bar*dp/dx = 0
        Energy equation: dp/dt+u_bar*dp/dx+gamma*p_bar*du/dx = 0
        Args:
            input: [x,t]; stacked as [input at data points, sampled input for physics loss]
            output: [pressure,velocity]; [output at data points]
            input_batch_size: size of the data-driven input
            training: whether we make predictions in training mode or not
        Returns:
            loss_value: total loss as a sum of weighted individual data-driven and physics losses
            output_pred_dd: [pressure,velocity]; [predicted output at data points]
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

        ## LOW-ORDER EQUATIONS
        ## momentum equation
        # momentum_eqn_up = dudt_up + dpdx_up
        # momentum_eqn_down = dudt_down + dpdx_down
        # momentum_eqn = momentum_eqn_up + momentum_eqn_down
        # pi_momentum_loss = tf.reduce_mean(tf.square(momentum_eqn))
        #
        ## energy equation
        # energy_eqn_up = dpdt_up + dudx_up
        # energy_eqn_down = dpdt_down + dudx_down
        #
        # energy_eqn = energy_eqn_up + energy_eqn_down
        # pi_energy_loss = tf.reduce_mean(tf.square(energy_eqn))

        # COMPUTE JUMP CONDITION LOSS
        # momentum jump
        # input_size = input.shape[0]
        # h = 1e-5
        # x_f_left = (self.x_f-h)* tf.ones((input_size, 1))
        # input_f_left = tf.concat([x_f_left, t[:,None]], axis=1)
        # output_pred_f_left = self.predict(input_f_left, training = False)
        #
        # u_f_left = output_pred_f_left[:,1]
        # p_f_left = output_pred_f_left[:,0]
        # rho_f_left = 1/(self.c_up**2)*p_f_left

        # x_f_right = (self.x_f+h)* tf.ones((input_size, 1))
        # input_f_right = tf.concat([x_f_right, t[:,None]], axis=1)
        # output_pred_f_right = self.predict(input_f_right, training = False)

        # u_f_right = output_pred_f_right[:,1]
        # p_f_right = output_pred_f_right[:,0]

        # momentum_jump_eqn = self.A2*p_f_right-self.A1*p_f_left \
        #                    +self.A1*(self.u_down-self.u_up)*(self.u_up*rho_f_left+u_f_left*self.rho_up) \
        #                    +self.A1*(u_f_right-u_f_left)*self.rho_up*self.u_up \
        #                    -(self.A2-self.A1)*p_f_left

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
