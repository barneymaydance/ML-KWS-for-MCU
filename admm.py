import tensorflow as tf
import numpy as np

class Model(object):
    def __init__(self, dense_w, reg_lambda):
        self.alpha = {}
        self.Q = {}  # alpha * Q = Z
        self.Z = {}
        self.U = {}

        self.rho_input = tf.placeholder(tf.float32, name='rho_input')
        self.alpha_input={}
        self.Q_input={}
        self.U_input={}

        self.rho = reg_lambda

        for name, weight in dense_w.items():
            self.alpha_input[name] = tf.placeholder(tf.float32, name="aplpha_placeholder_" + name)
            self.Q_input[name] = tf.placeholder(tf.float32, shape=weight.shape, name="Q_placeholder_" + name)
            self.U_input[name] = tf.placeholder(tf.float32, shape=weight.shape, name="U_placeholder_" + name)

def project_to_centroid(FLAGS, model, W, name):

    U=model.U[name]
    Q=model.Q[name]

    alpha = np.mean(np.abs(W))  # initialize alpha
    num_iter_quant = 5  # caffe code use 20
    if FLAGS.quant_type == "binary":
        Q = np.where((W + U) > 0, 1, -1)
        alpha = np.sum(np.multiply((W + U), Q))
        QtQ = np.sum(Q ** 2)
        alpha /= QtQ

    elif FLAGS.quant_type == "ternary":
        for n in range(num_iter_quant):
            for n in range(num_iter_quant):
                Q = np.where((W + U) / alpha > 0.5, 1, Q)
                Q = np.where((W + U) / alpha < -0.5, -1, Q)
                Q = np.where(((W + U) / alpha >= -0.5) & ((W + U) / alpha <= 0.5), 0, Q)
                alpha = np.sum(np.multiply((W + U), Q))
                QtQ = np.sum(Q ** 2)
                alpha /= QtQ

    elif FLAGS.quant_type =="fixed":
        half_num_bits = FLAGS.num_bits - 1
        centroids = []
        for value in range(-2 ** half_num_bits + 1, 2 ** half_num_bits):
            centroids.append(value)

        for n in range(num_iter_quant):
            Q = np.where(np.round((W + U) / alpha) <= centroids[0], centroids[0], Q)
            Q = np.where(np.round((W + U) / alpha) >= centroids[-1], centroids[-1], Q)
            Q = np.where((np.round((W + U) / alpha) < centroids[-1]) & (np.round((W + U) / alpha) > centroids[0]),
                         np.round((W + U) / alpha), Q)

            # for i, value in enumerate(centroids):
            #
            #     if i == 0:
            #         Q = np.where(((W + U) / alpha) < (value + 0.5), value, Q)
            #     elif i == len(centroids) - 1:
            #         Q = np.where(((W + U) / alpha) >= (value - 0.5), value, Q)
            #     else:
            #         Q = np.where((((W + U) / alpha) >= (value - 0.5)) & (((W + U) / alpha) < (value + 0.5)), value,
            #                      Q)

            alpha = np.sum(np.multiply((W + U), Q))
            QtQ = np.multiply(Q, Q)
            QtQ = np.sum(QtQ)
            alpha /= QtQ
    else:
        raise ValueError("Quantized type is not supported!")

    model.Q[name] = Q
    model.Z[name] = alpha* model.Q[name]
    model.U[name] = U
    model.alpha[name] = alpha
    return model.Z[name], model.alpha[name], model.Q[name]

def admm_initialization(FLAGS, sess, model, dense_W):

    for name, weight in dense_W.items():
        w= sess.run(weight)
        model.alpha[name] = 0
        model.U[name] = np.zeros_like(w)
        model.Q[name] = np.zeros_like(w)
        model.Z[name] = np.zeros_like(w)
        updated_Z, updated_aplha, updated_Q = project_to_centroid(FLAGS, model, w, name)
        model.Z[name] = updated_Z
        model.alpha[name] = updated_aplha
        model.Q[name] = updated_Q

    return model

def z_u_update(FLAGS, sess, step, model, dense_w):
    if step != 1 and (step - 1) % FLAGS.admm_step_interval == 0:
        print("Step {}, updating Z, U!!!!!!".format(step))
        for name, weight in dense_w.items():
            w = sess.run(weight)
            Z_prev = np.array(model.Z[name])

            updated_Z, updated_aplha, updated_Q = project_to_centroid(FLAGS, model, w, name)  # equivalent to Euclidean Projection

            mu=1.5
            model.rho = model.rho * mu

            print("at layer {}. W(k+1)-Z(k+1): {}".format(name, np.sqrt(
                np.sum((w - updated_Z) ** 2))))
            print("at layer {}, Z(k+1)-Z(k): {}".format(name, np.sqrt(
                np.sum((updated_Z - Z_prev) ** 2))))

            model.U[name] = w - model.Z[name] + model.U[name]  # U(k+1) = W(k+1) - Z(k+1) +U(k)
    return model

def apply_quantization(FLAGS, sess, model, dense_w):
    for name, weight in dense_w.items():
        if FLAGS.quant_type == "binary":
            comparison=tf.math.greater_equal(weight, tf.constant(0.0))
            quant_value=tf.ones_like(weight) * model.alpha[name]
            w=tf.where(comparison, quant_value, -quant_value)
            assign_op=weight.assign(w)
            sess.run(assign_op)

        elif FLAGS.quant_type == "ternary":
            cp1 = tf.math.greater(weight, tf.constant(0.5*model.alpha[name],dtype=tf.float32))
            cp2 = tf.math.less(weight,tf.constant(-0.5*model.alpha[name],dtype=tf.float32))
            quant_value = tf.ones_like(weight) * model.alpha[name]
            w = tf.where(cp1, quant_value, tf.zeros_like(weight))
            w = tf.where(cp2,-quant_value, w)
            assign_op = weight.assign(w)
            sess.run(assign_op)

        elif FLAGS.quant_type=="fixed":
            pass