# Module that contains periodic activation functions
import tensorflow as tf


def xSin2x(x, a=5, dtype=tf.float32):
    """Define x+1/a*sin^2(a*x) activation function"""
    a = tf.constant(a, dtype=dtype)
    return x + 1 / a * (tf.sin(a * x) ** 2)


def xSinx(x, a=5, dtype=tf.float32):
    """Define x+1/a*sin(a*x) activation function"""
    a = tf.constant(a, dtype=dtype)
    return x + 1 / a * tf.sin(a * x)


def sinx(x, a=5, dtype=tf.float32):
    """Define 1/a*sin(a*x) activation function"""
    a = tf.constant(a, dtype=dtype)
    return 1 / a * tf.sin(a * x)


def periodic_normal(shape, dtype=tf.float32):
    """Weights initialization from a normal distribution"""
    stddev = tf.sqrt(1 / shape[0])
    return tf.random.normal(shape, stddev=stddev, dtype=dtype)


def periodic_uniform(shape, dtype=tf.float32):
    """Weights initialization from a uniform distribution"""
    var = tf.constant(1 / shape[0], dtype=dtype)
    val = tf.sqrt(3 * var)
    return tf.random.uniform(shape, minval=-val, maxval=val, dtype=dtype)
