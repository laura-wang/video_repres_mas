import os

import tensorflow as tf

import model
import time
import input_data

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

flags = tf.app.flags
flags.DEFINE_integer('batch_size', 30, 'Training batch size, default:30')
flags.DEFINE_float('base_lr',0.001, 'base learning rate')
flags.DEFINE_string('model_save_path', 'motion_pattern_all_new_global', 'path to save model')
flags.DEFINE_integer('display', 1, 'print out the loss every dispaly iterations')
flags.DEFINE_integer('dimension',14, 'print out the loss every dispaly iterations')
flags.DEFINE_integer('max_iter',80000, 'max iteration')
flags.DEFINE_integer('cpu_num', 6, 'num of cpu process to read data, default:6')


FLAGS = flags.FLAGS
model_name = "model.ckpt"

momentum = 0.9


def train():
    img_input = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, 16, 112, 112, 3))
    y_target = tf.placeholder(tf.float32, shape=(FLAGS.batch_size,FLAGS.dimension))

    y = model.C3D(img_input, dimensions=FLAGS.dimension,dropout=False,regularizer=True) # not label!

    global_step = tf.Variable(0, trainable=False)

    varlist_weight = []
    varlist_bias = []
    trainable_variables = tf.trainable_variables()
    for var in trainable_variables:
        if 'weight' in var.name:
            varlist_weight.append(var)
        elif 'bias' in var.name:
            varlist_bias.append(var)

    lr_weight = tf.train.exponential_decay(FLAGS.base_lr, global_step, 20000, 0.1,
                                           staircase=True)
    lr_bias = tf.train.exponential_decay(FLAGS.base_lr * 2, global_step, 20000, 0.1,
                                         staircase=True)

    opt_weight = tf.train.MomentumOptimizer(lr_weight, momentum=momentum)
    opt_bias = tf.train.MomentumOptimizer(lr_bias, momentum=momentum)

    mse_loss = tf.reduce_mean(tf.squared_difference(y, y_target))

    weight_decay_loss = tf.add_n(tf.get_collection('weight_decay_loss'))

    loss = mse_loss + weight_decay_loss

    tf.summary.scalar('mse_loss', mse_loss)
    tf.summary.scalar('weight_decay_loss', weight_decay_loss)
    tf.summary.scalar('total_loss', loss)

    grad_weight = opt_weight.compute_gradients(loss, varlist_weight)
    grad_bias = opt_bias.compute_gradients(loss, varlist_bias)
    apply_gradient_op_weight = opt_weight.apply_gradients(grad_weight)
    apply_gradient_op_bias = opt_bias.apply_gradients(grad_bias, global_step=global_step)
    train_op = tf.group(apply_gradient_op_weight, apply_gradient_op_bias)

    saver = tf.train.Saver()
    merged = tf.summary.merge_all()

    rgb_list = 'list/rgb_train_linux.list'
    u_flow_list = 'list/u_flow_train_linux.list'
    v_flow_list = 'list/v_flow_train_linux.list'


    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter('./visual_logs/train', sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init)

        for i in range(FLAGS.max_iter):
            start_time = time.time()

            train_images, train_labels, next_batch_start = input_data.read_all(
                rgb_filename=rgb_list,
                u_flow_filename=u_flow_list,
                v_flow_filename=v_flow_list,
                batch_size=FLAGS.batch_size,
                start_pos=-1,
                shuffle=True,
                cpu_num=FLAGS.cpu_num
            )


            duration = time.time() - start_time
            print('read data time %.3f sec' % (duration))

            summary, loss_value, ce_loss, _, old_weight = sess.run([
                merged, loss, mse_loss, train_op, grad_weight], feed_dict={
                img_input: train_images,
                y_target: train_labels
            })

            if i % (FLAGS.display) == 0:
                print("mse_loss:", ce_loss)
                print("loss:", loss_value)
                train_writer.add_summary(summary, i)
            duration = time.time() - start_time
            print('Step %d: %.3f sec' % (i, duration))


            if i % 1000 == 0:
                saver.save(sess, os.path.join(FLAGS.model_save_path, model_name), global_step=global_step)


def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()
