#coding:utf-8
from resnet import * 
import tensorflow as tf

MOMENTUM = 0.9

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', 'C:\\Users\\salaslyrin\\Desktop\\ResNet\\_MYRESNET\\train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_float('learning_rate', 0.01, "learning rate.")
tf.app.flags.DEFINE_integer('batch_size', 64, "batch size")
tf.app.flags.DEFINE_integer('max_steps', 5000, "max steps")
tf.app.flags.DEFINE_boolean('resume', False,
                            'resume from latest saved state')
tf.app.flags.DEFINE_boolean('minimal_summaries', True,
                            'produce fewer summaries to save HD space')


def cos_loss(logits, labels):
    logits = tf.nn.l2_normalize(logits, 1)
    labels = tf.nn.l2_normalize(labels, 1)
    loss = tf.losses.cosine_distance(logits, labels, dim=0)
    print("logits:",logits)
    print("labels:",labels)
    print("loss:", loss)
    return loss


def train(is_training, logits, images, labels):
    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)
    val_step = tf.get_variable('val_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)

    loss_ = logits
    tf.summary.scalar('loss',loss_)

    #predictions = tf.nn.softmax(logits)

    #top1_error = top_k_error(predictions, labels, 1)


    # loss_avg
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, ema.apply([loss_]))
    tf.summary.scalar('loss_avg', ema.average(loss_))

    # validation stats
    #ema = tf.train.ExponentialMovingAverage(0.9, val_step)
    #val_op = tf.group(val_step.assign_add(1), ema.apply([top1_error]))
    #top1_error_avg = ema.average(top1_error)
    #tf.scalar_summary('val_top1_error_avg', top1_error_avg)

    tf.summary.scalar('learning_rate', FLAGS.learning_rate)

    opt = tf.train.MomentumOptimizer(FLAGS.learning_rate, MOMENTUM)
    grads = opt.compute_gradients(loss_)
    for grad, var in grads:
        if grad is not None and not FLAGS.minimal_summaries:
            tf.histogram_summary(var.op.name + '/gradients', grad)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    if not FLAGS.minimal_summaries:
        # Display the training images in the visualizer.
        tf.image_summary('images', images)

        for var in tf.trainable_variables():
            tf.histogram_summary(var.op.name, var)

    batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    train_op = tf.group(apply_gradient_op, batchnorm_updates_op)

    #opt = tf.train.MomentumOptimizer(FLAGS.learning_rate, 0.9)
    #train_op = opt.minimize(loss_)

    saver = tf.train.Saver(tf.global_variables())

    summary_op = tf.summary.merge_all()

    init = tf.global_variables_initializer()

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

    if FLAGS.resume:
        latest = tf.train.latest_checkpoint(FLAGS.train_dir)
        if not latest:
            print ("No checkpoint to continue from in", FLAGS.train_dir)
            sys.exit(1)
        print ("resume", latest)
        saver.restore(sess, latest)

    for x in range(FLAGS.max_steps + 1):
        start_time = time.time()

        step = sess.run(global_step)

        i = [train_op, loss_]
        #print("logits:",sess.run(logits[0]))
        #print("labels:",sess.run(labels[0]))
        #print("loss_:",sess.run(loss_))

        write_summary = step % 100 and step > 1
        if write_summary:
            i.append(summary_op)

        o = sess.run(i, { is_training: True })
        #print("ooooooooo:",o)
        loss_value = o[1]

        duration = time.time() - start_time

        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        if step % 100 == 0:
            examples_per_sec = FLAGS.batch_size / float(duration)
            format_str = ('step %d, loss = %.2f (%.1f examples/sec; %.3f '
                          'sec/batch)')
            print(format_str % (step, loss_value, examples_per_sec, duration))

        if write_summary:
            summary_str = o[2]
            summary_writer.add_summary(summary_str, step)

        # Save the model checkpoint periodically.
        if step > 1 and step % 100 == 0:
            checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=global_step)
    sess.close()