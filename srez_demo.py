from srez_model import loss_DSSIS_tf11
from srez_train import _save_stats
# import moviepy.editor as mpe
import numpy as np
import numpy.random
import os.path
import scipy.misc
import tensorflow as tf
import time
import png


FLAGS = tf.app.flags.FLAGS

# def demo1(sess):
#     """Demo based on images dumped during training"""

#     # Get images that were dumped during training
#     filenames = tf.gfile.ListDirectory(FLAGS.train_dir)
#     filenames = sorted(filenames)
#     filenames = [os.path.join(FLAGS.train_dir, f) for f in filenames if f[-4:]=='.png']

#     assert len(filenames) >= 1

#     fps        = 30

#     # Create video file from PNGs
#     print("Producing video file...")
#     filename  = os.path.join(FLAGS.train_dir, 'demo1.mp4')
#     clip      = mpe.ImageSequenceClip(filenames, fps=fps)
#     clip.write_videofile(filename)
#     print("Done!")
    

def demo2(data, num_sample):
    d = data
    batch_size = FLAGS.batch_size
    num_batch = num_sample / batch_size

    # Cache features and labels (they are small)
    list_features = []
    list_labels = []
    for batch in range(int(num_batch)):
        feature, label = d.sess.run([d.features, d.labels])
        list_features.append(feature)
        list_labels.append(label)
    print('Prepared {0} feature batches'.format(num_batch))

    test_header = ['batch', 'L1_error', 'L2_error', 'SNR', 'SSIM', 'time']
    test_stats = []

    for index_batch in range(int(num_batch)):
        print('Batch {}'.format(index_batch))

        feature = list_features[index_batch]
        label = list_labels[index_batch]

        print(feature.mean(), feature.var(), feature.max(), feature.min())
        print(label.mean(), label.var(), label.max(), label.min())
    
        # Show progress with features
        feed_dict = {d.gene_minput: feature}     
        # ops = [d.gene_moutput, d.gene_mlayers, d.disc_mlayers, d.disc_moutput, d.disc_gradients]
        ops = [d.gene_moutput]
        # ops = [d.gene_loss, d.gene_ls_loss, d.gene_dc_loss, d.disc_real_loss, d.disc_fake_loss, d.list_gene_losses] 
        # dum = tf.random_normal((FLAGS.batch_size, 256, 256, 2))
        # dum = tf.complex(dum[:,:,:,0], dum[:,:,:,1])
        # dum = tf.abs(dum)
        # dum = tf.reshape(dum, [FLAGS.batch_size, 256, 256, 1])
        # ops=[dum]

        # Run
        forward_passing_time = time.time()
        # gene_output, gene_layers, disc_layers, disc_output, disc_gradients = d.sess.run(ops, feed_dict=feed_dict)
        gene_output, = d.sess.run(ops, feed_dict=feed_dict)
        # gene_loss, gene_ls_loss, gene_dc_loss, disc_real_loss, disc_fake_loss, list_gene_losses = d.sess.run(ops, feed_dict=feed_dict)   
        
        # Stats
        inference_time = time.time() - forward_passing_time
        gene_output = (gene_output - tf.reduce_min(gene_output)) / (tf.reduce_max(gene_output) - tf.reduce_min(gene_output))
        l1_error = tf.metrics.mean_absolute_error(label, gene_output)
        l2_error  = tf.metrics.root_mean_squared_error(label, gene_output)
        snr = tf.reduce_mean(label**2) / l2_error**2
        ssim = 1.0 - 2.0 * loss_DSSIS_tf11(label, gene_output)
        l1_error, l2_error, snr, ssim = d.sess.run([l1_error, l2_error, snr, ssim])
        print('Time: {}s'.format(inference_time))
        print('L1 error: {}'.format(l1_error))
        print('L2 error: {}'.format(l2_error))
        print('SNR: {}'.format(snr))
        print('SSIM: {}'.format(ssim))
        test_stats.append([index_batch, l1_error, l2_error, snr, ssim, inference_time])

        # Visual
        if FLAGS.summary_period > 0 and index_batch % FLAGS.summary_period == 0:
            print('Saving comparison figure')
            save_image_output(d, feature, label, gene_output, 
                index_batch, 'test{}'.format(index_batch), batch_size)

    print('Saving stats')
    _save_stats("{}_test_stats.csv".format(index_batch), test_stats, test_header)
    print('Demo complete.')


def save_image_output(data, feature, label, gene_output, 
        batch, suffix, max_samples=8):
    d = data

    size = [label.shape[1], label.shape[2]]

    # complex input zpad into r and channel
    complex_zpad = (feature - feature.min()) / (feature.max() - feature.min())
    complex_zpad = tf.image.resize_nearest_neighbor(feature, size)
    complex_zpad = tf.maximum(tf.minimum(complex_zpad, 1.0), 0.0)

    # zpad magnitude
    mag_zpad = tf.sqrt(complex_zpad[:,:,:,0]**2+complex_zpad[:,:,:,1]**2)/tf.sqrt(2.0)
    # mag_zpad = tf.maximum(tf.minimum(mag_zpad, 1.0), 0.0)
    mag_zpad = tf.reshape(mag_zpad, [FLAGS.batch_size, size[0], size[1]])
    # mag_zpad = tf.concat(axis=3, values=[mag_zpad, mag_zpad])
    
    # output magnitude 
    mag_output = tf.maximum(tf.minimum(gene_output, 1.0), 0.0)
    mag_output = tf.reshape(mag_output, [FLAGS.batch_size, size[0], size[1]])
    # mag_output = tf.concat(axis=3, values=[mag_output, mag_output])

    mag_gt = tf.maximum(tf.minimum(label, 1.0), 0.0)
    mag_gt = tf.reshape(mag_gt, [FLAGS.batch_size, size[0], size[1]])
    # mag_gt = tf.concat(axis=3, values=[label, label])

    # concate for visualize image
    image = tf.concat(axis=2, values=[mag_zpad, mag_output, mag_gt])
    image = image[0:max_samples,:,:]
    image = tf.concat(axis=0, values=[image[i,:,:] for i in range(int(max_samples))])
    image = d.sess.run(image)
    print('Save to image size {} type {}', image.shape, type(image))

    # 3rd channel for visualization
    # mag_3rd = np.maximum(image[:,:,0],image[:,:,1])
    # image = np.concatenate((image, mag_3rd[:,:,np.newaxis]),axis=2)

    # Save to image file
    print('Save to image,', image.shape)
    filename = 'batch%06d_%s.png' % (batch, suffix)
    filename = os.path.join(FLAGS.train_dir, filename)

    # scipy.misc.toimage(image, cmin=0., cmax=1.).save(filename)
    with open(filename, 'wb') as f:
        image *= 65535
        z = (image).astype(np.uint16)
        writer = png.Writer(width=z.shape[1], height=z.shape[0], bitdepth=16, greyscale=True)
        zlist = z.tolist()
        writer.write(f, zlist)

    print("    Saved %s" % (filename,))
