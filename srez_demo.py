from __future__ import absolute_import, division, print_function
from srez_model import loss_DSSIS_tf11
from srez_train import _save_stats
import moviepy.editor as mpe
import numpy as np
import numpy.random
import os.path
import scipy.misc
import tensorflow as tf
import time
import png
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import imageio


FLAGS = tf.app.flags.FLAGS


def normalise(a):
    return (a - tf.reduce_min(a)) / (tf.reduce_max(a) - tf.reduce_min(a))


def clip(a):
    return tf.maximum(tf.minimum(a, 1.0), 0.0)


def save_image_output(data, feature, label, gene_output, 
        batch, suffix, max_samples=8):
    d = data

    size = [label.shape[1], label.shape[2]]

    # complex input zpad into r and channel
    complex_zpad = feature
    # complex_zpad = tf.image.resize_nearest_neighbor(feature, size)
    # complex_zpad = clip(complex_zpad)

    # zpad magnitude
    mag_zpad = tf.sqrt(complex_zpad[:,:,:,0]**2+complex_zpad[:,:,:,1]**2)
    mag_zpad = normalise(mag_zpad)
    mag_zpad = clip(mag_zpad)
    mag_zpad = tf.reshape(mag_zpad, [FLAGS.batch_size, size[0], size[1]])
    # mag_zpad = tf.concat(axis=3, values=[mag_zpad, mag_zpad])
    
    # output magnitude 
    mag_output = normalise(gene_output)
    mag_output = clip(mag_output)
    mag_output = tf.reshape(mag_output, [FLAGS.batch_size, size[0], size[1]])
    # mag_output = tf.concat(axis=3, values=[mag_output, mag_output])

    mag_gt = clip(label)
    mag_gt = tf.reshape(mag_gt, [FLAGS.batch_size, size[0], size[1]])
    # mag_gt = tf.concat(axis=3, values=[label, label])

    # concate for visualize image
    image = tf.concat(axis=2, values=[mag_zpad, mag_output, mag_gt])
    image = image[0:max_samples,:,:]
    image = tf.concat(axis=0, values=[image[i,:,:] for i in range(int(max_samples))])
    image = d.sess.run(image)

    # 3rd channel for visualization
    # mag_3rd = np.maximum(image[:,:,0],image[:,:,1])
    # image = np.concatenate((image, mag_3rd[:,:,np.newaxis]),axis=2)

    # Save to image file
    print('Save to image size {} type {}'.format(image.shape, type(image)))
    filename = 'batch{:06d}_{}.png'.format(batch, suffix)
    filename = os.path.join(FLAGS.train_dir, filename)

    # scipy.misc.toimage(image, cmin=0., cmax=1.).save(filename)
    with open(filename, 'wb') as f:
        image *= 65535
        z = (image).astype(np.uint16)
        writer = png.Writer(width=z.shape[1], height=z.shape[0], bitdepth=16, greyscale=True)
        zlist = z.tolist()
        writer.write(f, zlist)

    print("    Saved {}".format(filename))


def plot_image_output(data, feature, label, output, error=None, batch=None, max_samples=2):
    # Preprocess data for visualisation
    zf_complex = feature  # Input is complex zero-fill image
    zf = tf.sqrt(zf_complex[...,0]**2 + zf_complex[...,1]**2)
    zf = tf.squeeze(tf.clip_by_value(zf, 0.0, 1.0)) 
    gt = tf.squeeze(tf.clip_by_value(label, 0.0, 1.0))
    go = tf.squeeze(tf.clip_by_value(output, 0.0, 1.0))
    er = tf.squeeze(tf.clip_by_value(error, 0.0, 1.0))
    # Concatenate and flatten into 2D image
    images = data.sess.run([zf, go, gt, er])
    [zf, go, gt, er] = images
    # Plot
    figure, ax_array = plt.subplots(2, 4, sharex='col', sharey='row', figsize=(15, 6.9))
    ax_array[0, 0].set_ylabel('Sample 1')
    ax_array[1, 0].set_ylabel('Sample 2')
    ax_array[1, 0].set_xlabel('Input (zero-fill)')
    ax_array[1, 1].set_xlabel('Output')
    ax_array[1, 2].set_xlabel('Ground truth')
    ax_array[1, 3].set_xlabel('Error')
    for i in range(max_samples):
        axes = ax_array[i]
        axes[0].imshow(zf[i], vmin=0.0, vmax=1.0, cmap='gray')
        axes[1].imshow(go[i], vmin=0.0, vmax=1.0, cmap='gray')
        axes[2].imshow(gt[i], vmin=0.0, vmax=1.0, cmap='gray')
        im3 = axes[3].imshow(er[i], vmin=er[i].min(), vmax=er[i].max(), cmap='jet')
        divider = make_axes_locatable(axes[3])
        cax3 = divider.append_axes("right", size="5%", pad=0.05)
        figure.colorbar(im3, ax=axes[3], cax=cax3)
    # Save
    plt.savefig('{}/batch.png'.format(FLAGS.train_dir))
    names = ['zf', 'go', 'gt', 'er']
    for name, image in zip(names, images):
        for i in range(max_samples):
            c = 'jet' if 'er' in name else 'gray'
            plt.imsave('{}/{}{}.png'.format(FLAGS.train_dir, name, i), image[i], cmap=c)
    plt.show()


def demo1(sess):
    """Demo based on images dumped during training"""

    # Get images that were dumped during training
    filenames = tf.gfile.ListDirectory(FLAGS.train_dir)
    filenames = sorted(filenames)
    filenames = [os.path.join(FLAGS.train_dir, f) for f in filenames if f[-4:]=='.png']

    assert len(filenames) >= 1

    fps        = 30

    # Create video file from PNGs
    print("Producing video file...")
    filename  = os.path.join(FLAGS.train_dir, 'demo1.mp4')
    clip      = mpe.ImageSequenceClip(filenames, fps=fps)
    clip.write_videofile(filename)
    print("Done!")
    

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
    print('Prepared {} feature batches'.format(num_batch))

    test_header = ['Batch', 'MAE', 'RMSE', 'SNR', 'PSNR', 'SSIM', 'Time']
    test_stats = []

    for index_batch in range(int(num_batch)):
        print('Batch {}'.format(index_batch))

        feature = list_features[index_batch]
        label = list_labels[index_batch]

        print("Feature: mean {}, std {}, min {}, max {}".format(
            feature.mean(), feature.std(), feature.max(), feature.min()))
        print("Label:   mean {}, std {}, min {}, max {}".format(
            label.mean(), label.std(), label.max(), label.min()))
    
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
        # gene_output, gene_layers, disc_layers, disc_output, disc_gradients = d.sess.run(ops, feed_dict=feed_dict)
        forward_passing_time = time.time()
        gene_output, = d.sess.run(ops, feed_dict=feed_dict)
        inference_time = time.time() - forward_passing_time
        # gene_loss, gene_ls_loss, gene_dc_loss, disc_real_loss, disc_fake_loss, list_gene_losses = d.sess.run(ops, feed_dict=feed_dict)   
        
        # Stats
        label = tf.convert_to_tensor(label)
        gene_output = tf.convert_to_tensor(gene_output)
        # gene_output = normalise(gene_output)
        # gene_output = clip(gene_output)
        slice_time = inference_time / batch_size
        error = gene_output - label
        l1_error = tf.reduce_mean(tf.abs(error), axis=(1, 2, 3))
        mse = tf.reduce_mean(tf.square(error), axis=(1, 2, 3))
        l2_error = tf.sqrt(mse)
        snr = 10.0 * tf.log(tf.reduce_mean(tf.square(label), axis=(1, 2, 3)) / mse) / tf.log(10.0)
        psnr = 10.0 * tf.log(1.0 / mse) / tf.log(10.0)
        # ssim = 1.0 - 2.0 * loss_DSSIS_tf11(label, gene_output)  # convert loss to actual metric
        ssim = tf.image.ssim(label, gene_output, max_val=1.0)
        l1_error, l2_error, snr, psnr, ssim = d.sess.run([l1_error, l2_error, snr, psnr, ssim])
        print('Slice time: {}s'.format(slice_time))
        print('L1 error: {}'.format(l1_error))
        print('L2 error: {}'.format(l2_error))
        print('SNR: {}'.format(snr))
        print('PSNR: {}'.format(psnr))
        print('SSIM: {}'.format(ssim))
        for b in range(FLAGS.batch_size):
            test_stats.append([index_batch, l1_error[b], l2_error[b], snr[b], 
                            psnr[b], ssim[b], slice_time])

        # Visual
        if FLAGS.summary_period > 0 and index_batch % FLAGS.summary_period == 0:
            print('Saving comparison figure')
            idx = FLAGS.batch_size*index_batch
            print(d.test_filenames_input[idx:idx+FLAGS.batch_size])
            save_image_output(d, feature, label, gene_output, 
                index_batch, 'test{:04d}'.format(index_batch), batch_size)

    print('Saving stats')
    _save_stats("{}_test_stats.csv".format(index_batch), test_stats, test_header)
    print('Demo complete.')


def demo3(data, num_sample):
    # Set convenience variables
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
    print('Prepared {} feature batches'.format(num_batch))
    # Show mask
    plt.imshow(d.mask, cmap='gray')
    plt.title('Sampling mask\n(acc=4.08, K=0.1, r=0.23, CTR=256/12)')
    plt.show()
    # Iterate test samples
    for index_batch in range(int(num_batch)):
        print('Batch {}'.format(index_batch))
        # Feature and label stats
        feature = list_features[index_batch]
        label = list_labels[index_batch]
        print("Feature: mean {}, std {}, min {}, max {}".format(
            feature.mean(), feature.std(), feature.min(), feature.max()))
        print("Label:   mean {}, std {}, min {}, max {}".format(
            label.mean(), label.std(), label.min(), label.max()))
        # Run the model
        feed_dict = {d.gene_minput: feature}     
        ops = [d.gene_moutput]
        forward_passing_time = time.time()
        gene_output, = d.sess.run(ops, feed_dict=feed_dict)
        inference_time = time.time() - forward_passing_time
        # Performance stats
        label = tf.convert_to_tensor(label)
        gene_output = tf.convert_to_tensor(gene_output)
        slice_time = inference_time / batch_size
        error = gene_output - label
        mse = tf.reduce_mean(tf.square(error), axis=(1, 2, 3))
        psnr = 10.0 * tf.log(1.0 / mse) / tf.log(10.0)
        ssim = tf.image.ssim(label, gene_output, max_val=1.0)
        psnr, ssim = d.sess.run([psnr, ssim])
        print('Slice time: {}s'.format(slice_time))
        print('\nPSNR: {}'.format(psnr))
        print('SSIM: {}\n'.format(ssim))
        # Visual
        if FLAGS.summary_period > 0 and \
                index_batch % FLAGS.summary_period == 0:
            print('Creating comparison figure')
            idx = FLAGS.batch_size * index_batch
            print(d.test_filenames_input[idx : idx + FLAGS.batch_size])
            plot_image_output(d, feature, label, gene_output, error, 
                    index_batch, int(batch_size))
