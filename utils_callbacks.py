import tensorflow as tf

# Save Checkpoint Callback
class CustomCheckpointCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        epoch = epoch + 1
        
        # Save model weights to local as hdf5
        checkpoint_file_name = f'checkpoint_epoch_{str(epoch).zfill(4)}.hdf5'
        self.model.save_weights(f'checkpoints/{checkpoint_file_name}')

            
class CustomTensorboardLoggerCallback(tf.keras.callbacks.Callback):
    def __init__(self, file_writer):
        super().__init__()
        self.writer = file_writer

    def on_epoch_end(self, epoch, logs=None):
    
        lr = self.model.optimizer.learning_rate
        loss = logs['loss']
        
        with self.writer.as_default():
            tf.summary.scalar('loss', loss, step = epoch + 1)
            tf.summary.scalar('lr', lr, step = epoch + 1)