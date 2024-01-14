import math
import tensorflow as tf

# get policy set by the caller script
policy = tf.keras.mixed_precision.global_policy()

# About normalize_before from fairseq source code
# https://github.com/facebookresearch/fairseq/blob/0272196aa803ecc94a8bffa6132a8c64fd7de286/fairseq/modules/transformer_layer.py#L19
# Below explanation is about where to apply LayerNorm in Encoder/Decoder layers.
"""
In the original paper each operation (multi-head attention or FFN) is
postprocessed with: `dropout -> add residual -> layernorm`. In the
tensor2tensor code they suggest that learning is more robust when
preprocessing each layer with layernorm and postprocessing with:
`dropout -> add residual`. We default to the approach in the paper, but the
tensor2tensor approach can be enabled by setting
*cfg.encoder.normalize_before* to ``True``.
"""
# - BART: https://github.com/huggingface/transformers/blob/main/src/transformers/models/bart/modeling_tf_bart.py
# - MBART: https://github.com/huggingface/transformers/blob/main/src/transformers/models/mbart/modeling_tf_mbart.py

# Related to topic above, BART and mBART implementations in HF are different.
# BART uses normalize_before = False while mBART uses normalize_before = True.
# Since I will follow mBART, I also implement according to normalize_before = True.

class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, d_kv, d_ff, num_heads, dropout, **kwargs):
        super(TransformerEncoderLayer, self).__init__(**kwargs)
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_kv,
                                                            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02))
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.layernorm_1 = tf.keras.layers.LayerNormalization(epsilon = 1e-5)
        
        self.dense_proj = tf.keras.Sequential(
            [tf.keras.layers.Dense(d_ff, activation="gelu",
                                   kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02)),
             tf.keras.layers.Dense(d_model, 
                                   kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02)), 
             ]
        )
        self.layernorm_2 = tf.keras.layers.LayerNormalization(epsilon = 1e-5)

    def call(self, inputs, encoder_padding_mask, training = False):
        encoder_padding_mask = encoder_padding_mask[:, tf.newaxis, tf.newaxis, :]
        
        attention_output = self.layernorm_1(inputs)
        attention_output = self.attention(query=attention_output, value=attention_output, key=attention_output, 
                                          attention_mask=encoder_padding_mask)
        attention_output = self.dropout(attention_output, training = training)
        proj_input = inputs + attention_output
        
        proj_output = self.layernorm_2(proj_input)
        proj_output = self.dense_proj(proj_output)
        proj_output = self.dropout(proj_output, training = training)
        return proj_input + proj_output
    
    
class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, d_kv, d_ff, num_heads, dropout, **kwargs):
        super(TransformerDecoderLayer, self).__init__(**kwargs)
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.self_attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_kv,
                                                              kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02))
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.layernorm_1 = tf.keras.layers.LayerNormalization(epsilon = 1e-5)
        
        self.cross_attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_kv,
                                                              kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02))
        self.layernorm_2 = tf.keras.layers.LayerNormalization(epsilon = 1e-5)
        
        self.dense_proj = tf.keras.Sequential(
            [tf.keras.layers.Dense(d_ff, activation="gelu",
                                   kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02)),
             tf.keras.layers.Dense(d_model, 
                                   kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02)), 
            ]
        )
        self.layernorm_3 = tf.keras.layers.LayerNormalization(epsilon = 1e-5)

    def call(self, inputs, encoder_outputs, cross_attention_mask, training = False):
        causal_mask = self.get_causal_attention_mask(inputs)

        self_attention_output = self.layernorm_1(inputs)
        self_attention_output = self.self_attention(query=self_attention_output, value=self_attention_output, key=self_attention_output, 
                                                    attention_mask=causal_mask)
        self_attention_output = self.dropout(self_attention_output, training = training)
        out_1 = inputs + self_attention_output

        cross_attention_output = self.layernorm_2(out_1)
        cross_attention_output = self.cross_attention(
            query=cross_attention_output,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=cross_attention_mask
        )
        cross_attention_output = self.dropout(cross_attention_output, training = training)
        out_2 = out_1 + cross_attention_output

        proj_output = self.layernorm_3(out_2)
        proj_output = self.dense_proj(proj_output)
        proj_output = self.dropout(proj_output, training = training)
        return out_2 + proj_output

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, text_max_len = input_shape[0], input_shape[1]
        i = tf.range(text_max_len)[:, tf.newaxis]
        j = tf.range(text_max_len)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)


class SharedEmbeddingLayer(tf.keras.layers.Layer):
    """
    Calculates input embeddings and pre-softmax linear with shared weights.
    Adapted from: https://github.com/tensorflow/models/blob/a009f4fb9d2fc4949e32192a944688925ef78659/official/transformer/v2/embedding_layer.py#L24
    """
    def __init__(self, vocab_size, embedding_size, initializer_range = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        # This is what PEGASUS does instead of scaling the embedding matrix later on
        self.initializer_range = embedding_size ** -0.5 if initializer_range is None else initializer_range
        
    def build(self, input_shape):
        self.shared_weights = self.add_weight("shared_weights",
                                              shape=[self.vocab_size, self.embedding_size],
                                              initializer=tf.random_normal_initializer(mean=0., stddev=self.embedding_size**-0.5))

    def call(self, inputs, mode = "embedding"):
        if mode == "embedding":
            return self._embedding(inputs)
        elif mode == "linear":
            return self._linear(inputs)
        else:
            raise ValueError(f"mode {mode} is not valid.")
            
    def _embedding(self, inputs):
        embeddings = tf.gather(self.shared_weights, inputs)
        return embeddings

    def _linear(self, inputs):
        batch_size = tf.shape(inputs)[0]
        length = tf.shape(inputs)[1]
        
        x = tf.reshape(inputs, [-1, self.embedding_size])
        logits = tf.matmul(x, self.shared_weights, transpose_b=True)

        return tf.reshape(logits, [batch_size, length, self.vocab_size])


class PositionalEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, shared_embedding_layer, text_max_len, vocab_size, d_model, dropout, **kwargs):
        super(PositionalEmbeddingLayer, self).__init__(**kwargs)
        
        self.token_embeddings = shared_embedding_layer
        self.position_embedding_matrix = self.get_position_encoding(text_max_len, d_model) # This is not weight. It's a constant tensor
        self.embedding_norm_layer = tf.keras.layers.LayerNormalization(epsilon = 1e-5)
        self.sum_emb_dropout = tf.keras.layers.Dropout(dropout)
        self.text_max_len = text_max_len
        self.vocab_size = vocab_size
        self.d_model = d_model

    def call(self, inputs, training = False):
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embedding_matrix
        sum_of_embeddings = embedded_tokens + embedded_positions
        sum_of_embeddings = self.embedding_norm_layer(sum_of_embeddings)
        sum_of_embeddings = self.sum_emb_dropout(sum_of_embeddings, training = training)
        return sum_of_embeddings

    def get_position_encoding(self, length, d_model, min_timescale=1.0, max_timescale=1.0e4):
        """
        Return positional encoding.
        Taken from original tf transformers implementation
        https://github.com/tensorflow/models/blob/a009f4fb9d2fc4949e32192a944688925ef78659/official/transformer/model/model_utils.py#L32
        """
        # We compute the positional encoding in float32 even if the model uses
        # float16, as many of the ops used, like log and exp, are numerically unstable
        # in float16.
        
        position = tf.cast(tf.range(length), tf.float32)
        num_timescales = d_model // 2
        log_timescale_increment = (
          math.log(float(max_timescale) / float(min_timescale)) /
          (tf.cast(num_timescales, tf.float32) - 1))
        inv_timescales = min_timescale * tf.exp(
          tf.cast(tf.range(num_timescales), tf.float32) * -log_timescale_increment)
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        
        # to be able to use mixedprecision16
        if policy.compute_dtype == 'float16':
            signal = tf.cast(signal, dtype = tf.float16)
        
        return signal

class CustomNonPaddingSCCELabelSmoothLoss(tf.keras.losses.Loss):
    def __init__(self, label_smoothing, vocab_size, name="custom_scce_loss"):
        super().__init__(name=name)
        self.label_smoothing = label_smoothing
        self.vocab_size = vocab_size
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, label_smoothing = self.label_smoothing, reduction=tf.keras.losses.Reduction.NONE
        )

    def call(self, y_true, y_pred):
        #y_pred = tf.cast(y_pred, tf.float32)
        mask = tf.cast((y_true > 0), dtype=tf.float32)
        y_true = tf.one_hot(y_true, self.vocab_size)
        loss = self.loss_fn(y_true, y_pred)
        loss = loss * mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000, last_epoch = 0, steps_per_epoch = 1000):
        """
        M.T note: last_epoch and steps_per_epoch args are added by me to be able to continue from a checkpoint.
        Otherwise learning rate re-starts even if I pass "last_epoch" arg to model.fit() function.
        """
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps
        self.last_epoch = last_epoch # 0 means model is training for the first time
        self.steps_per_epoch = steps_per_epoch
    
    def __call__(self, step):
        step = self.last_epoch * self.steps_per_epoch + step
        step = tf.cast(step, tf.float32) # Not casting step into float32 causes problems with Tensorboard
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
    
    def get_config(self):
        config = {'d_model': self.d_model,
                'warmup_steps': self.warmup_steps,
                'last_epoch': self.last_epoch,
                'steps_per_epoch': self.steps_per_epoch}
        
        return config


class TransformerEncoder(tf.keras.Model):
    def __init__(self, N, SOURCE_LEN, VOCAB_SIZE, D_MODEL, D_FF, D_KV, H, DROPOUT, shared_embedding_layer, 
                 name="TransformerEncoder"):
        super(TransformerEncoder, self).__init__(name=name)
        
        self.encoder_input_embedding_layer = PositionalEmbeddingLayer(shared_embedding_layer, SOURCE_LEN, VOCAB_SIZE, D_MODEL, DROPOUT)
        self.encoder_blocks = [TransformerEncoderLayer(D_MODEL, D_KV, D_FF, H, DROPOUT) for _ in range(N)]
        self.post_encoder_norm_layer = tf.keras.layers.LayerNormalization(epsilon = 1e-5) # Introduced by mBART for fp16 training

    def call(self, inputs, training = False):
        padding_mask = self.compute_padding_mask(inputs)
        out = self.encoder_input_embedding_layer(inputs, training)
        for i in range(len(self.encoder_blocks)):
            out = self.encoder_blocks[i](out, padding_mask, training)
        out = self.post_encoder_norm_layer(out)
        
        return out, padding_mask
        
    def compute_padding_mask(self, inputs):
        """
        According to MultiHeadAttention https://keras.io/api/layers/attention_layers/multi_head_attention/
        1 means attend and 0 means otherwise.
        """
        return tf.math.not_equal(inputs, 0)

class TransformerDecoder(tf.keras.Model):
    def __init__(self, N, TARGET_LEN, VOCAB_SIZE, D_MODEL, D_FF, D_KV, H, DROPOUT, shared_embedding_layer, 
                 name="TransformerDecoder"):
        super(TransformerDecoder, self).__init__(name=name)
        self.target_len = TARGET_LEN
        self.decoder_input_embedding_layer = PositionalEmbeddingLayer(shared_embedding_layer, TARGET_LEN, VOCAB_SIZE, D_MODEL, DROPOUT)
        self.decoder_blocks = [TransformerDecoderLayer(D_MODEL, D_KV, D_FF, H, DROPOUT) for _ in range(N)]
        self.post_decoder_norm_layer = tf.keras.layers.LayerNormalization(epsilon = 1e-5) # Introduced by mBART for fp16 training

    def call(self, decoder_inputs, encoder_outputs, padding_mask, training = False):
        cross_attention_mask = self.compute_cross_attention_mask(padding_mask, self.target_len)
        out = self.decoder_input_embedding_layer(decoder_inputs, training)
        for i in range(len(self.decoder_blocks)):
            out = self.decoder_blocks[i](out, encoder_outputs, cross_attention_mask, training)
        out = self.post_decoder_norm_layer(out)
        
        return out
        
    def compute_cross_attention_mask(self, encoder_padding_mask, target_seq_length):
        """
        Computes the attention mask for cross attention(encoder-decoder attention)
        Computed mask has shape: (B, T, S) as Keras MHA requires https://keras.io/api/layers/attention_layers/multi_head_attention/
        """
        return tf.tile(tf.expand_dims(encoder_padding_mask, axis = 1), [1, target_seq_length, 1])

class MODEL(tf.keras.Model):
    def __init__(self, N, H, D_MODEL, SOURCE_LEN, TARGET_LEN, VOCAB_SIZE, D_FF, D_KV, DROPOUT):
        """
        This is an mBART network. 
        Hence its differences from BART are:
            - It contains extra LayerNormalization layers on top of Encoder and Decoder.
            - LayerNorm in MHA layers are pre-Dense, aka normalize_before = True.
            
            # About normalize_before from fairseq source code
            # https://github.com/facebookresearch/fairseq/blob/0272196aa803ecc94a8bffa6132a8c64fd7de286/fairseq/modules/transformer_layer.py#L19
            # Below explanation is about where to apply LayerNorm in Encoder/Decoder layers.
            
            In the original paper each operation (multi-head attention or FFN) is
            postprocessed with: `dropout -> add residual -> layernorm`. In the
            tensor2tensor code they suggest that learning is more robust when
            preprocessing each layer with layernorm and postprocessing with:
            `dropout -> add residual`. We default to the approach in the paper, but the
            tensor2tensor approach can be enabled by setting
            *cfg.encoder.normalize_before* to ``True``.
            
            # - BART: https://github.com/huggingface/transformers/blob/main/src/transformers/models/bart/modeling_tf_bart.py
            # - MBART: https://github.com/huggingface/transformers/blob/main/src/transformers/models/mbart/modeling_tf_mbart.py
            # Related to topic above, BART and mBART implementations in HF are different.
            # BART uses normalize_before = False while mBART uses normalize_before = True.
            # Since I will follow mBART, I also implement according to normalize_before = True.
        
        N: Number of Encoder and Decoder layers. N is fixed for both.
        H: Number of Attention heads.
        D_MODEL: AKA d_dim.
        D_FF: This is D_MODEL * 4 by default.
        D_KV: This is D_MODEL // H by default.
        VOCAB_SIZE: Vocabulary size.
        SOURCE_LEN: Sequence length of Encoder.
        TARGET_LEN: Sequence length of Decoder.
        DROPOUT: Dropout.
        """
        super(MODEL, self).__init__()
        self.N = N
        self.H = H
        self.D_MODEL = D_MODEL
        self.D_FF = D_FF
        self.D_KV = D_KV
        self.VOCAB_SIZE = VOCAB_SIZE
        self.SOURCE_LEN = SOURCE_LEN
        self.TARGET_LEN = TARGET_LEN
        self.DROPOUT = DROPOUT
        
        self.shared_embedding_layer = SharedEmbeddingLayer(VOCAB_SIZE, D_MODEL)
        self.encoder = TransformerEncoder(N * 3, SOURCE_LEN, VOCAB_SIZE, D_MODEL, D_FF, D_KV,
                                          H, DROPOUT, self.shared_embedding_layer)
        self.decoder = TransformerDecoder(N, TARGET_LEN, VOCAB_SIZE, D_MODEL, D_FF, D_KV,
                                          H, DROPOUT, self.shared_embedding_layer)

    def call(self, inputs, training = False):
        encoder_inputs, decoder_inputs = inputs
        encoder_outputs, padding_mask = self.encode(encoder_inputs, training)
        decoder_outputs = self.decode(decoder_inputs, encoder_outputs, padding_mask, training)
        out = self.shared_embedding_layer(decoder_outputs, mode = 'linear')
        
        # softmax is omitted as hf doesn't use this either
        # instead we set from_logits = True on the loss function.
        # out = tf.nn.softmax(out) 
        
        # needed for fp16 training. https://www.tensorflow.org/guide/mixed_precision?hl=en
        out = tf.cast(out, dtype = 'float32')
        return out
    
    def encode(self, encoder_inputs, training = False):
        encoder_outputs, padding_mask = self.encoder(encoder_inputs, training)
        return encoder_outputs, padding_mask
    
    def decode(self, decoder_inputs, encoder_outputs, padding_mask, training = False):
        decoder_outputs = self.decoder(decoder_inputs, encoder_outputs, padding_mask, training)
        return decoder_outputs
