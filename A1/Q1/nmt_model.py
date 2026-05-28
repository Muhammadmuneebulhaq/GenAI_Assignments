import tensorflow as tf

class Encoder(tf.keras.layers.Layer):
    """Vanilla RNN Encoder"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.0):
        super(Encoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.rnn_layers = [
            tf.keras.layers.SimpleRNN(hidden_dim, return_sequences=(i < num_layers - 1),
                                    return_state=True, dropout=dropout if i > 0 else 0.0)
            for i in range(num_layers)
        ]
    
    def call(self, x, training=False):
        x = self.embedding(x)
        states = []
        for i, rnn_layer in enumerate(self.rnn_layers):
            if i < len(self.rnn_layers) - 1:
                x, state = rnn_layer(x, training=training)
            else:
                _, state = rnn_layer(x, training=training)
            states.append(state)
        return state, states

class Decoder(tf.keras.layers.Layer):
    """Vanilla RNN Decoder"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.0):
        super(Decoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.rnn_layers = [
            tf.keras.layers.SimpleRNN(hidden_dim, return_sequences=True, return_state=True,
                                    dropout=dropout if i > 0 else 0.0)
            for i in range(num_layers)
        ]
        self.output_dense = tf.keras.layers.Dense(vocab_size)
    
    def call(self, x, states_input, training=False):
        x = self.embedding(x)
        final_states = []
        for i, rnn_layer in enumerate(self.rnn_layers):
            x, state = rnn_layer(x, initial_state=states_input[i], training=training)
            final_states.append(state)
        logits = self.output_dense(x)
        return logits, final_states

class NMTModel(tf.keras.Model):
    """Full NMT Model"""
    def __init__(self, encoder, decoder, vocab_sizes, pad_idx=0):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_sizes = vocab_sizes
        self.pad_idx = pad_idx
    
    def call(self, encoder_inputs, decoder_inputs, training=False):
        context_vector, encoder_states = self.encoder(encoder_inputs, training=training)
        decoder_logits, _ = self.decoder(decoder_inputs, encoder_states, training=training)
        return decoder_logits
