import numpy as np


class RNN:
    def __init__(self, sequence_length, vocabulary_size, hidden_size=100):
        """
            sequence_length : number of steps to unroll the RNN for
            vocabulary_size : number of characters/words/
        """
        self.sequence_length = sequence_length
        self.vocabulary_size = vocabulary_size
        self.hidden_size = hidden_size
        ##########################################
        ######### Parameters of the RNN ##########
        ##########################################
        self.W = np.random.randn(hidden_size, hidden_size)
        self.U = np.random.randn(hidden_size, vocabulary_size)
        self.V = np.random.randn(vocabulary_size, hidden_size)
        self.b = np.random.randn(hidden_size, 1)
        self.c = np.random.randn(vocabulary_size, 1)
        ##########################################
        ### Zero out the memory before training ##
        ##########################################
        self.h = np.zeros(shape=(hidden_size, 1))
        ###########################################
        ##### Scaling according to the DL book ####
        ###########################################
        matrix_scaler = np.sqrt(2 / (vocabulary_size + hidden_size))
        vector_scaler = np.sqrt(1 / vocabulary_size)
        self.W *= np.sqrt(2) / hidden_size
        self.U *= matrix_scaler
        self.V *= matrix_scaler
        self.b *= vector_scaler
        self.c *= vector_scaler
        ###########################################
        ##### Gradient memory for AdaGrad opt. ####
        ###########################################
        self.memory_W = np.zeros_like(self.W)
        self.memory_U = np.zeros_like(self.U)
        self.memory_V = np.zeros_like(self.V)
        self.memory_b = np.zeros_like(self.b)
        self.memory_c = np.zeros_like(self.c)

    def _foward_step(self, input):
        """
            input : must be one-hot encoded vocabulary_size vector
        """
        self.a = self.b + np.dot(self.W, self.h) + np.dot(self.U, input)
        self.h = np.tanh(self.a)
        self.o = self.c + np.dot(self.V, self.h)
        return np.exp(self.o) / np.sum(np.exp(self.o))

    def _back_propagate(self, inputs, targets, probs, activations,
                        hidden_states):
        self.V_derivative = np.zeros_like(self.V)
        self.U_derivative = np.zeros_like(self.U)
        self.W_derivative = np.zeros_like(self.W)
        self.b_derivative = np.zeros_like(self.b)
        self.c_derivative = np.zeros_like(self.c)

        for ind in reversed(range(self.sequence_length)):
            loss_derivative = probs[ind]
            loss_derivative[targets[ind]] -= 1
            self.V_derivative += np.outer(loss_derivative, hidden_states[ind])
            self.c_derivative += loss_derivative
            db = np.dot(loss_derivative.T,
                        self.V).T * (1 - np.tanh(activations[ind])**2)
            self.b_derivative += db
            self.U_derivative += np.outer(db, inputs[ind])
            self.W_derivative += np.outer(db, hidden_states[ind - 1])

    def _apply_gradients(self):
        ##########################################
        #########  Clipping gradients   ##########
        ##########################################
        np.clip(self.W_derivative, -1, 1, out=self.W_derivative)
        np.clip(self.U_derivative, -1, 1, out=self.U_derivative)
        np.clip(self.V_derivative, -1, 1, out=self.V_derivative)
        np.clip(self.b_derivative, -1, 1, out=self.b_derivative)
        np.clip(self.c_derivative, -1, 1, out=self.c_derivative)
        ##########################################
        #######     AdaGrad optiization    #######
        ##########################################
        self.memory_W += self.W_derivative ** 2
        self.memory_U += self.U_derivative ** 2
        self.memory_V += self.V_derivative ** 2
        self.memory_b += self.b_derivative ** 2
        self.memory_c += self.c_derivative ** 2
        self.W -= self.learning_rate * self.W_derivative / np.sqrt(self.memory_W + 1e-12)
        self.U -= self.learning_rate * self.U_derivative / np.sqrt(self.memory_U + 1e-12)
        self.V -= self.learning_rate * self.V_derivative / np.sqrt(self.memory_V + 1e-12)
        self.b -= self.learning_rate * self.b_derivative / np.sqrt(self.memory_b + 1e-12)
        self.c -= self.learning_rate * self.c_derivative / np.sqrt(self.memory_c + 1e-12)

    def _sample(self, seed_idx, sample_length=250):
        x = np.zeros((self.vocabulary_size, 1))
        x[seed_idx] = 1.
        seq = []
        h = np.zeros(shape=(self.hidden_size, 1))
        for t in range(sample_length):
            a = self.b + np.dot(self.W, h) + np.dot(self.U, x)
            h = np.tanh(a)
            o = self.c + np.dot(self.V, h)
            probs = np.exp(o) / np.sum(np.exp(o))
            idx = np.random.choice(range(self.vocabulary_size),
                                   p=probs.ravel())
            x = np.zeros((self.vocabulary_size, 1))
            x[idx] = 1.
            seq.append(idx)
        chars = [self.idx2char[idx] for idx in seq]
        return "".join(chars)

    def train(self, data, epochs=100, learning_rate=0.001):
        self.chars = data["chars"]
        self.char2idx = data["char_to_idx"]
        self.idx2char = data["idx_to_char"]
        self.learning_rate = learning_rate
        for epoch in range(epochs):
            loss_on_epoch_end = self._epoch_train()
            sample_seq = self._sample(self.char2idx["k"])
            print("Epoch : %d/%d" % (epoch + 1, epochs), "\t Loss : ",
                  loss_on_epoch_end)
            print("Sample : \n", sample_seq)

    def _epoch_train(self):
        offset = 0
        losses = []
        while offset + self.sequence_length + 1 < len(self.chars):
            loss = 0.

            input_seq = [
                self.char2idx[ch]
                for ch in self.chars[offset:offset + self.sequence_length]
            ]
            target_seq = [
                self.char2idx[ch]
                for ch in self.chars[offset + 1:offset + self.sequence_length +
                                     1]
            ]

            one_hot_chars, probs, activations, hidden_states = {}, {}, {}, {
                -1: self.h
            }

            for ind, ch in enumerate(input_seq):
                one_hot_chars[ind] = np.zeros(shape=(self.vocabulary_size, 1))
                one_hot_chars[ind][ch] = 1.
                probs[ind] = self._foward_step(one_hot_chars[ind])
                hidden_states[ind] = self.h
                activations[ind] = self.a
                loss -= np.log(probs[ind][target_seq[ind]][0] + 1e-12)

            self._back_propagate(one_hot_chars, target_seq, probs, activations,
                                 hidden_states)
            self._apply_gradients()

            offset += 1

            losses.append(loss)

        avarage_loss_on_epoch_end = np.mean(losses)
        return avarage_loss_on_epoch_end