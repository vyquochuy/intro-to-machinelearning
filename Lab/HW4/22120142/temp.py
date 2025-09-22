import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def d_sigmoid(a):
    return a * (1 - a)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # chống tràn
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def compute_cross_entropy(y_true, y_prob):
    N = y_true.shape[0]
    y_onehot = np.eye(y_prob.shape[1])[y_true]
    log_probs = np.log(y_prob + 1e-12)  # chống log(0)
    loss = -np.sum(y_onehot * log_probs) / N
    return loss

def train_nnet(X, y, initial_Ws, mb_size, lr, max_epoch, hid_layer_sizes):
    Ws = [w.copy() for w in initial_Ws]
    N, d_plus_1 = X.shape
    num_classes = Ws[-1].shape[1]
    losses = []

    for epoch in range(max_epoch):
        indices = np.random.permutation(N)
        for start in range(0, N, mb_size):
            end = min(start + mb_size, N)
            batch_idx = indices[start:end]
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]

            # === Forward Pass ===
            As = [X_batch]
            A = X_batch

            for i, W in enumerate(Ws):
                Z = A @ W
                is_last = (i == len(Ws) - 1)
                if is_last:
                    A = softmax(Z)
                else:
                    A = sigmoid(Z)
                    A = np.hstack((np.ones((A.shape[0], 1)), A))  # thêm bias
                As.append(A)

            # === Backward Pass ===
            grads = [None] * len(Ws)
            y_onehot = np.eye(num_classes)[y_batch]

            # Gradient tại tầng cuối
            delta = (As[-1] - y_onehot) / len(y_batch)

            for i in reversed(range(len(Ws))):
                A_prev = As[i]
                grads[i] = A_prev.T @ delta

                if i > 0:
                    W_no_bias = Ws[i][1:, :]
                    delta = delta @ W_no_bias.T
                    A_prev_no_bias = As[i][:, 1:]  # bỏ bias
                    delta *= d_sigmoid(A_prev_no_bias)

            # === Cập nhật trọng số ===
            for i in range(len(Ws)):
                Ws[i] -= lr * grads[i]

        # === Tính độ lỗi cross-entropy sau mỗi epoch ===
        probs = compute_nnet_output(Ws, X, return_what='prob')
        loss = compute_cross_entropy(y, probs)
        losses.append(loss)
        # Optional: in ra tiến độ
        # print(f"Epoch {epoch+1}/{max_epoch} - Loss: {loss:.4f}")
    return Ws, losses
