"""Code for the Lab 10 exercise (STAT 154/254): neural network fundamentals.

Put this file in the same folder as your notebook (on Google Colab, upload it with the folder icon on the left),
and start your notebook with

    from lab10_code import *

This gives you the imports below, the code from the presentation that the exercise builds on, and the spiral data
for Part 2. The first import downloads MNIST (about 60 MB) to ./data.
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from torchvision import datasets, transforms

# use a GPU if there is one: "cuda" for NVIDIA GPUs, "mps" for Apple silicon
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


# =================================================================================================================
# Part 1: neural networks as functions (section 1 of the presentation)
# =================================================================================================================

class ShallowNet(nn.Module):
    """f(x) = phi_0 + sum_d phi_d ReLU(theta_d0 + theta_d1 x), with D hidden units."""

    def __init__(self, D):
        super().__init__()
        self.hidden = nn.Linear(1, D)    # the thetas: weight (D, 1) holds the slopes theta_d1, bias (D,) the theta_d0
        self.output = nn.Linear(D, 1)    # the phis: weight (1, D) holds phi_1, ..., phi_D, bias (1,) holds phi_0

    def hidden_units(self, x):
        return torch.relu(self.hidden(x))

    def forward(self, x):
        return self.output(self.hidden_units(x))

    def joints(self):
        """The points x = -theta_d0 / theta_d1 where each hidden unit switches on or off."""
        return (-self.hidden.bias / self.hidden.weight[:, 0]).detach().numpy()


def relu(z):
    return np.maximum(z, 0)


def f1(x):
    """Shallow network with 3 hidden units: a zigzag that sweeps [0, 1] three times as x goes from 0 to 1."""
    return 3 * relu(x) - 6 * relu(x - 1 / 3) + 6 * relu(x - 2 / 3)


# =================================================================================================================
# Part 2: classifiers learn their own features (section 2 of the presentation, on spirals instead of circles)
# =================================================================================================================

def make_spirals(n_per_class=300, n_classes=3, turns=0.75, noise=0.09, seed=0):
    """Points along n_classes spiral arms around the origin, one arm per class, plus Gaussian noise."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for k in range(n_classes):
        r = rng.uniform(0, 1, n_per_class) ** 0.8               # distance from the center
        angle = 2 * np.pi * (k / n_classes + turns * r)           # each arm turns as r grows
        points = np.column_stack([r * np.cos(angle), r * np.sin(angle)])
        X.append(points + noise * rng.standard_normal((n_per_class, 2)))
        y.append(np.full(n_per_class, k))
    return np.vstack(X), np.concatenate(y)


X_spiral, y_spiral = make_spirals()                    # 900 points in 2 dimensions, 3 classes
X_spiral_t = torch.tensor(X_spiral, dtype=torch.float32)
y_spiral_t = torch.tensor(y_spiral)                    # class labels as integers 0, 1, 2
spiral_data = torch.utils.data.TensorDataset(X_spiral_t, y_spiral_t)
class_colors = ListedColormap(["C0", "C1", "C2"])      # one color per class, for plt.scatter(..., cmap=class_colors)


class SpiralNet(nn.Module):
    """The circle network of the presentation, with 3 classes: an encoder computes the embedding z(x), and the
    classifier is a logistic regression on z(x)."""

    def __init__(self, hidden=16, embed_dim=2, n_classes=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, embed_dim),          # the embedding z(x)
        )
        self.classifier = nn.Linear(embed_dim, n_classes)    # logistic regression on z(x)

    def embedding(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.classifier(self.embedding(x))      # the logits


def train_spiral_net(lr, optimizer_class=torch.optim.Adam, embed_dim=2, n_epochs=50, batch_size=32, seed=0):
    """Train a SpiralNet on the spirals, like train_circle_net in the presentation.
    Returns the network and the training cross-entropy after each epoch."""
    torch.manual_seed(seed)
    net = SpiralNet(embed_dim=embed_dim)
    loader = torch.utils.data.DataLoader(spiral_data, batch_size=batch_size, shuffle=True)
    optimizer = optimizer_class(net.parameters(), lr=lr)
    epoch_losses = []
    for epoch in range(n_epochs):
        for xb, yb in loader:                     # one epoch = one pass over all the minibatches
            optimizer.zero_grad()                 # 1. reset the gradients from the previous step
            logits = net(xb)                      # 2. forward pass
            loss = F.cross_entropy(logits, yb)    # 3. softmax + negative log-likelihood, averaged over the batch
            loss.backward()                       # 4. backpropagation: fills p.grad for every parameter p
            optimizer.step()                      # 5. gradient step
        with torch.no_grad():                     # no gradients needed to evaluate the loss
            epoch_losses.append(F.cross_entropy(net(X_spiral_t), y_spiral_t).item())
    return net, epoch_losses


def plot_decision_regions(net, X, y, ax=None, lim=1.4):
    """Scatter plot of 2-dimensional data X colored by class y, on top of the class that net predicts (shaded)."""
    if ax is None:
        ax = plt.gca()
    grid = np.linspace(-lim, lim, 300)
    g1, g2 = np.meshgrid(grid, grid)
    G = torch.tensor(np.column_stack([g1.ravel(), g2.ravel()]), dtype=torch.float32)
    net.eval()
    with torch.no_grad():
        pred = net(G).argmax(dim=1).reshape(g1.shape).numpy()
    n_classes = int(np.max(y)) + 1
    cmap = ListedColormap([f"C{k}" for k in range(n_classes)])
    ax.contourf(g1, g2, pred, levels=np.arange(n_classes + 1) - 0.5, cmap=cmap, alpha=0.2)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, vmin=-0.5, vmax=n_classes - 0.5, s=6)
    ax.set(xlabel="$x_1$", ylabel="$x_2$")
    ax.set_aspect("equal")
    return ax


# =================================================================================================================
# Part 3: training a CNN on MNIST (sections 3 to 5 of the presentation)
# =================================================================================================================

# ToTensor turns each image into a float tensor of shape (1, 28, 28) with values in [0, 1]
train_full = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())
train_set, val_set = torch.utils.data.random_split(train_full, [50_000, 10_000],
                                                   generator=torch.Generator().manual_seed(0))

train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)   # minibatches for SGD
val_loader = torch.utils.data.DataLoader(val_set, batch_size=1000)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000)

N_EPOCHS = 5    # the number of epochs used in the presentation


def evaluate(model, loader, transform=None):
    """Average cross-entropy and accuracy of the model on a data set."""
    model.eval()
    total_loss, n_correct, n = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            if transform is not None:
                xb = transform(xb)
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            total_loss += F.cross_entropy(logits, yb, reduction="sum").item()
            n_correct += (logits.argmax(dim=1) == yb).sum().item()
            n += len(yb)
    return total_loss / n, n_correct / n


def train(model, train_loader, n_epochs, lr=1e-3, weight_decay=0.0, transform=None, verbose=True):
    """Train with Adam. After each epoch, record the training loss (the average of the minibatch losses during
    the epoch, which is cheap to compute but lags a little behind) and the validation loss and accuracy.
    transform, if given, is applied to every minibatch of images, training and validation."""
    model.to(device)      # the GPU, if there is one
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    for epoch in range(n_epochs):
        model.train()      # model.train() and model.eval() matter for layers like dropout; a good habit
        total_loss, n = 0.0, 0
        for xb, yb in train_loader:
            if transform is not None:
                xb = transform(xb)
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
            n += len(yb)
        val_loss, val_acc = evaluate(model, val_loader, transform)
        history["train_loss"].append(total_loss / n)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        if verbose:
            print(f"  epoch {epoch + 1:2d}: training loss {total_loss / n:.3f}, "
                  f"validation loss {val_loss:.3f}, validation accuracy {val_acc:.1%}")
    return history


def make_mlp():
    """The fully connected network of the presentation: two hidden layers of 128 units."""
    return nn.Sequential(
        nn.Flatten(),                                  # (batch, 1, 28, 28) -> (batch, 784)
        nn.Linear(28 * 28, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, 10),                            # the 10 logits
    )


def make_cnn(channels=(16, 32, 64), hidden=64):
    """A CNN with one block (3 x 3 convolution, ReLU, 2 x 2 max pooling) per entry of channels, which is the number
    of channels of that block, followed by a fully connected layer with `hidden` units and the 10 logits.
    Each block halves the height and width of the image (28 -> 14 -> 7 -> 3 -> 1), so use at most 4 blocks.
    make_cnn() is the CNN of the presentation; make_cnn((32,)) has a single block with 32 channels."""
    layers, c_in, size = [], 1, 28
    for c_out in channels:
        layers += [nn.Conv2d(c_in, c_out, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
        c_in, size = c_out, size // 2
    layers += [nn.Flatten(), nn.Linear(c_in * size * size, hidden), nn.ReLU(), nn.Linear(hidden, 10)]
    return nn.Sequential(*layers)


def print_shapes(model):
    """Pass one image through the network, printing the output shape and number of parameters of each layer."""
    x = torch.zeros(1, 1, 28, 28)
    print(f"{'input':<12}{str(tuple(x.shape[1:])):<16}")
    for layer in model:
        x = layer(x)
        n_params = sum(p.numel() for p in layer.parameters())
        print(f"{layer.__class__.__name__:<12}{str(tuple(x.shape[1:])):<16}{n_params:>9,} parameters")
    print(f"{'total':<28}{sum(p.numel() for p in model.parameters()):>9,} parameters")


def shift(images, dx=0, dy=0):
    """Translate images (last two dimensions 28 x 28) by dx pixels right and dy pixels down, filling with zeros."""
    out = torch.roll(images, shifts=(dy, dx), dims=(-2, -1))
    if dy > 0:
        out[..., :dy, :] = 0
    if dy < 0:
        out[..., dy:, :] = 0
    if dx > 0:
        out[..., :, :dx] = 0
    if dx < 0:
        out[..., :, dx:] = 0
    return out


def shift_accuracy(model, k, loader=None):
    """Accuracy on the images of loader (the test set by default) shifted by k pixels, averaged over the four
    directions. For example, shift_accuracy(model, 3, val_loader) uses validation images shifted by 3 pixels."""
    if loader is None:
        loader = test_loader
    return np.mean([evaluate(model, loader, transform=lambda imgs: shift(imgs, dx, dy))[1]
                    for dx, dy in [(k, 0), (-k, 0), (0, k), (0, -k)]])
