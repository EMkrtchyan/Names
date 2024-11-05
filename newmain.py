import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F


words = open('Data/SortedLowercase.txt', 'r', encoding='utf-8').read().splitlines()


N = torch.zeros((42,42), dtype=torch.int32)
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}

xs, ys = [], []

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

g = torch.Generator().manual_seed(21334331327)
W = torch.randn((43, 43), generator=g, requires_grad=True)


xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()

for k in range(3000):

    xenc = F.one_hot(xs, num_classes=43).float()
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdims=True)
    loss = -probs[torch.arange(num),ys].log().mean() + 0.01*(W**2).mean()
    print(loss.item())

    W.grad = None
    loss.backward()

    W.data += -5*W.grad


# finally, sample from the 'neural net' model
g = torch.Generator().manual_seed(2147483647)

for i in range(20):
  
  out = []
  ix = 0
  while True:
    
    # ----------
    # BEFORE:
    #p = P[ix]
    # ----------
    # NOW:
    xenc = F.one_hot(torch.tensor([ix]), num_classes=43).float()
    logits = xenc @ W # predict log-counts
    counts = logits.exp() # counts, equivalent to N
    p = counts / counts.sum(1, keepdims=True) # probabilities for next character
    # ----------
    
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix])
    if ix == 0:
      break
  print(''.join(out))