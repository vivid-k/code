import torch
import numpy
sample_prob = torch.FloatTensor(6).uniform_(0, 1)
sample_mask = sample_prob < 0.2
a = sample_mask.nonzero()
print(sample_mask)

b = numpy.array([[[5,6,4,1],
                 [7,8,3,2],
                 [1,2,1,3]]])
c = numpy.array([[[5],[6],[7]]])
x = torch.tensor([[3,3],[3,3]])
y = b*c  #x.dot(x)
z = torch.mul(b,c)  #x.mul(x)
print(y)
print(z)
e = c[-1,:]
d = b*c[-1,:, None]
print(d)
all_masks = numpy.array([[0,1,
0]])
(finished,) = numpy.where(all_masks[-1] == 0) 

print(finished)
m = set()
m.add(torch.tensor([1,4,3]))
m.add(torch.tensor([2,4,3]))
m.add(torch.tensor([1,5,3]))
a = [1,4,3]
a = torch.tensor(a)
if a in m:
    print("yes")


