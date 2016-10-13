import matplotlib.pyplot as plt

blx = [2**k for k in range(5,11)]
div =65536

nblks = [div/k for k in blx]
pcct = []

for nbx,bx in zip(nblks,blx):
    x = range(bx,3,-4)+range(4,bx+1,4)
    nall = bx*len(x)*nbx
    pct = float(sum(x)*nbx)/float(nall)
    pcct.append(pct)
    print bx, "Percent threads active: {}".format(pct)

plt.plot(blx,pcct)
plt.show()
