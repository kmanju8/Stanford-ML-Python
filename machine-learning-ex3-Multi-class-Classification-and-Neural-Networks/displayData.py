import numpy as np
from PIL import Image

def plotsamples(X):

    samples=np.random.randint(0, high=5000, size=100)
    images=np.zeros((200,200))
    
    for i in range(100):
        digit=(X[(samples[i]),:].reshape(20,20))
        images[20*int(np.floor(i/10)):(20*int(np.floor(i/10))+20), 20*int(np.floor(i%10)):int(20*np.floor(i%10)+20)]=digit

    ans=np.uint8(np.transpose(images))
    img = Image.fromarray(ans,'L')
    img.show()

