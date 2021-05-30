# Comments on Ex 3

Most of the difficulty came from part 1 of the exercise; implementing pretrained weights is hardly a challenge.

For this section of the course however, the challenge by and large came from the more independent programming aspects, over the course content.
To begin with, the intial tasks require linearization of the classification algorithm; while the maths was rather trivial, making sure tensors were of the right shape was far more finnicky.
Additionally plotting a sample of the data set to visualize the digits did not come easily.
Luckily I am accustomed to the Pillow library so it wasn't too tough, but it was clear some hoops must've been jumped in the original code in order to get plots like those shown in the assignment.

First part was fine overall however. Training the multiclassifier took quite some time, a little under an hour, which I feel is the limit to which I am happy waiting when running locally.
From here, I am keen to switch to pytorch. Whether I need to or not for the exercises is less of the issue, but I would like my code to be scalable for my purposes, say if I choose to work on different datasets.


Ultimately, as with previous exercises, writing my own 'boilerplate' code is where the challenge lies, but it has been great for learning more python libraries.
From next week the challenge clearly increases with backpropogation, though I already suspect the real challenge will be in getting accustomed to pytorch.