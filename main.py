# To model images, we need joint distributions of parameters. 
# Our goal is to model complex distribution and sample from them.

# Classifier free guidance: Unet given an image and time step (t)
                     #--> In addition now prompt is also given as condition
                     # we train a single network, at some probablity we set conditioned signal to zero.
                     # So sometimes we get a conditioned output and sometimes we get an unconditioned output and we combine the weight to indicate how much we want the network to pay attention to the conditioning signal
                     #  