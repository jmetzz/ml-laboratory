
# coding: utf-8

# In[4]:


# start by importing some libraries that we will use
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# to be able to reproduce the same results on every run 
# let's already generate the random seed
np.random.seed(444)

plt.plot([1,2,3])


# In[5]:


fig, _ = plt.subplots()
type(fig)


# In[7]:


one_tick = fig.axes[0].yaxis.get_major_ticks()[0]
type(one_tick)


# In[18]:


fig, ax = plt.subplots()


# In[19]:


ax

