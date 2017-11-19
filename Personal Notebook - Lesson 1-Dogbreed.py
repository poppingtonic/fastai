
# coding: utf-8

# In[1]:


get_ipython().magic('reload_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('matplotlib inline')


# In[2]:


from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *


# In[24]:


get_ipython().magic('pinfo get_cv_idxs')


# In[17]:


PATH = "data/dogbreed/"
bs = 64
label_csv = f'{PATH}labels.csv'
n = len(list(open(label_csv)))-1
val_idxs = get_cv_idxs(n)
files = get_ipython().getoutput('ls {PATH}train | head')
files


# In[18]:


get_ipython().magic('pinfo2 get_cv_idxs')


# In[19]:


sz = 299


# In[20]:


label_df = pd.read_csv(label_csv)
label_df.head()


# In[21]:


files


# In[22]:


img = plt.imread(f'{PATH}train/{files[8]}')
plt.imshow(img);


# In[23]:


arch = resnext50


# In[24]:


label_df.pivot_table(index='breed', aggfunc=len).sort_values('id', ascending=False)


# In[25]:


# data = ImageClassifierData.from_paths(PATH, tfms=tfms_from_model(arch, sz))
tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
data = ImageClassifierData.from_csv(PATH, 'train', f'{PATH}labels.csv', suffix='.jpg', tfms=tfms, test_name='test', val_idxs=val_idxs, bs=bs)


# In[26]:


size_d = {k: PIL.Image.open(PATH+k).size for k in data.trn_ds.fnames}
row_sz, col_sz = list(zip(*size_d.values()))
row_sz[:5]


# In[27]:


get_ipython().magic('pinfo ImageClassifierData.from_csv')


# In[28]:


plt.hist(row_sz);


# In[29]:


def get_data(sz, bs):
    tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
    data = ImageClassifierData.from_csv(PATH, 'train', f'{PATH}labels.csv', test_name='test', num_workers=4, val_idxs=val_idxs, suffix='.jpg', tfms=tfms, bs=bs)
    return data if sz>300 else data.resize(340, 'tmp')


# In[30]:


data = get_data(sz, bs)


# In[31]:


learn = ConvLearner.pretrained(arch, data, precompute=True)
learn.fit(1e-2, 5)


# In[32]:


learn.precompute = False


# In[33]:


learn.precompute


# In[34]:


learn.fit(1e-2, 2, cycle_len=1)


# In[35]:


learn.lr_find()
learn.sched.plot()


# In[43]:


learn.save('299_pre')


# In[44]:


learn.lr_find()
learn.sched.plot()


# In[ ]:


learn.fit(0.2, 5, cycle_len=1)


# In[42]:


# learn.set_data(get_data(299, bs))
learn.freeze()


# In[43]:


learn.fit(1e-2, 3, cycle_len=1)


# In[44]:


learn.fit(1e-2, 3, cycle_len=1, cycle_mult=2)


# In[45]:


log_preds, y = learn.TTA()
probs = np.exp(log_preds)
accuracy(log_preds, y), metrics.log_loss(y, probs)


# In[46]:


learn.save('299_pre')


# In[47]:


learn.fit(1e-2, 1, cycle_len=2)


# In[48]:


log_preds, y = learn.TTA()
probs = np.exp(log_preds)
accuracy(log_preds, y), metrics.log_loss(y, probs)


# In[49]:


learn.save('299_pre')


# In[39]:


log_preds = learn.predict()
log_preds.shape


# In[41]:


log_preds[:10]


# In[42]:


preds = np.argmax(log_preds, axis=1)
probs = np.exp(log_preds[:,1])


# In[43]:


def rand_by_mask(mask):
    return np.random.choice(np.where(mask)[0], 4, replace=False)
def rand_by_correct(is_correct):
    return rand_by_mask((preds == data.val_y) == is_correct)


# In[92]:


def plot_val_with_title(idxs, title):
    imgs = np.stack([data.val_ds[x][0] for x in idxs])
    title_probs = [probs[x] for x in idxs]
    print(title)
    return plots(data.val_ds.denorm(imgs), rows=1, titles=title_probs)


# In[93]:


def plots(ims, figsize=(12, 6), rows=1, titles=None):
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, len(ims)//rows, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
            plt.imshow(ims[i])


# In[94]:


def load_img_id(ds, idx):
    return np.array(PIL.Image.open(PATH+ds.fnames[idx]))
def plot_val_with_title(idxs, title):
    imgs = [load_img_id(data.val_ds, x) for x in idxs]
    title_probs = [probs[x] for x in idxs]
    print(title)
    return plots(imgs, rows=1, titles=title_probs, figsize=(16, 8))


# In[97]:


plot_val_with_title(rand_by_correct(True), "correctly classified")


# In[98]:


# 2. A few incorrect labels at random
plot_val_with_title(rand_by_correct(False), "Incorrectly classified")


# In[101]:


def most_by_mask(mask, mult):
    idxs = np.where(mask)[0]
    return idxs[np.argsort(mult * probs[idxs])[:4]]

def most_by_correct(y, is_correct):
    mult = -1 if (y == 1) == is_correct else 1
    return most_by_mask((preds == data.val_y) == is_correct & (data.val_y == y), mult)


# In[102]:


plot_val_with_title(most_by_correct(0, True), "Most correct cats")


# In[103]:


plot_val_with_title(most_by_correct(1, True), "Most correct dogs")


# In[104]:


plot_val_with_title(most_by_correct(1, False), "Most incorrect dogs")


# In[105]:


plot_val_with_title(most_by_correct(0, False), "Most incorrect cats")


# In[106]:


most_uncertain = np.argsort(np.abs(probs -0.5))[:4]
plot_val_with_title(most_uncertain, "Most uncertain predictions")


# In[56]:


learn = ConvLearner.pretrained(arch, data, precompute=True)
lrf = learn.lr_find()


# In[69]:


learn.sched.plot_lr()


# In[58]:


tfms = tfms_from_model(resnet34, sz, aug_tfms=transforms_side_on, max_zoom=1.1)


# In[70]:


learn.sched.plot()


# In[71]:


def get_augs():
    data = ImageClassifierData.from_paths(PATH, bs=2, tfms=tfms, num_workers=4)
    x, _ = next(iter(data.aug_dl))
    return data.trn_ds.denorm(x)[1]


# In[72]:


ims = np.stack([get_augs() for i in range(6)])


# In[73]:


plots(ims, rows=2)


# In[74]:


data = ImageClassifierData.from_paths(PATH, tfms=tfms)
learn = ConvLearner.pretrained(arch, data, precompute=True)


# In[75]:


learn.fit(1e-2, 1)


# In[76]:


learn.precompute = False


# In[77]:


learn.fit(1e-2, 3, cycle_len=1)


# In[111]:


get_ipython().magic('pinfo2 learn.fit_gen')


# In[78]:


learn.sched.plot_lr()


# In[79]:


learn.save('224_lastlayer')


# In[80]:


learn.unfreeze()


# In[81]:


lr = np.array([1e-4, 1e-3, 1e-2])


# In[82]:


learn.fit(lr, 3, cycle_len=1, cycle_mult=2)


# In[83]:


learn.sched.plot_lr()


# In[84]:


learn.save('224_all')


# In[85]:


log_preds, y = learn.TTA()
accuracy(log_preds, y)


# In[86]:


preds = np.argmax(log_preds, axis=1)
probs = np.exp(log_preds[:, 1])


# In[87]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, preds)


# In[88]:


plot_confusion_matrix(cm, data.classes)


# In[107]:


plot_val_with_title(most_by_correct(0, False), "most incorrect cats")


# In[108]:


plot_val_with_title(most_by_correct(1, False), "most incorrect dogs")

