+++
date = '2026-07-18T12:29:35+01:00'
draft = false
title = 'Research Note-0'
+++
# Lessons for multi-GPU training
This is the learning note when I was trying to run experiments on multiple GPUs.
## Decide on batch size convention
Per-GPU batch size:what you pass to your dataloader(batch_size=..)

Effective batch size : per_gpu_batch_size * num_gpus*grad_accum_steps

grad_accum_steps: refers to how many steps updating the gradient, in my case grad_accum_steps=1

## Check sampler
Single gpu: randomsampler. 

Multi GPU(DDP): Lightning automatically injects DistributedSampler.

## Adjust batch size or learning rate
If you want the same effective batch size across 1-GPU and 2-GPU runs, divide your batch_size by num_gpus.

Example: change 1 GPU, batch=256 → 2 GPUs, set batch=128 in your script.

Alternatively, you need to keep the same batch_size per GPU and scale the learning rate (linear scaling rule). i.e. , For 1GPU, bs = 256, lr = 0.1;  then for 2GPUS, bs=256 for each rank, total_bs = 512, lr = 0.2

## Sanity check logging
Print inside training_step:
```python
if self.global_rank == 0:
    print(f"Per-GPU batch size: {batch.size(0)}, Effective batch size: {batch.size(0)*trainer.world_size}")
```
## Evaluation consistency
For dataloader, ensure your validation/test loaders are created with shuffle=False


## Randomness control
Set random seed for each run, to get reproduced results, i.e., pl.seed_everything(seed) 


## Some pitfalls
Do not use the shape of tensor to do some computation: i.e. when computing attention score, the dim of each head, use the rep-computed value, not the shape of the tensor.

When I tried to share weights between the embeddings of encoder and decoder, I pass them as parameters into a class, the problem is that creating them outside of the model does not update the internal registry. PyTorch didn’t update the internal parameter registry correctly inside proj. The nn.Module that you passed to buildModel already had its own proj._parameters['weight'] pointing to the original proj.weight. Your reassignment didn’t re-register it, so optimizers don’t see the shared parameter properly. But if you create them inside of the model’s __init__, it immediately updates the internal registry of the newly constructed module.  Or you can create them outside of the model like this :
```python
emb = nn.Embedding(vocab_size, d_model)
proj = nn.Linear(d_model, vocab_size, bias=False)

# Remove proj’s old parameter and register emb.weight
proj.weight = emb.weight # this only rebinds the attribute, this means simply making proj.weight point to emb.weight instead. But the module’s internal parameter registry (proj._parameters['weight']) still points to the old original parameter.
proj.register_parameter('weight', emb.weight) #  this updates proj._parameters to point to emb.weight
```


my previous code is:
```python
class ModelB(nn.Module):
    define __init__(self, embed, proj):
        super().__init()
        self.embed = embed
        self.proj = proj
        
def createModelB():
    embed = nn.Embedding(v_s, d_model)
    proj = nn.Linear(d_model, v_s, bias=False)
    proj.weights = emb.weights #proj.weights still point to the old orginal parameter
    modelb = ModelB(emb, proj)
    return modelb
```

correct version:
```python
class ModelB(nn.Module):
    define __init__(self, v_s, d_model):
        super().__init()
        self.embed = nn.Embedding(v_s, d_model)
        self.proj = nn.Linear(d_model, v_s, biad=False)
        self.proj.weight = self.embed.weight
        
def createModelB():

    modelb = ModelB(v_s, d_model)
    return modelb
```

another correct way:

```python
class ModelB(nn.Module):
    define __init__(self, embed, proj):
        super().__init()
        self.embed = nn.Embedding(v_s, d_model)
        self.proj = nn.Linear(d_model, v_s, biad=False)
        self.proj.weight = self.embed.weight
        
def createModelB():
    embed =  nn.Embedding(v_s, d_model)
    proj = nn.Linear(d_model, v_s, biad=False)
    proj.weight = embed.weight
    proj.register_parameter('weight', embed.weight)
    modelb = ModelB(embed, proj)
    return modelb
```
