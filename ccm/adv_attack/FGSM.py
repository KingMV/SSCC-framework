import torch
import torch.nn as nn


class FGSM(object):
    def __init__(self, model, eps=0.007):
        self.eps = eps
        self.model = model

    def forward(self, images, labels):
        images = images.clone().detach().cuda()
        labels = labels.clone().detach().cuda()

        images_min, images_max = images.min(), images.max()

        loss = nn.MSELoss(reduction='mean')
        images.requires_grad = True

        outputs = self.model(images)
        # cost = loss(outputs, labels) / outputs.shape[0]
        cost = loss(outputs, labels)
        # Update adversarial images
        grad = torch.autograd.grad(cost, images,
                                   retain_graph=False, create_graph=False)[0]
        # print(images.max(), images.min(), grad.max(), grad.min())
        adv_images = images + self.eps * grad.sign()
        adv_images = torch.clamp(adv_images, min=images_min, max=images_max).detach()
        # adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        adv_noise = torch.clamp(self.eps * grad.sign(), min=-1, max=1)
        return adv_images, adv_noise

    def __call__(self, *args, **kwargs):
        images = self.forward(*args, **kwargs)
        return images
