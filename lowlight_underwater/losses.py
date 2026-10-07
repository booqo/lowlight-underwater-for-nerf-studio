"""Image priors used by the low-light underwater model.

Extracted from the original experiment helpers without changing the loss formulas.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DarkChannelPriorLossV3(nn.Module):
    def __init__(self, cost_ratio=1000.):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.smooth_l1 = nn.SmoothL1Loss(beta=0.2)
        self.mse = nn.MSELoss()
        self.relu = nn.ReLU()
        self.cost_ratio = cost_ratio

    def forward(self, direct, depth=None):
        pos = self.l1(self.relu(direct), torch.zeros_like(direct))
        neg = self.smooth_l1(self.relu(-direct), torch.zeros_like(direct))
        # if (neg > 0):
        #     print(f"negative values inducing loss: {neg}")
        bs_loss = self.cost_ratio * neg + pos
        return bs_loss


class GrayWorldPriorLoss(nn.Module): # not good？
    def __init__(self, target_intensity=0.5):
        super(GrayWorldPriorLoss, self).__init__()
        self.target_intensity = target_intensity

    def forward(self, J):

        # assert B,C,H,W or B,C or H,W,C
        if len(J.size()) == 4:  # BCHW
            channel_intensities = torch.mean(J, dim=[-2, -1], keepdim=True)
        elif len(J.size()) == 3:
            if J.size(0) == 1:
                channel_intensities = torch.mean(J, dim=[-2, -1], keepdim=True)
            else:
                channel_intensities = torch.mean(J, dim=[-2, -1], keepdim=True)
        elif len(J.size()) == 2:
            channel_intensities = J.mean()
        else:
            raise ValueError(f"Unsupported image shape: {J.size()}")


        intensity_loss = (channel_intensities - self.target_intensity).square().mean()


        if torch.any(torch.isnan(intensity_loss)) or torch.any(torch.isinf(intensity_loss)):
            print("Warning: NaN or Inf detected in intensity loss!")
            intensity_loss = torch.zeros_like(intensity_loss)

        return intensity_loss


class EdgeSimilarityLoss(nn.Module): # work
    def __init__(self):
        super(EdgeSimilarityLoss, self).__init__()

        sobel_x = torch.tensor([[[[-1, 0, 1],
                                  [-2, 0, 2],
                                  [-1, 0, 1]]]], dtype=torch.float32)
        sobel_y = torch.tensor([[[[-1, -2, -1],
                                  [0, 0, 0],
                                  [1, 2, 1]]]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def rgb_to_grayscale(self, img):

        if img.dim() == 4:
            # [B, C, H, W]
            if img.shape[1] != 3:
                raise ValueError(f"Expected 3 channels for RGB, got {img.shape[1]}")
            gray = 0.299 * img[:, 0:1, :, :] + 0.587 * img[:, 1:2, :, :] + 0.114 * img[:, 2:3, :, :]
        elif img.dim() == 3:
            if img.shape[0] == 3:
                # [C, H, W]
                gray = 0.299 * img[0:1, :, :] + 0.587 * img[1:2, :, :] + 0.114 * img[2:3, :, :]
                gray = gray.unsqueeze(0)  # [1, 1, H, W]
            elif img.shape[2] == 3:
                # [H, W, C]
                gray = 0.299 * img[:, :, 0:1] + 0.587 * img[:, :, 1:2] + 0.114 * img[:, :, 2:3]
                gray = gray.permute(2, 0, 1).unsqueeze(0)  # [1, 1, H, W]
            elif img.shape[0] == 1 or img.shape[2] == 1:
                #  [1, H, W] or [H, W, 1]
                if img.shape[0] == 1:
                    gray = img.unsqueeze(1)  # [1, 1, H, W]
                else:
                    gray = img.permute(2, 0, 1).unsqueeze(0)  # [1, 1, H, W]
            else:
                raise ValueError(f"Unsupported channel dimension for 3D tensor: {img.shape}")
        elif img.dim() == 2:
            # [H, W]
            gray = img.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        else:
            raise ValueError(f"Unsupported image dimensions: {img.dim()}")
        return gray

    def forward(self, img1, img2):

        gray1 = self.rgb_to_grayscale(img1)  # [B, 1, H, W] 或 [1, 1, H, W]
        gray2 = self.rgb_to_grayscale(img2)  # [B, 1, H, W] 或 [1, 1, H, W]


        # self.display_images(gray1)
        # self.display_images(gray2)


        grad_x1 = F.conv2d(gray1, self.sobel_x, padding=1)  # [B, 1, H, W]
        grad_y1 = F.conv2d(gray1, self.sobel_y, padding=1)  # [B, 1, H, W]
        grad1 = torch.sqrt(grad_x1 ** 2 + grad_y1 ** 2 + 1e-6)

        grad_x2 = F.conv2d(gray2, self.sobel_x, padding=1)  # [B, 1, H, W]
        grad_y2 = F.conv2d(gray2, self.sobel_y, padding=1)  # [B, 1, H, W]
        grad2 = torch.sqrt(grad_x2 ** 2 + grad_y2 ** 2 + 1e-6)


        # self.display_gradients(grad1)
        # self.display_gradients(grad2)


        loss = F.l1_loss(grad1, grad2)
        return loss

    def display_images(self, gray):

        import matplotlib.pyplot as plt


        gray = gray.detach().cpu().numpy()


        if gray.shape[0] == 1 and gray.shape[1] == 1:
            gray_img = gray[0, 0, :, :]  # [H, W]
        else:
            raise ValueError(f"Unsupported grayscale image shape for display: {gray.shape}")

        plt.figure(figsize=(5, 5))
        plt.imshow(gray_img, cmap='gray')
        plt.title('Grayscale Image')
        plt.axis('off')
        plt.show()

    def display_gradients(self, grad):

        import matplotlib.pyplot as plt


        grad = grad.detach().cpu().numpy()


        if grad.shape[0] == 1 and grad.shape[1] == 1:
            grad_img = grad[0, 0, :, :]  # [H, W]
        else:
            raise ValueError(f"Unsupported gradient image shape for display: {grad.shape}")

        plt.figure(figsize=(5, 5))
        plt.imshow(grad_img, cmap='gray')
        plt.title('Gradient Image')
        plt.axis('off')
        plt.show()


def exposure_loss(enhanced_image, block_size=32, target_mean=0.5): # work

    H, W, C = enhanced_image.shape


    gray = 0.299 * enhanced_image[:, :, 0] + 0.587 * enhanced_image[:, :, 1] + 0.114 * enhanced_image[:, :, 2]  # [H, W]


    loss = 0.0
    num_blocks = 0


    for i in range(0, H, block_size):
        for j in range(0, W, block_size):
            block = gray[i:i + block_size, j:j + block_size]
            if block.numel() == 0:
                continue
            block_mean = block.mean()
            loss += (block_mean - target_mean) ** 2
            num_blocks += 1


    if num_blocks == 0:
        return torch.tensor(0.0, device=enhanced_image.device)


    loss = loss / num_blocks

    return loss
