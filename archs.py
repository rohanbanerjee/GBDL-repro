import torch
import random
import math
from torch import nn
from torch.nn import init
import torch.nn.functional as F

__all__ = ['MC_UNet', 'LRL', 'UNet', 'LRL_imp', 'LRL_lite', 'LRL_new']


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, act = nn.ReLU(inplace=True)):
        super().__init__()
        self.relu = act
        self.conv1 = nn.Conv3d(in_channels, middle_channels, (3, 3, 3), padding=(1, 1, 1))
        self.in1 = nn.InstanceNorm3d(middle_channels)
        self.conv2 = nn.Conv3d(middle_channels, out_channels, (3, 3, 3), padding=(1, 1, 1))
        self.in2 = nn.InstanceNorm3d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.in1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.in2(out)
        out = self.relu(out)

        return out

class VGGBlock_MC(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, act=nn.ReLU(inplace=True)):
        super().__init__()
        self.relu = act
        self.conv1 = nn.Conv3d(in_channels, middle_channels, 3, padding=1)
        self.in1 = nn.InstanceNorm3d(middle_channels)
        self.conv2 = nn.Conv3d(middle_channels, out_channels, 3, padding=1)
        self.in2 = nn.InstanceNorm3d(out_channels)

    def forward(self, x, train=True):
        out = self.conv1(F.dropout3d(x, training=train, p=0.3))
        out = self.in1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.in2(out)
        out = self.relu(out)

        return out 
 
 
class MC_UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()
 
        nb_filter = [32, 64, 96, 192, 384] 

        self.pool = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.up = nn.Upsample(scale_factor=(1.0, 2.0, 2.0), mode='trilinear', align_corners=True)
 
        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])

        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[1], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[2], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[3], nb_filter[4])

        self.conv3_1 = VGGBlock_MC(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock_MC(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock_MC(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock_MC(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv3d(nb_filter[0], num_classes, 1, padding=0)

        for m in self.modules():
          if isinstance(m, nn.Conv3d):
            init.kaiming_normal_(m.weight)
            if m.bias is not None:
               init.zeros_(m.bias)
          elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def forward(self, input, train=True, T=10):


        x0_0 = self.conv0_0(input)     
        
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
 
        output_final = None 

        for i in range(0, T):
 
          x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
          x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
          x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
          x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))
     
          output = self.final(x0_4)
        
          if T <= 1: 
            output_final = output     
          else:
            if i == 0:
               output_final = output.unsqueeze(0)
            else:
               output_final = torch.cat((output_final, output.unsqueeze(0)), dim=0)
   
        return output_final


class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()
 
        nb_filter = [32, 64, 96, 192, 384] 

        self.pool = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.up = nn.Upsample(scale_factor=(1.0, 2.0, 2.0), mode='trilinear', align_corners=True)
 
        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])

        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[1], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[2], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[3], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv3d(nb_filter[0], num_classes, 1, padding=0)

        for m in self.modules():
          if isinstance(m, nn.Conv3d):
            init.kaiming_normal_(m.weight)
            if m.bias is not None:
               init.zeros_(m.bias)
          elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def forward(self, input, train=True, T=1):


        x0_0 = self.conv0_0(input)     
        
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
 
        output_final = None 

        for i in range(0, T):
 
          x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
          x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
          x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
          x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))
     
          output = self.final(x0_4)
        
          if T <= 1: 
            output_final = output     
          else:
            if i == 0:
               output_final = output.unsqueeze(0)
            else:
               output_final = torch.cat((output_final, output.unsqueeze(0)), dim=0)
   
        return output_final






class LRL(nn.Module):
      def __init__(self, num_classes, input_channels=3, resize_w = 128, resize_h = 128, **kwargs):
        super().__init__()

        self.nb_filter = [64, 128, 256, 512, 512]

        self.pool = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.up = nn.Upsample(scale_factor=(1.0, 2.0, 2.0), mode='trilinear', align_corners=True)
        self.relu = nn.ReLU(inplace=True)

        w = int(resize_w/32)
        h = int(resize_h/32)

        self.conv0_x = VGGBlock(input_channels, self.nb_filter[0], self.nb_filter[0], nn.ReLU(inplace=True))
        self.conv1_x = VGGBlock(self.nb_filter[0], self.nb_filter[1], self.nb_filter[1], nn.ReLU(inplace=True))
        self.conv2_x = VGGBlock(self.nb_filter[1], self.nb_filter[2], self.nb_filter[2], nn.ReLU(inplace=True))
        self.conv3_x = VGGBlock(self.nb_filter[2], self.nb_filter[3], self.nb_filter[3], nn.ReLU(inplace=True))
        self.conv4_x = VGGBlock(self.nb_filter[3], self.nb_filter[4], self.nb_filter[4], nn.ReLU(inplace=True))
      
        self.conv_down = nn.Conv3d(self.nb_filter[4], self.nb_filter[4], (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.conv_down_ = nn.Conv3d(self.nb_filter[4], self.nb_filter[4], (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.mean_vec = nn.Linear(self.nb_filter[4]*w*h, int(self.nb_filter[3]/2)) 

        self.covar_half = nn.Linear(self.nb_filter[4]*w*h, int(self.nb_filter[3]/2))
        self.covar_diag = nn.Linear(self.nb_filter[4]*w*h, int(self.nb_filter[3]/2))

        self.x_ori_vec = nn.Linear(self.nb_filter[4]*w*h, int(self.nb_filter[3]/2))

        self.transform = nn.Linear(int(self.nb_filter[3]/2), self.nb_filter[4]*w*h)
        self.transform_ = nn.Linear(int(self.nb_filter[3]/2), self.nb_filter[4]*w*h)

        self.conv_up = nn.ConvTranspose3d(self.nb_filter[4], self.nb_filter[4], (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.conv_up_ = nn.ConvTranspose3d(self.nb_filter[4], self.nb_filter[4], (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
       
        self.conv3_1_x = VGGBlock(self.nb_filter[3]+self.nb_filter[4], self.nb_filter[3], self.nb_filter[3], nn.ReLU(inplace=True))
        self.conv2_2_x = VGGBlock(self.nb_filter[2]+self.nb_filter[3], self.nb_filter[2], self.nb_filter[2], nn.ReLU(inplace=True))
        self.conv1_3_x = VGGBlock(self.nb_filter[1]+self.nb_filter[2], self.nb_filter[1], self.nb_filter[1], nn.ReLU(inplace=True))
        self.conv0_4_x = VGGBlock(self.nb_filter[0]+self.nb_filter[1], self.nb_filter[0], self.nb_filter[0], nn.ReLU(inplace=True))

        
        self.conv00 = nn.Conv3d(self.nb_filter[0], input_channels, 3, stride=1, padding=1)


        self.tanh = nn.Tanh()
       

        self.conv0_0 = VGGBlock(input_channels, self.nb_filter[0], self.nb_filter[0], nn.ReLU(inplace=True))
        self.conv1_0 = VGGBlock(self.nb_filter[0], self.nb_filter[1], self.nb_filter[1], nn.ReLU(inplace=True))
        self.conv2_0 = VGGBlock(self.nb_filter[1], self.nb_filter[2], self.nb_filter[2], nn.ReLU(inplace=True))
        self.conv3_0 = VGGBlock(self.nb_filter[2], self.nb_filter[3], self.nb_filter[3], nn.ReLU(inplace=True))
        self.conv4_0 = VGGBlock(self.nb_filter[3], self.nb_filter[4], self.nb_filter[4], nn.ReLU(inplace=True))

        self.conv3_1 = VGGBlock(self.nb_filter[3]+self.nb_filter[4]+self.nb_filter[4], self.nb_filter[3], self.nb_filter[3], nn.ReLU(inplace=True))
        self.conv2_2 = VGGBlock(self.nb_filter[2]+self.nb_filter[3], self.nb_filter[2], self.nb_filter[2], nn.ReLU(inplace=True))
        self.conv1_3 = VGGBlock(self.nb_filter[1]+self.nb_filter[2], self.nb_filter[1], self.nb_filter[1], nn.ReLU(inplace=True))
        self.conv0_4 = VGGBlock(self.nb_filter[0]+self.nb_filter[1], self.nb_filter[0], self.nb_filter[0], nn.ReLU(inplace=True))

        self.final = nn.Conv3d(self.nb_filter[0], num_classes, kernel_size=1)
       
        for m in self.modules():
          if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            init.kaiming_normal_(m.weight)
            if m.bias is not None:
               init.zeros_(m.bias) 

      def reparameterize(self, mu, covar_half, M=1):
         
  
        B, depth, D = mu.size()
        
        if M == 1:
          eps = torch.randn(B * depth, 1, D).cuda()
          eps_ = torch.bmm(eps, covar_half)
          return eps_.view(B, depth, D) + mu
        else:
          latent = None
          # latent will be (M, B, depth, D)
          for i in range(M):
             eps = torch.randn(B * depth, 1, D).cuda()
             eps_ = torch.bmm(eps, covar_half)
             lat = eps_.view(B, depth, D) + mu
             if i == 0:
                latent =  lat.unsqueeze(0)
             else:
                latent = torch.cat((latent, lat.unsqueeze(0)), 0)
          return latent.view(B*M, depth, D)


      def forward(self, input, M=1):
        x0_x = self.conv0_x(input)
        x1_x = self.conv1_x(self.pool(x0_x))
        x2_x = self.conv2_x(self.pool(x1_x))
        x3_x = self.conv3_x(self.pool(x2_x))
        x4_x = self.conv4_x(self.pool(x3_x))
        x4_ = self.conv_down(x4_x)
        print("input --> ", input.size())
        print("x0_x --> ", x0_x.size())
        print("x1_x --> ", x1_x.size())
        print("x2_x --> ", x2_x.size())
        print("x3_x --> ", x3_x.size())
        print("x4_x --> ", x4_x.size())
        print("x4_ --> ", x4_.size())

        # change to four dimensions    
 
        x4_d = torch.transpose(x4_, 1, 2).flatten(start_dim=2)
        print("x4_d --> ", x4_d.size())
 
        mean = self.mean_vec(x4_d)
 
        B, depth, D = mean.size()
        identity = torch.eye(D).unsqueeze(0).expand(B * depth ,-1, -1).cuda() * 10.0
 
        covar_half_vec = self.covar_half(x4_d)
        covar_half_vec = covar_half_vec.view(-1, D).unsqueeze(2)
 
        covar_half = torch.bmm(covar_half_vec, covar_half_vec.transpose(1, 2)) + identity
         
        covar = torch.bmm(covar_half, covar_half)

        Z = self.reparameterize(mean,  covar_half, M)

        Z_ = self.transform(Z).view(B * M, x4_.size()[-4], x4_.size()[-3], x4_.size()[-2], x4_.size()[-1])
  
        # change to five dimensions
        x4_x_size = [B * M, x4_x.size()[-4], x4_x.size()[-3], x4_x.size()[-2], x4_x.size()[-1]]
          
        x4_0_v =  self.conv_up(Z_, output_size=x4_x_size)
        print("x4_0_v --> ", x4_0_v.size())


        x3_1x = self.conv3_1_x(torch.cat([x3_x.repeat(M, 1, 1, 1, 1), self.up(x4_0_v)], 1))
        x2_2x = self.conv2_2_x(torch.cat([x2_x.repeat(M, 1, 1, 1, 1), self.up(x3_1x)], 1))
        x1_3x = self.conv1_3_x(torch.cat([x1_x.repeat(M, 1, 1, 1, 1), self.up(x2_2x)], 1))
        x0_4x = self.conv0_4_x(torch.cat([x0_x.repeat(M, 1, 1, 1, 1), self.up(x1_3x)], 1))

        print("x3_1x --> ", x3_1x.size())
        print("x2_2x --> ", x2_2x.size())
        print("x1_3x --> ", x1_3x.size())
        print("x0_4x --> ", x0_4x.size())

        x_ori = self.tanh(self.conv00(x0_4x))
  

        x0_0 = self.conv0_0(input)

        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        x4_0_ = self.conv_down_(x4_0)

        print("x0_0 --> ", x0_0.size())
        print("x1_0 --> ", x1_0.size())
        print("x2_0 --> ", x2_0.size())
        print("x3_0 --> ", x3_0.size())
        print("x4_0_ --> ", x4_0_.size())


        Z_f =  self.x_ori_vec(torch.transpose(x4_0_, 1, 2).flatten(start_dim=2))
        Z_f_ = self.transform_(Z_f).view(B, x4_0_.size()[-4], x4_0_.size()[-3], x4_0_.size()[-2], x4_0_.size()[-1]).repeat(M, 1, 1, 1, 1)
         
        x4_0_size = [B * M, x4_0.size()[-4], x4_0.size()[-3], x4_0.size()[-2], x4_0.size()[-1]]
        x4_0_z =  self.conv_up_(Z_f_, output_size=x4_0_size)

        x4_0 = torch.cat([x4_0_z, x4_0_v], 1)
        x3_1 = self.conv3_1(torch.cat([x3_0.repeat(M, 1, 1, 1, 1), self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0.repeat(M, 1, 1, 1, 1), self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0.repeat(M, 1, 1, 1, 1), self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0.repeat(M, 1, 1, 1, 1), self.up(x1_3)], 1))

        print("x4_0_z --> ", x4_0_z.size())
        print("x4_0 --> ", x4_0.size())
        print("x3_1 --> ", x3_1.size())
        print("x2_2 --> ", x2_2.size())
        print("x1_3 --> ", x1_3.size())
        print("x0_4 --> ", x0_4.size())


        output = self.final(x0_4)
        print("output --> ", output.size())
 
        return output, mean, covar.view(-1, depth, D, D), x_ori, Z


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.in_norm = nn.InstanceNorm3d(out_channels)  # Instance Normalization
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.in_norm(x)  # Apply Instance Normalization
        x = self.relu(x)
        return x



class LRL_imp(nn.Module):
      def __init__(self, num_classes, input_channels=3, resize_w = 128, resize_h = 128, **kwargs):
        super().__init__()

        self.nb_filter = [64, 128, 256, 512, 512]

        self.pool = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.up = nn.Upsample(scale_factor=(1.0, 2.0, 2.0), mode='trilinear', align_corners=True)
        self.relu = nn.ReLU(inplace=True)

        w = int(resize_w/32)
        h = int(resize_h/32)

        #newly implemented
        self.conv0_x = ConvBlock(input_channels, self.nb_filter[0], kernel_size = (3, 3, 3))
        self.conv1_x = ConvBlock(self.nb_filter[0], self.nb_filter[1], kernel_size = (3, 3, 3))
        self.conv2_x = ConvBlock(self.nb_filter[1], self.nb_filter[2], kernel_size = (3, 3, 3))
        self.conv3_x = ConvBlock(self.nb_filter[2], self.nb_filter[3], kernel_size = (3, 3, 3))
        self.conv4_x = ConvBlock(self.nb_filter[3], self.nb_filter[4], kernel_size = (3, 3, 3))
      



        self.conv_down = nn.Conv3d(self.nb_filter[4], self.nb_filter[4], (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.conv_down_ = nn.Conv3d(self.nb_filter[4], self.nb_filter[4], (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.mean_vec = nn.Linear(self.nb_filter[4]*w*h, int(self.nb_filter[3]/2)) 

        self.covar_half = nn.Linear(self.nb_filter[4]*w*h, int(self.nb_filter[3]/2))
        self.covar_diag = nn.Linear(self.nb_filter[4]*w*h, int(self.nb_filter[3]/2))

        self.x_ori_vec = nn.Linear(self.nb_filter[4]*w*h, int(self.nb_filter[3]/2))

        self.transform = nn.Linear(int(self.nb_filter[3]/2), self.nb_filter[4]*w*h)
        self.transform_ = nn.Linear(int(self.nb_filter[3]/2), self.nb_filter[4]*w*h)

        self.conv_up = nn.ConvTranspose3d(self.nb_filter[4], self.nb_filter[4], (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.conv_up_ = nn.ConvTranspose3d(self.nb_filter[4], self.nb_filter[4], (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))


        #newly implemented       
        self.conv3_1_x = ConvBlock(self.nb_filter[3]+self.nb_filter[4], self.nb_filter[3], kernel_size = (3, 3, 3))
        self.conv2_2_x = ConvBlock(self.nb_filter[2]+self.nb_filter[3], self.nb_filter[2], kernel_size = (3, 3, 3))
        self.conv1_3_x = ConvBlock(self.nb_filter[1]+self.nb_filter[2], self.nb_filter[1], kernel_size = (3, 3, 3))
        self.conv0_4_x = ConvBlock(self.nb_filter[0]+self.nb_filter[1], self.nb_filter[0], kernel_size = (3, 3, 3))

        
        self.conv00 = nn.Conv3d(self.nb_filter[0], input_channels, 3, stride=1, padding=1)


        self.tanh = nn.Tanh()
       
        #newly implemented
        self.conv0_0 = ConvBlock(input_channels, self.nb_filter[0], kernel_size = (3, 3, 3))
        self.conv1_0 = ConvBlock(self.nb_filter[0], self.nb_filter[1], kernel_size = (3, 3, 3))
        self.conv2_0 = ConvBlock(self.nb_filter[1], self.nb_filter[1], kernel_size = (3, 3, 3))
        self.conv3_0 = ConvBlock(self.nb_filter[2], self.nb_filter[3], kernel_size = (3, 3, 3))
        self.conv4_0 = ConvBlock(self.nb_filter[3], self.nb_filter[4], kernel_size = (3, 3, 3))

        #newly implemented
        self.conv3_1 = ConvBlock(self.nb_filter[3]+self.nb_filter[4], self.nb_filter[3], kernel_size = (3, 3, 3))
        self.conv2_2 = ConvBlock(self.nb_filter[2]+self.nb_filter[3], self.nb_filter[2], kernel_size = (3, 3, 3))
        self.conv1_3 = ConvBlock(self.nb_filter[1]+self.nb_filter[2], self.nb_filter[1], kernel_size = (3, 3, 3))
        self.conv0_4 = ConvBlock(self.nb_filter[0]+self.nb_filter[1], self.nb_filter[0], kernel_size = (3, 3, 3))

        self.final = nn.Conv3d(self.nb_filter[0], num_classes, kernel_size=(1, 1, 1))
       
        #newly implemented
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

      #newly implemented
      def reparameterize(self, mu, covar_half, M=1):
        B, depth, D = mu.size()

        if M == 1:
            eps = torch.randn(B, depth, D).cuda()
            eps_ = torch.bmm(eps.view(B*depth, 1, D), covar_half)
            return eps_.view(B, depth, D) + mu
        else:
            eps = torch.randn(M, B, depth, D).cuda()
            eps_ = torch.matmul(eps.view(M*B*depth, 1, D), covar_half.repeat(M*B, 1, 1))
            return eps_.view(M, B, depth, D) + mu.unsqueeze(0).repeat(M, 1, 1, 1)



      def forward(self, input, M=1):
        x0_x = self.conv0_x(input)
        x1_x = self.conv1_x(self.pool(x0_x))
        x2_x = self.conv2_x(self.pool(x1_x))
        x3_x = self.conv3_x(self.pool(x2_x))
        x4_x = self.conv4_x(self.pool(x3_x))
        x4_ = self.conv_down(x4_x)

        # change to four dimensions    
 
        x4_d = torch.transpose(x4_, 1, 2).flatten(start_dim=2)
 
        mean = self.mean_vec(x4_d)
 
        B, depth, D = mean.size()
        identity = torch.eye(D).unsqueeze(0).expand(B * depth ,-1, -1).cuda() * 10.0
        covar_half_vec = self.covar_half(x4_d)
        covar_half_vec = covar_half_vec.view(-1, D).unsqueeze(2)
 
        covar_half = torch.bmm(covar_half_vec, covar_half_vec.transpose(1, 2)) + identity
         
        covar = torch.bmm(covar_half, covar_half)

        Z = self.reparameterize(mean,  covar_half, M)

        Z_ = self.transform(Z).view(B * M, x4_.size()[-4], x4_.size()[-3], x4_.size()[-2], x4_.size()[-1])
  
        # change to five dimensions
        x4_x_size = [B * M, x4_x.size()[-4], x4_x.size()[-3], x4_x.size()[-2], x4_x.size()[-1]]
          
        x4_0_v =  self.conv_up(Z_, output_size=x4_x_size)

        x3_1x = self.conv3_1_x(torch.cat([x3_x.repeat(M, 1, 1, 1, 1), self.up(x4_0_v)], 1))
        x2_2x = self.conv2_2_x(torch.cat([x2_x.repeat(M, 1, 1, 1, 1), self.up(x3_1x)], 1))
        x1_3x = self.conv1_3_x(torch.cat([x1_x.repeat(M, 1, 1, 1, 1), self.up(x2_2x)], 1))
        x0_4x = self.conv0_4_x(torch.cat([x0_x.repeat(M, 1, 1, 1, 1), self.up(x1_3x)], 1))


        x_ori = self.tanh(self.conv00(x0_4x))
  

        x0_0 = self.conv0_0(input)

        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        x4_0_ = self.conv_down_(x4_0)


        Z_f =  self.x_ori_vec(torch.transpose(x4_0_, 1, 2).flatten(start_dim=2))
        Z_f_ = self.transform_(Z_f).view(B, x4_0_.size()[-4], x4_0_.size()[-3], x4_0_.size()[-2], x4_0_.size()[-1]).repeat(M, 1, 1, 1, 1)
         
        x4_0_size = [B * M, x4_0.size()[-4], x4_0.size()[-3], x4_0.size()[-2], x4_0.size()[-1]]
        x4_0_z =  self.conv_up_(Z_f_, output_size=x4_0_size)


        # this is where they concatenate the latent representation from the upper to lower
        x4_0 = torch.cat([x4_0_z, x4_0_v], 1)
        x3_1 = self.conv3_1(torch.cat([x3_0.repeat(M, 1, 1, 1, 1), self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0.repeat(M, 1, 1, 1, 1), self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0.repeat(M, 1, 1, 1, 1), self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0.repeat(M, 1, 1, 1, 1), self.up(x1_3)], 1))

        output = self.final(x0_4)
 
        return output, mean, covar.view(-1, depth, D, D), x_ori, Z


class LRL_lite(nn.Module):
  def __init__(self, num_classes, input_channels=3, resize_w = 128, resize_h = 128, **kwargs):
    super().__init__()

    # 32 slices, so each height and width is divided by 32
    w = int(resize_w/32)
    h = int(resize_h/32)
    
    # self.nb_filters = [64 --> 0, 128 --> 1, 256 --> 2, 512 --> 3, 512 --> 4]
    self.up = nn.Upsample(scale_factor=(2.0, 2.0, 2.0))
    self.conv0_x = ConvBlock(input_channels, 64, kernel_size = (3, 3, 3), stride = (1, 1, 1), padding = (1, 1, 1))
    self.conv1_x = ConvBlock(64, 128, kernel_size = (3, 3, 3), stride = (1, 1, 1), padding = (1, 1, 1))
    self.conv2_x = ConvBlock(128, 256, kernel_size = (3, 3, 3), stride = (1, 1, 1), padding = (1, 1, 1))
    self.conv3_x = ConvBlock(256, 512, kernel_size = (3, 3, 3), stride = (1, 1, 1), padding = (1, 1, 1))
    self.conv4_x = ConvBlock(512, 512, kernel_size = (3, 3, 3), stride = (1, 1, 1), padding = (1, 1, 1))

    self.mean_vec = nn.Linear(512*w*h, 256)

    #use covar_half wherever covar_half and ori_vec is used
    self.covar_half = nn.Linear(512*w*h, 256)

    #upscaling --> opposite of covar_half 
    self.transform = nn.Linear(256, 512*w*h)

    self.conv_down = nn.Conv3d(512, 512, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
    self.conv_up = nn.ConvTranspose3d(512, 512, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

    self.conv3_1_x = ConvBlock(512 + 512, 512, kernel_size = (3, 3, 3), stride = (1, 1, 1), padding = (1, 1, 1))
    self.conv2_2_x = ConvBlock(256 + 512, 256, kernel_size = (3, 3, 3), stride = (1, 1, 1), padding = (1, 1, 1))
    self.conv1_3_x = ConvBlock(128 + 256, 128, kernel_size = (3, 3, 3), stride = (1, 1, 1), padding = (1, 1, 1))
    self.conv0_4_x = ConvBlock(64 + 128, 64, kernel_size = (3, 3, 3), stride = (1, 1, 1), padding = (1, 1, 1))

    self.conv00 = nn.Conv3d(64, input_channels, kernel_size = (3, 3, 3), stride = (1, 1, 1), padding = (1, 1, 1))
    self.tanh = nn.Tanh()
    self.pool = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))

    self.conv0_0 = ConvBlock(input_channels, 64, kernel_size = (3, 3, 3), stride = (1, 1, 1), padding = (1, 1, 1))
    self.conv1_0 = ConvBlock(64, 128, kernel_size = (3, 3, 3), stride = (1, 1, 1), padding = (1, 1, 1))
    self.conv2_0 = ConvBlock(128, 256, kernel_size = (3, 3, 3), stride = (1, 1, 1), padding = (1, 1, 1))
    self.conv3_0 = ConvBlock(256, 512, kernel_size = (3, 3, 3), stride = (1, 1, 1), padding = (1, 1, 1))
    self.conv4_0 = ConvBlock(512, 512, kernel_size = (3, 3, 3), stride = (1, 1, 1), padding = (1, 1, 1))

    self.conv3_1 = ConvBlock(512 + 512, 512, kernel_size = (3, 3, 3), stride = (1, 1, 1), padding = (1, 1, 1))
    self.conv2_2 = ConvBlock(256 + 512, 256, kernel_size = (3, 3, 3), stride = (1, 1, 1), padding = (1, 1, 1))
    self.conv1_3 = ConvBlock(128 + 256, 128, kernel_size = (3, 3, 3), stride = (1, 1, 1), padding = (1, 1, 1))
    self.conv0_4 = ConvBlock(64 + 128, 64, kernel_size = (3, 3, 3), stride = (1, 1, 1), padding = (1, 1, 1))

    self.final = nn.Conv3d(64, num_classes, kernel_size=1)

    for m in self.modules():
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)

  def reparameterize(self, mu, covar_half):
    B, depth, D = mu.size()
    eps = torch.randn(B, depth, D).cuda()
    eps_ = torch.bmm(eps.view(B*depth, 1, D), covar_half)
    return eps_.view(B, depth, D) + mu


  def forward(self, input):
    # x0_x = self.conv0_x(input)
    # x1_x = self.pool(self.conv1_x(x0_x))
    # x2_x = self.pool(self.conv2_x(x1_x))
    # x3_x = self.pool(self.conv3_x(x2_x))
    # x4_x = self.pool(self.conv4_x(x3_x))
    # x4_ = self.conv_down(x4_x)

    x0_x = self.conv0_x(input)
    x1_x = self.conv1_x(self.pool(x0_x))
    x2_x = self.conv2_x(self.pool(x1_x))
    x3_x = self.conv3_x(self.pool(x2_x))
    x4_x = self.conv4_x(self.pool(x3_x))
    x4_ = self.conv_down(x4_x)

    print("x0_x --> ", x0_x.size())
    print("x1_x --> ", x1_x.size())
    print("x2_x --> ", x2_x.size())
    print("x3_x --> ", x3_x.size())
    print("x4_x --> ", x4_x.size())
    print("x4_ --> ", x4_.size())

    x4_d = x4_.transpose(1, 2).flatten(start_dim=2)
    print("x4_d --> ", x4_d.size())

    mean = self.mean_vec(x4_d)
    B, depth, D = mean.size()
    identity = torch.eye(D).unsqueeze(0).expand(B * depth ,-1, -1).cuda() * 10.0

    covar_half_vec = self.covar_half(x4_d)
    covar_half_vec = covar_half_vec.view(-1, D).unsqueeze(2)

    covar_half = torch.bmm(covar_half_vec, covar_half_vec.transpose(1, 2)) + identity
      
    covar = torch.bmm(covar_half, covar_half)

    Z = self.reparameterize(mean,  covar_half)

    Z_ = self.transform(Z).view(B, x4_.size()[-4], x4_.size()[-3], x4_.size()[-2], x4_.size()[-1])

    # change to five dimensions
    x4_x_size = [B, x4_x.size()[-4], x4_x.size()[-3], x4_x.size()[-2], x4_x.size()[-1]]
      
    x4_0_v =  self.conv_up(Z_, output_size=x4_x_size)
    print("x4_0_v --> ", x4_0_v.size())
    #upper decoder
    x3_1x = self.conv3_1_x(torch.cat([x3_x, self.up(x4_0_v)], 1))
    x2_2x = self.conv2_2_x(torch.cat([x2_x, self.up(x3_1x)], 1))
    x1_3x = self.conv1_3_x(torch.cat([x1_x, self.up(x2_2x)], 1))
    x0_4x = self.conv0_4_x(torch.cat([x0_x, self.up(x1_3x)], 1))
    print("x3_1x --> ", x3_1x.size())
    print("x2_2x --> ", x2_2x.size())
    print("x1_3x --> ", x1_3x.size())
    print("x0_4x --> ", x0_4x.size())



    x_ori = self.tanh(self.conv00(x0_4x))

    #lower encoder
    x0_0 = self.conv0_0(input)

    x1_0 = self.conv1_0(self.pool(x0_0))
    x2_0 = self.conv2_0(self.pool(x1_0))
    x3_0 = self.conv3_0(self.pool(x2_0))
    x4_0 = self.conv4_0(self.pool(x3_0))
    x4_0_ = self.conv_down(x4_0)

    print("x0_0 --> ", x0_0.size())
    print("x1_0 --> ", x1_0.size())
    print("x2_0 --> ", x2_0.size())
    print("x3_0 --> ", x3_0.size())
    print("x4_0_ --> ", x4_0_.size())


    Z_f =  self.covar_half(torch.transpose(x4_0_, 1, 2).flatten(start_dim=2))
    Z_f_ = self.transform(Z_f).view(B, x4_0_.size()[-4], x4_0_.size()[-3], x4_0_.size()[-2], x4_0_.size()[-1])
      
    x4_0_size = [B, x4_0.size()[-4], x4_0.size()[-3], x4_0.size()[-2], x4_0.size()[-1]]
    x4_0_z =  self.conv_up(Z_f_, output_size=x4_0_size)

    #lower decoder  
    # this is where they sum the representations from the upper encoder and LR to the lower decoder


    # x3_1 = nn.AdaptiveAvgPool3d(x2_2.shape[2:])(x3_1)
    # x4 = nn.AdaptiveAvgPool3d(x2_2.shape[2:])(x4)

    x4_0 = torch.cat([x4_0_z, x4_0_v], 1)
    x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
    x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
    x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
    x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

    print("x4_0_z --> ", x4_0_z.size())
    print("x4_0 --> ", x4_0.size())
    print("x3_1 --> ", x3_1.size())
    print("x2_2 --> ", x2_2.size())
    print("x1_3 --> ", x1_3.size())
    print("x0_4 --> ", x0_4.size())

    output = self.final(x0_4)
    print("output --> ". output.size())


    #output = output of the lower decoder
    #mean = mean of the latent representation of the upper autoencoder
    #covar = covariance of the latent representation of the upper autoencoder
    #x_ori = output of the upper decoder
    #Z = latent representation of the upper autoencoder
    return output, mean, covar.view(-1, depth, D, D), x_ori, Z

class LRL_new(nn.Module):

    def __init__(self, num_classes, input_channels=3, resize_w=128, resize_h=128, **kwargs):
      super().__init__()
      self.nb_filter = [64, 128, 256, 512, 512]
      self.pool = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
      self.up = nn.Upsample(scale_factor=(1.0, 2.0, 2.0))
      w, h = resize_w // 32, resize_h // 32

      def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
      self.conv0_x = conv_block(input_channels, self.nb_filter[0])
      self.conv1_x = nn.Sequential(conv_block(self.nb_filter[0], self.nb_filter[1]))
      self.conv2_x = nn.Sequential(conv_block(self.nb_filter[1], self.nb_filter[2]))
      self.conv3_x = nn.Sequential(conv_block(self.nb_filter[2], self.nb_filter[3]))
      self.conv4_x = nn.Sequential(conv_block(self.nb_filter[3], self.nb_filter[4]))
      self.conv_down = nn.Conv3d(self.nb_filter[4], self.nb_filter[4], (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
      self.conv_down_ = nn.Conv3d(self.nb_filter[4], self.nb_filter[4], (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
      self.mean_vec = nn.Linear(self.nb_filter[4] * w * h, self.nb_filter[3] // 2)
      self.covar_half = nn.Linear(self.nb_filter[4] * w * h, self.nb_filter[3] // 2)
      self.x_ori_vec = nn.Linear(self.nb_filter[4] * w * h, self.nb_filter[3] // 2)
      self.transform = nn.Linear(self.nb_filter[3] // 2, self.nb_filter[4] * w * h)
      self.transform_ = nn.Linear(self.nb_filter[3] // 2, self.nb_filter[4] * w * h)
      self.conv_up = nn.ConvTranspose3d(self.nb_filter[4], self.nb_filter[4], (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
      self.conv_up_ = nn.ConvTranspose3d(self.nb_filter[4], self.nb_filter[4], (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
      self.conv3_1_x = conv_block(self.nb_filter[3] + self.nb_filter[4], self.nb_filter[3])
      self.conv2_2_x = conv_block(self.nb_filter[2] + self.nb_filter[3], self.nb_filter[2])
      self.conv1_3_x = conv_block(self.nb_filter[1] + self.nb_filter[2], self.nb_filter[1])
      self.conv0_4_x = conv_block(self.nb_filter[0] + self.nb_filter[1], self.nb_filter[0])
      self.conv00 = nn.Conv3d(self.nb_filter[0], input_channels, 3, stride=1, padding=1)
      self.tanh = nn.Tanh()
      self.conv0_0 = conv_block(input_channels, self.nb_filter[0])
      self.conv1_0 = nn.Sequential(conv_block(self.nb_filter[0], self.nb_filter[1]))
      self.conv2_0 = nn.Sequential(conv_block(self.nb_filter[1], self.nb_filter[2]))
      self.conv3_0 = nn.Sequential(conv_block(self.nb_filter[2], self.nb_filter[3]))
      self.conv4_0 = nn.Sequential(conv_block(self.nb_filter[3], self.nb_filter[4]))
      self.conv3_1 = conv_block(self.nb_filter[3] + self.nb_filter[4] + self.nb_filter[4], self.nb_filter[3])
      self.conv2_2 = conv_block(self.nb_filter[2] + self.nb_filter[3], self.nb_filter[2])
      self.conv1_3 = conv_block(self.nb_filter[1] + self.nb_filter[2], self.nb_filter[1])
      self.conv0_4 = conv_block(self.nb_filter[0] + self.nb_filter[1], self.nb_filter[0])
      self.final = nn.Conv3d(self.nb_filter[0], num_classes, kernel_size=1)
      for m in self.modules():
          if isinstance(m, (nn.Conv2d, nn.Conv3d)):
              init.kaiming_normal_(m.weight)
              if m.bias is not None:
                  init.zeros_(m.bias)
    def reparameterize(self, mu, covar_half):
      B, depth, D = mu.size()
      eps = torch.randn(B, depth, D).cuda()
      eps_ = torch.bmm(eps.view(B*depth, 1, D), covar_half)
      return eps_.view(B, depth, D) + mu


    # def forward(self, input):
    #   x0_x = self.conv0_x(input)
    #   x1_x = self.conv1_x(self.pool(x0_x))
    #   x2_x = self.conv2_x(self.pool(x1_x))
    #   x3_x = self.conv3_x(self.pool(x2_x))
    #   x4_x = self.conv4_x(self.pool(x3_x))
    #   x4_ = self.conv_down(x3_x)
    #   x4_d = torch.transpose(x4_, 1, 2).flatten(start_dim=2)
    #   mean = self.mean_vec(x4_d)
    #   B, depth, D = mean.size()
    #   identity = torch.eye(D).unsqueeze(0).expand(B * depth, -1, -1).cuda() * 10.0
    #   covar_half_vec = self.covar_half(x4_d).view(-1, D).unsqueeze(2)
    #   covar_half = torch.bmm(covar_half_vec, covar_half_vec.transpose(1, 2)) + identity
    #   covar = torch.bmm(covar_half, covar_half)
    #   Z = self.reparameterize(mean, covar_half)
    #   Z_ = self.transform(Z).view(B, x4_.size()[-4], x4_.size()[-3], x4_.size()[-2], x4_.size()[-1])
    #   x4_x_size = [B, x4_x.size()[-4], x4_x.size()[-3], x4_x.size()[-2], x4_x.size()[-1]]
    #   x4_0_v = self.conv_up(Z_, output_size=x4_x_size)
    #   x3_1x = self.conv3_1_x(torch.cat([x3_x, self.up(x4_0_v)], 1))
    #   x2_2x = self.conv2_2_x(torch.cat([x2_x, self.up(x3_1x)], 1))
    #   x1_3x = self.conv1_3_x(torch.cat([x1_x, self.up(x2_2x)], 1))
    #   x0_4x = self.conv0_4_x(torch.cat([x0_x, self.up(x1_3x)], 1))
    #   x_ori = self.tanh(self.conv00(x0_4x))
    #   x0_0, x1_0 = self.conv0_0(input), self.conv1_0(self.pool(x0_0))
    #   x2_0, x3_0 = self.conv2_0(self.pool(x1_0)), self.conv3_0(self.pool(x2_0))
    #   x4_0, x4_0_ = self.conv4_0(self.pool(x3_0)), self.conv_down_(x3_0)
    #   Z_f = self.x_ori_vec(torch.transpose(x4_0_, 1, 2).flatten(start_dim=2))
    #   Z_f_ = self.transform_(Z_f).view(B, x4_0_.size()[-4], x4_0_.size()[-3], x4_0_.size()[-2], x4_0_.size()[-1])
    #   x4_0_size = [B, x4_0.size()[-4], x4_0.size()[-3], x4_0.size()[-2], x4_0.size()[-1]]
    #   x4_0_z = self.conv_up_(Z_f_, output_size=x4_0_size)
    #   x4_0 = torch.cat([x4_0_z, x4_0_v], 1)
    #   x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
    #   x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
    #   x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
    #   x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))
    #   output = self.final(x0_4)
    #   print(output.size())
    #   print(covar.view(-1, depth, D, D))
    #   print(x_ori.size())
    #   print(z.size())

    #   return output, mean, covar.view(-1, depth, D, D), x_ori, Z


    def forward(self, input):
      x0_x = self.conv0_x(input)
      x1_x = self.conv1_x(self.pool(x0_x))
      x2_x = self.conv2_x(self.pool(x1_x))
      x3_x = self.conv3_x(self.pool(x2_x))
      x4_x = self.conv4_x(self.pool(x3_x))
      x4_ = self.conv_down(x4_x)
      x4_d = x4_.transpose(1, 2).flatten(start_dim=2)
      mean = self.mean_vec(x4_d)
      B, depth, D = mean.size()
      identity = torch.eye(D).unsqueeze(0).expand(B * depth ,-1, -1).cuda() * 10.0
      covar_half_vec = self.covar_half(x4_d)
      covar_half_vec = covar_half_vec.view(-1, D).unsqueeze(2)
      covar_half = torch.bmm(covar_half_vec, covar_half_vec.transpose(1, 2)) + identity        
      covar = torch.bmm(covar_half, covar_half)
      Z = self.reparameterize(mean,  covar_half)
      Z_ = self.transform(Z).view(B, x4_.size()[-4], x4_.size()[-3], x4_.size()[-2], x4_.size()[-1])
      x4_x_size = [B, x4_x.size()[-4], x4_x.size()[-3], x4_x.size()[-2], x4_x.size()[-1]]       
      x4_0_v =  self.conv_up(Z_, output_size=x4_x_size)

      #upper decoder
      x3_1x = self.conv3_1_x(torch.cat([x3_x, self.up(x4_0_v)], 1))
      x2_2x = self.conv2_2_x(torch.cat([x2_x, self.up(x3_1x)], 1))
      x1_3x = self.conv1_3_x(torch.cat([x1_x, self.up(x2_2x)], 1))
      x0_4x = self.conv0_4_x(torch.cat([x0_x, self.up(x1_3x)], 1))
      x_ori = self.tanh(self.conv00(x0_4x))

      #lower encoder
      x0_0 = self.conv0_0(input)
      x1_0 = self.conv1_0(self.pool(x0_0))
      x2_0 = self.conv2_0(self.pool(x1_0))
      x3_0 = self.conv3_0(self.pool(x2_0))
      x4_0 = self.conv4_0(self.pool(x3_0))
      x4_0_ = self.conv_down(x4_0)
      Z_f =  self.covar_half(torch.transpose(x4_0_, 1, 2).flatten(start_dim=2))
      Z_f_ = self.transform(Z_f).view(B, x4_0_.size()[-4], x4_0_.size()[-3], x4_0_.size()[-2], x4_0_.size()[-1])        
      x4_0_size = [B, x4_0.size()[-4], x4_0.size()[-3], x4_0.size()[-2], x4_0.size()[-1]]
      x4_0_z =  self.conv_up(Z_f_, output_size=x4_0_size)
      x4_0 = torch.cat([x4_0_z, x4_0_v], 1)
      x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
      x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
      x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
      x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))
      output = self.final(x0_4)
      #output = output of the lower decoder
      #mean = mean of the latent representation of the upper autoencoder
      #covar = covariance of the latent representation of the upper autoencoder
      #x_ori = output of the upper decoder
      #Z = latent representation of the upper autoencoder
      return output, mean, covar.view(-1, depth, D, D), x_ori, Z
