
from turtle import st
import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        inner_channel = int(growth_rate)

        self.aspp1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inner_channel, kernel_size=1, bias=False),
        )
        self.aspp3 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inner_channel, kernel_size=3, padding='same', bias=False),
        )
        self.aspp9 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inner_channel, kernel_size=3, dilation=4, padding='same', bias=False),
        )
        self.aspp18 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inner_channel, kernel_size=3, dilation=9, padding='same', bias=False),
        )
        self.aftcat = nn.Sequential(
            nn.BatchNorm2d(inner_channel * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channel * 4, growth_rate, kernel_size=1, bias=False),
        )

    def forward(self, x):
        fmap1 = self.aspp1(x)
        fmap2 = self.aspp3(x)
        fmap3 = self.aspp9(x)
        fmap4 = self.aspp18(x)
        out = self.aftcat(torch.cat([fmap1, fmap2, fmap3, fmap4], dim=1))
        return torch.cat([x, out], dim=1)

class Transition(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.down_sample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.MaxPool2d(2, stride=2)
        )
    def forward(self, x):
        temp = self.down_sample(x)
        return temp


class BlockASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        inner_channel = int(out_channels//4)

        self.aspp1 = nn.Sequential(
            nn.BatchNorm2d(out_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels*2, inner_channel, kernel_size=1, bias=False),
        )
        self.aspp3 = nn.Sequential(
            nn.BatchNorm2d(out_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels*2, inner_channel, kernel_size=3, padding='same', bias=False),
        )
        self.aspp9 = nn.Sequential(
            nn.BatchNorm2d(out_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels*2, inner_channel, kernel_size=3, dilation=4, padding='same', bias=False),
        )
        self.aspp18 = nn.Sequential(
            nn.BatchNorm2d(out_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels*2, inner_channel, kernel_size=3, dilation=9, padding='same', bias=False),
        )
        self.aftcat = nn.Sequential(
            nn.BatchNorm2d(inner_channel * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channel * 4, out_channels, kernel_size=1, bias=False),
        )

    def forward(self, x):
        fmap1 = self.aspp1(x)
        fmap2 = self.aspp3(x)
        fmap3 = self.aspp9(x)
        fmap4 = self.aspp18(x)
        out = self.aftcat(torch.cat([fmap1, fmap2, fmap3, fmap4], dim=1))
        return out

class Encoder(nn.Module):
    def __init__(self, starting_channel, channels, growth_rate):
        super().__init__()
        self.channels = channels
        self.growth_rate = growth_rate
        self.starting_channel = starting_channel
        self.convInit = nn.Conv2d(4, starting_channel, 3, padding=1)
        self.enblocks = nn.ModuleList()
        self.transitions = nn.ModuleList()

        for channel_idx, numOfBlocks in enumerate(self.channels):
            big_block = self.make_big_denseBlock(self.starting_channel, numOfBlocks, channel_idx)
            self.enblocks.append(big_block)
            self.starting_channel += self.growth_rate * numOfBlocks
            
            out_channel = self.starting_channel // 2
            transition = Transition(self.starting_channel, out_channel)
            self.transitions.append(transition)
            self.starting_channel = out_channel
        
    #function for making big dense blocks // parameter blocks is int which indicate number of blocks in big block
    def make_big_denseBlock(self, starting_channel, blocks, layer_idx):
        st_channel = starting_channel 
        big_denseBlock = nn.Sequential()
        for i in range(blocks):
            big_denseBlock.add_module('{}:bottleneck'.format(i), Bottleneck(st_channel, self.growth_rate))
            st_channel += self.growth_rate
        return big_denseBlock

    def forward(self, x):
        x = self.convInit(x)

        blockOutputs = [] 
        for i in range(len(self.channels)-1):
            x = self.enblocks[i](x)
            blockOutputs.append(x)
            x = self.transitions[i](x)
        x = self.enblocks[-1](x)
        blockOutputs.append(x)
        # for d in blockOutputs: print(d.shape)

        return blockOutputs

class Decoder(nn.Module):
    def __init__(self, channels): #channels decrease by 1/2
        super().__init__()
        self.channels = channels
        self.convsTrans = nn.ModuleList([
            nn.ConvTranspose2d(channels[i], channels[i+1], 2, 2) for i in range(len(channels)-1)
        ])
        self.upconvs = nn.ModuleList([
            BlockASPP(channels[i], channels[i+1]) for i in range(len(channels)-1)
        ])
        self.convLast = nn.Conv2d(channels[-1], 1, 1)


    def forward(self, enFeatures): #enFeature는 순서가 거꾸로 된 상태에서 입력!
        x = enFeatures[0]
        for i in range(len(self.upconvs)):
            x = self.convsTrans[i](x) 
            x = torch.cat([x, enFeatures[i+1]], dim = 1)
            x = self.upconvs[i](x) #2 layer conv(channel halves)

        x = self.convLast(x)
        return x    
        
class DAUnet(nn.Module):
    def __init__(self, starting_channel, encChannels, decChannels, growth_rate):
        super().__init__()        
        self.encoder = Encoder(starting_channel, encChannels, growth_rate)
        self.decoder = Decoder(decChannels)

    def forward(self, x):
        enFeatures = self.encoder(x)
        deFeature = self.decoder(enFeatures[::-1])

        return deFeature
