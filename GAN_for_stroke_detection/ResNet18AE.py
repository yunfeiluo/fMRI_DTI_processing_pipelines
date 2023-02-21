from net_blocks import *

class Res18Encoder(nn.Module):
    def __init__(self, begin_channel, num_classes=0):
        super(Res18Encoder, self).__init__()
        self.classifier = num_classes == 0
        if self.classifier:
            print("Building ResNet18 Classifier...")

        self.clip_liner = nn.Sequential( 
            # nn.Conv3d(1, begin_channel, (4, 4, 4), stride=(1, 1, 1), padding=(12, 3, 12)),# if 91*109*91
            nn.Conv3d(1, begin_channel, (6, 6, 6), stride=(2, 2, 2), padding=(7, 5, 7)), # 182*218*182
            nn.BatchNorm3d(begin_channel),
            nn.ReLU()
        ) # 96*112*96

        self.pool = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2), return_indices=True) # 48*56*48

        self.res_conv_liner = nn.Sequential(
            ResBlock(begin_channel),
            nn.ReLU(),
            ResBlock(begin_channel), 
            nn.ReLU(), # 48*56*48
            ResTransBlock(begin_channel, begin_channel * 2),
            nn.ReLU(),
            ResBlock(begin_channel * 2),
            nn.ReLU(),# 24*28*24
            ResTransBlock(begin_channel * 2, begin_channel * 4),
            nn.ReLU(),
            ResBlock(begin_channel * 4),
            nn.ReLU(), # 12*14*12
            ResTransBlock(begin_channel * 4, begin_channel * 8),
            nn.ReLU(),
            ResBlock(begin_channel * 8),
            nn.ReLU() # 6*7*6
        ) 

        self.latent = ResBlock(begin_channel * 8)

        self.classify_liner, self.avgpool = None, None
        if not self.classifier:
            self.avgpool = nn.AvgPool3d((6, 7, 6), stride=(1, 1, 1)) # 1*1*1
            self.classify_liner = nn.Sequential(
                nn.Linear(begin_channel * 8, num_classes)
            )
    
    def forward(self, x):
        out, ind = self.pool(self.clip_liner(x))
        if not self.classifier:
            out = self.avgpool(self.res_conv_liner(out))
            return self.classify_liner(out.view(out.shape[0], -1))
        return self.latent(self.res_conv_liner(out)), ind

class Res18Decoder(nn.Module):
    def __init__(self, end_channel): # 91*109*91
        super(Res18Decoder, self).__init__()
        self.latent_liner = nn.Sequential(
            invResBlock(end_channel * 8),
            nn.ReLU()
        )

        self.deconv_res_liner = nn.Sequential(
            invResBlock(end_channel * 8),
            nn.ReLU(),
            invResTransBlock(end_channel * 8, end_channel * 4),
            nn.ReLU(),
            invResBlock(end_channel * 4), #
            nn.ReLU(),
            invResTransBlock(end_channel * 4, end_channel * 2),
            nn.ReLU(), #
            invResBlock(end_channel * 2),
            nn.ReLU(),
            invResTransBlock(end_channel * 2, end_channel),
            nn.ReLU(), #
            invResBlock(end_channel),
            nn.ReLU()
        )

        self.unpool = nn.MaxUnpool3d((2, 2, 2), stride=(2, 2, 2))

        self.clip_liner = nn.Sequential(
            # nn.ConvTranspose3d(end_channel, 1, (4, 4, 4), stride=(1, 1, 1), padding=(12, 3, 12)), # 91*109*91    
            nn.ConvTranspose3d(end_channel, 1, (6, 6, 6), stride=(2, 2, 2), padding=(7, 5, 7)), # 182*218*182
            # nn.BatchNorm3d(end_channel)
        )        
    
    def forward(self, x, ind):
        out = self.deconv_res_liner(self.latent_liner(x))
        out = self.unpool(out, ind)
        return self.clip_liner(out)

class Res18Autoencoder(nn.Module):
    def __init__(self, begin_channel):
        super(Res18Autoencoder, self).__init__()
        print('Building Autoencoder with ResNet18 structure...')
        self.encoder = Res18Encoder(begin_channel)
        self.decoder = Res18Decoder(begin_channel)

    def forward(self, x):
        encoder_out, ind = self.encoder(x)
        return encoder_out, self.decoder(encoder_out, ind)