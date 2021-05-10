import torch
import torch.nn as nn
import nn_ops
import utils

def _down_sample(x):
    return nn.functional.avg_pool2d(x, 2, 2)

def _increase_planes(x, n_out_planes):
    n_samples, n_planes, spatial_size1,spatial_size2  = x.size()
    x_zeros = torch.zeros(
    n_samples, n_out_planes - n_planes, spatial_size1, spatial_size2, 
    dtype=x.dtype, device=x.device)
    return torch.cat([x, x_zeros], 1)

def _downsample_and_increase_planes(x, n_out_planes):
    x = _down_sample(x)
    x = _increase_planes(x, n_out_planes)
    return x

def identity_func(n_in_planes, n_out_planes, stride):
    identity = lambda x: x
    if stride == 2 and n_in_planes != n_out_planes:
      identity = lambda x: _downsample_and_increase_planes(x, n_out_planes)
    elif stride == 2:
      identity = _down_sample
    elif n_in_planes != n_out_planes:
      identity = lambda x: _increase_planes(x, n_out_planes)
    return identity

class DecoderBlock(nn.Module):

  def __init__(self, n_in_planes, n_out_planes):
    super().__init__()
    self.block = nn.Sequential(
      nn_ops.deconv4x4(n_in_planes, n_out_planes, True),
      nn.BatchNorm2d(n_out_planes),
      nn.ReLU(inplace=True),
      nn_ops.conv3x3(n_out_planes, n_out_planes, 1, True),
      nn.BatchNorm2d(n_out_planes)
    )

    self.upsample = lambda x: nn.functional.upsample(
      x, scale_factor=2, mode='nearest')
    self.shortcut_conv = nn.Sequential()
    if n_in_planes != n_out_planes:
      self.shortcut_conv = nn.Sequential(
        nn.Conv2d(n_in_planes, n_out_planes, kernel_size=1),
        nn.BatchNorm2d(n_out_planes)
      )

  def forward(self, x):
    out = self.block(x)
    shortcut = self.shortcut_conv(x)
    shortcut = self.upsample(shortcut)

    out += shortcut
    out = nn.functional.relu(out)
    return out

class DecoderBlock4x(nn.Module):

  def __init__(self, n_in_planes, n_out_planes):
    super().__init__()
    self.block = nn.Sequential(
      nn_ops.deconv4x4(n_in_planes, n_out_planes, True),
      nn.BatchNorm2d(n_out_planes),
      nn.ReLU(inplace=True),
      nn_ops.conv3x3(n_out_planes, n_out_planes, 1, True),
      nn.BatchNorm2d(n_out_planes)
    )

    self.upsample = lambda x: nn.functional.upsample(
      x, scale_factor=4, mode='nearest')
    self.shortcut_conv = nn.Sequential()
    if n_in_planes != n_out_planes:
      self.shortcut_conv = nn.Sequential(
        nn.Conv2d(n_in_planes, n_out_planes, kernel_size=1),
        nn.BatchNorm2d(n_out_planes)
      )

  def forward(self, x):
    out = self.block(x)
    shortcut = self.shortcut_conv(x)
    shortcut = self.upsample(shortcut)

    out += shortcut
    out = nn.functional.relu(out)
    return out

class BasicBlock(nn.Module):

  expansion = 1

  def __init__(self, n_in_planes, n_out_planes, stride=1):
    super().__init__()
    assert stride == 1 or stride == 2

    self.block = nn.Sequential(
      nn_ops.conv3x3(n_in_planes, n_out_planes, stride),
      nn.BatchNorm2d(n_out_planes),
      nn.ReLU(inplace=True),
      nn_ops.conv3x3(n_out_planes, n_out_planes),
      nn.BatchNorm2d(n_out_planes)
    )

    self.identity = identity_func(n_in_planes, n_out_planes, stride)

  def forward(self, x):
    out = self.block(x)
    identity = self.identity(x)

    out += identity
    out = nn.functional.relu(out)
    return out


class celebA_Encoder(nn.Module):

  def __init__(self, d_latent, device='cuda', log_dir=''):
    super().__init__()
    self.d_latent = d_latent
    self.device = device
      
    n_blocks = [1, 1, 1, 1]
    mult = 8
    n_output_planes = [16 * mult, 32 * mult, 64 * mult, 128 * mult]
    self.n_in_planes = n_output_planes[0]
    
    self.layer0 = nn.Sequential(
      nn_ops.conv3x3(3, self.n_in_planes, 1),
      nn.BatchNorm2d(self.n_in_planes),
      nn.ReLU(inplace=True)
    )
    self.layer1 = self._make_layer(BasicBlock, n_blocks[0], n_output_planes[0], 2)
    self.layer2 = self._make_layer(BasicBlock, n_blocks[1], n_output_planes[1], 2)
    self.layer3 = self._make_layer(BasicBlock, n_blocks[2], n_output_planes[2], 2)
    self.layer4 = self._make_layer(BasicBlock, n_blocks[3], n_output_planes[3], 2)
    self.latent_mapping = nn.Sequential(
      nn.Linear(n_output_planes[3] * BasicBlock.expansion, d_latent, True),
      nn.BatchNorm1d(d_latent),
      nn.Tanh()
    )
    
    self.apply(nn_ops.variable_init)
    self.to(device)
    utils.model_info(self, 'celebA_encoder', log_dir)
  
  def _make_layer(self, block, n_blocks, n_out_planes, stride=1):
    layers = []
    layers.append(block(self.n_in_planes, n_out_planes, stride))
    self.n_in_planes = n_out_planes * block.expansion
    for i in range(1, n_blocks):
      layers.append(block(self.n_in_planes, n_out_planes))

    return nn.Sequential(*layers)
  
  def forward(self, x):
    x = self.layer0(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    # import ipdb;ipdb.set_trace()
    spatial_size = x.size(2)
    x = nn.functional.avg_pool2d(x, spatial_size, 1)
    x = x.view(x.size(0), -1)
    x = self.latent_mapping(x)
    return x
  


class celebA_Decoder(nn.Module):

  def __init__(self, d_latent, device='cuda', log_dir=''):
    super().__init__()

    self.d_latent = d_latent
    self.device = device

    self.mult = 8
    self.latent_mapping = nn.Sequential(
      nn.Linear(self.d_latent, 4 * 4 * 128 * self.mult),
      nn.BatchNorm1d(4 * 4 * 128 * self.mult),
      nn.ReLU()
    )
    self.block1 = DecoderBlock(128 * self.mult, 64 * self.mult)
    self.block2 = DecoderBlock(64 * self.mult, 32 * self.mult)
    self.block3 = DecoderBlock(32 * self.mult, 16 * self.mult)
    self.block4 = DecoderBlock(16 * self.mult, 8 * self.mult)
    self.block5 = DecoderBlock(8 * self.mult, 4 * self.mult)
    self.block6 = DecoderBlock(4 * self.mult, 2 * self.mult)
    self.output_conv = nn_ops.conv3x3(2 * self.mult, 3, 1, True)
    self.final_act = nn.Sigmoid()

    self.apply(nn_ops.variable_init)
    self.to(device)
    utils.model_info(self, 'celebA_decoder', log_dir)

  def forward(self, y):
    # import ipdb;ipdb.set_trace()
    x = self.latent_mapping(y)
    x = x.view(-1, 128 * self.mult, 4, 4)
    xC = self.block1(x)
    x = self.block2(xC)
    x = self.block3(x)
    xS = self.block4(x)
    x = self.block5(xS)
    x = self.block6(x)
    x = self.output_conv(x)
    x = self.final_act(x)
    return x,xC,xS


class Classifier(nn.Module):

    def __init__(self, d_latent, encoder_only=False, device='cuda',no_classes=5, log_dir=''):
        super().__init__()

        self.d_latent = d_latent
        self.device = device
        self.mult = 8
        self.encoder_only = encoder_only

        if not self.encoder_only:
          self.latent_mappingDec = nn.Sequential(
            nn.Linear(self.d_latent, 4 * 4 * 128 * self.mult),
            nn.BatchNorm1d(4 * 4 * 128 * self.mult),
            nn.ReLU()
          )
          self.block1 = DecoderBlock(128 * self.mult, 64 * self.mult)    

        # self.output_conv = nn_ops.conv3x3(64 * self.mult, 3, 1, True)
        # self.final_act = nn.Sigmoid()
      
        
        n_blocks = [1, 1, 1, 1]
        n_output_planes = [16 * self.mult, 32 * self.mult, 64 * self.mult, 128 * self.mult]
        self.layer0 = nn.Sequential(nn_ops.conv3x3(n_output_planes[2],n_output_planes[0], 1),nn.BatchNorm2d(n_output_planes[0]),nn.ReLU(inplace=True))
        self.n_in_planes = n_output_planes[0]
        # self.layer1 = self._make_layer(BasicBlock, n_blocks[0], n_output_planes[0], 2)
        self.latent_mappingEnc = nn.Sequential(nn.Linear(n_output_planes[0] * BasicBlock.expansion, int(d_latent/2), True),
            nn.BatchNorm1d(int(d_latent/2)),nn.Tanh())
        
        self.linear = nn.Sequential(nn.Linear(int(d_latent/2), no_classes, True),nn.Sigmoid())
        # self.linear = nn.Sequential(nn.Linear(d_latent, no_classes, True),nn.BatchNorm1d(d_latent),nn.Tanh())
        

        self.apply(nn_ops.variable_init)
        self.to(device)
        # utils.model_info(self, 'celebA_encoder', log_dir)
        

    def _make_layer(self, block, n_blocks, n_out_planes, stride=1):
        layers = []
        layers.append(block(self.n_in_planes, n_out_planes, stride))
        self.n_in_planes = n_out_planes * block.expansion
        for i in range(1, n_blocks):
          layers.append(block(self.n_in_planes, n_out_planes))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        if not self.encoder_only:
          x = self.latent_mappingDec(x)
          x = x.view(-1, 128 * self.mult, 4, 4)
          x = self.block1(x)
      
        # import ipdb;ipdb.set_trace()
        # x = self.output_conv(x)
        # x = self.final_act(x)

        x = self.layer0(x)
        # x = self.layer1(x)
        spatial_size = x.size(2)
        x = nn.functional.avg_pool2d(x, spatial_size, 1)
        x = x.view(x.size(0), -1)
        x = self.latent_mappingEnc(x)
        x = self.linear(x)
        return torch.sigmoid(x)

class Segment(nn.Module):

  def __init__(self, d_latent,no_classes=None, device='cuda', log_dir='',encoder_only=False,Tone=3):
    super().__init__()

    self.d_latent = d_latent
    self.no_classes = no_classes
    self.device = device
    self.encoder_only = encoder_only
    self.Tone = Tone

    self.mult = 8
    self.latent_mapping = nn.Sequential(
      nn.Linear(self.d_latent, 4 * 4 * 128 * self.mult),
      nn.BatchNorm1d(4 * 4 * 128 * self.mult),
      nn.ReLU()
    )
    self.block1 = DecoderBlock(128 * self.mult, 64 * self.mult)
    self.block2 = DecoderBlock(64 * self.mult, 32 * self.mult)
    self.block3 = DecoderBlock(32 * self.mult, 16 * self.mult)
    self.block4 = DecoderBlock(16 * self.mult, 8 * self.mult)
    self.block5 = DecoderBlock(8 * self.mult, 4 * self.mult)

    
    
    n_blocks = [1, 1, 1, 1]
    # n_output_planes = [32 * self.mult, 64 * self.mult, 128 * self.mult, 256 * self.mult]
    # n_output_planes = [16 * self.mult, 32 * self.mult, 64 * self.mult, 128 * self.mult]
    n_output_planes = [8 * self.mult, 16 * self.mult, 32 * self.mult, 64 * self.mult]
    n_last_plane = self.no_classes
    self.layer0 = nn.Sequential(nn_ops.conv3x3(n_output_planes[0],n_output_planes[1], 1),nn.BatchNorm2d(n_output_planes[1]),nn.ReLU(inplace=True))
    self.layer1 = nn.Sequential(nn_ops.conv3x3(n_output_planes[1],n_output_planes[0], 1),nn.BatchNorm2d(n_output_planes[0]),nn.ReLU(inplace=True))
    self.layer2 = nn.Sequential(nn_ops.conv3x3(n_output_planes[0],n_last_plane*3, 1),nn.BatchNorm2d(n_last_plane*3),nn.ReLU(inplace=True))



    # self.latent_mappingDev = nn.Sequential(
    #       nn.Linear(n_output_planes[3] * BasicBlock.expansion, d_latent, True),
    #       nn.BatchNorm1d(d_latent),
    #       nn.Tanh()
    #     )
            # self.layer1 = self._make_layer(BasicBlock, 1, n_output_planes[2], 2)#self._make_layer(BasicBlock, n_blocks[1], n_output_planes[1], 2)
    self.last_layerRGB = nn_ops.conv3x3(n_output_planes[0], n_last_plane*3, 1, True)
    self.final_actRGB = nn.Sigmoid()
    
    self.last_layerBW = nn_ops.conv3x3(n_output_planes[0], n_last_plane, 1, True)
    self.final_actBW = nn.Sigmoid()
  
    # self.layer2 = self._make_layer(BasicBlock, 1, n_output_planes[3], 1)#self._make_layer(BasicBlock, n_blocks[1], n_output_planes[1], 2)
    # # self.layer3 = self._make_layer(BasicBlock, n_blocks[2], n_output_planes[2], 2)
    # self.block3 = DecoderBlock(n_output_planes[2], 32 * self.mult)
    # # self.output_conv = nn_ops.conv3x3(2 * self.mult, 3, 1, True)

    # self.linear = nn.Sequential(nn.Linear(int(d_latent/2), no_classes, True),nn.Sigmoid())
    # self.linear = nn.Sequential(nn.Linear(d_latent, no_classes, True),nn.BatchNorm1d(d_latent),nn.Tanh())
    

    self.apply(nn_ops.variable_init)
    self.to(device)
    # utils.model_info(self, 'celebA_encoder', log_dir)
      
  def _make_layer(self, block, n_blocks, n_out_planes, stride=1):
      layers = []
      layers.append(block(self.n_in_planes, n_out_planes, stride))
      self.n_in_planes = n_out_planes * block.expansion
      for i in range(1, n_blocks):
        layers.append(block(self.n_in_planes, n_out_planes))

      return nn.Sequential(*layers)
      
  def forward(self, x):
      if not self.encoder_only:
        x = self.latent_mapping(x)
        x = x.view(-1, 128 * self.mult, 4, 4)
        xC = self.block1(x)
        x = self.block2(xC)
        x = self.block3(x)
        x = self.block4(x)
      # import ipdb;ipdb.set_trace()
      
      # x = self.block5(x)
      # x = self.output_conv(x)
      # x = self.final_act(x)

      x = self.layer0(x)
      xC = self.layer1(x)
      x = self.last_layerRGB(xC)
      window_size = x.size(3)
      x = x.view(-1,self.no_classes , 3, window_size, window_size)
      xRGB = self.final_actRGB(x)

      # import ipdb;ipdb.set_trace()
      # x = self.layerBW(x)
      x = self.last_layerBW(xC)
      window_sizeA = x.size(3)
      if window_size!=window_sizeA:
        print('Dick')

      x = x.view(-1,self.no_classes, window_sizeA, window_size)
      xBW = self.final_actBW(x)

      # x = nn.functional.avg_pool2d(x, spatial_size, 1)
      # x = x.view(x.size(0), -1)
      # x = self.latent_mappingDev(x)
      


      # spatial_size = x.size(2)
      # x = nn.functional.avg_pool2d(x, spatial_size, 1)
      # x = x.view(x.size(0), -1)
      # x = self.latent_mapping(x)
      # x = self.linear(x)
      if self.Tone == 1:
        return xBW
      elif self.Tone == 3:
        return xRGB
      elif self.Tone == 'all':
        return(xBW,xRGB)


if __name__ == '__main__':
  import torch
  
  Tone = 3
  asd = torch.zeros((32,256), requires_grad=True)
  classif1 = Classifier(256,device='cpu',no_classes=5,encoder_only=False)
  decoder1 = Segment(256,device='cpu',no_classes=5,encoder_only=False, Tone = Tone)
  res_classif = classif1(asd.cpu())
  res_decoder = decoder1(asd.cpu())
  # import ipdb;ipdb.set_trace()
  print('res_classif : {'+','.join([str(i) for i in res_classif.shape])+'}\nres_high : {'+','.join([str(i) for i in res_decoder.shape])+'}\n')

  dick = torch.zeros((32, 64, 64, 64), requires_grad=True)
  decoder2 = Segment(256,device='cpu',no_classes=5,encoder_only=True, Tone = Tone)
  asd = torch.zeros((32,256), requires_grad=True)
  res_decoderF = decoder2(dick.cpu())

  # print('}\nres_without_decoder_high : {'+','.join([str(i) for i in res_decoderF.shape])+'}\n')

  classif2 = Classifier(256,device='cpu',no_classes=5,encoder_only=True)
  fuckoff = torch.zeros((32, 512, 8, 8), requires_grad=True)
  res_classifF = classif2(fuckoff.cpu())

  print('res_without_decoder_classif : {'+','.join([str(i) for i in res_classifF.shape])+'}\nres_without_decoder_high : {'+','.join([str(i) for i in res_decoderF.shape])+'}\n')

  optimizer, lr_schedule = make_optimizer(decoder)
  loss_fn = get_loss_fn(cfg, logger)
  loss_fn2 = torch.nn.MSELoss()

  # 'LFW_TEST_LIST': '', 'METRIC': CfgNode({'IS_PEDCC': True, 'IS_MSE': True, 'S': 15.0, 'NAME': 'Softmax', 'N': 1, 'M': 0.5})

  # 'SOLVER': CfgNode({'CHECKPOINT_PERIOD': 10, 'MOMENTUM': 0.9, 'MAX_EPOCHS': 120, 'RESUME': 0, 'MILE_STONES': [25, 50, 80, 100], 'WEIGHT_DECAY': 0.0005, 'LOG_PERIOD': 100, 
  #   'OPTIMIZER_NAME': 'SGD', 'LR_SCHDULER': 'MultiStepLR', 'BASE_LR': 0.1, 'IMS_PER_BATCH': 256})
  
  def make_optimizer(model):
    for key, value in model.named_parameters():
        params = []
        for key, value in model.named_parameters():
            if not value.requires_grad:
                continue
            lr = 0.1
            weight_decay = 0.0005
            # if "bias" in key:
            #     lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            #     weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    # import ipdb;ipdb.set_trace()
    # params += [{"params": [value], "lr": 0.1, "weight_decay": weight_decay}]
    optimizer = getattr(torch.optim, 'SGD')(params, momentum=0.9)
    lr_schduler = getattr(torch.optim.lr_scheduler, 'MultiStepLR')(optimizer, [25, 50, 80, 100], gamma=0.1, last_epoch=-1)

    return optimizer, lr_schduler