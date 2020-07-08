def complexity_conv2d(cx, w_in, w_out, k, s, p, g=1, b=False):
    """A complexity of Conv2D

    # Arguments
        cx : Complexity
        w_in : Input channel
        w_out : Output channel
        k : Kernel size
        s : Stride
        p : Padding size
        b : Bias
    
    # Returns
        Calculated complexity
    """
    h, w, flops, params, acts = cx['h'], cx['w'], cx['flops'], cx['params'], cx['acts']
    h = (h + 2 * p - k) // s + 1
    w = (w + 2 * p - k) // s + 1
    flops += k * k * w_in * w_out * h * w // g
    params += k * k * w_in * w_out // g
    flops += w_out if b else 0
    params += w_out if b else 0
    acts += w_out * h * w
    return {"h": h, "w": w, "flops": flops, "params": params, "acts": acts}

def complexity_batchnorm2d(cx, w_in):
    """A complexity of BatchNorm2D

    # Arguments
        cx : Complexity
        w_in : Input channel
    
    # Returns
        Calculated complexity of BatchNorm2D
    """
    h, w, flops, params, acts = cx["h"], cx["w"], cx["flops"], cx["params"], cx["acts"]
    params += 2 * w_in
    return {"h": h, "w": w, "flops": flops, "params": params, "acts": acts}

def complexity_maxpool2d(cx, k, s, p):
    """A complexity of MaxPool2D

    # Arguments
        cx : Complexity
        w_in : Input channel
    
    # Returns
        Calculated complexity of MaxPool2D
    """
    h, w, flops, params, acts = cx["h"], cx["w"], cx["flops"], cx["params"], cx["acts"]
    h = (h + 2 * p - k) // s + 1
    w = (w + 2 * p - k) // s + 1
    return {"h": h, "w": w, "flops": flops, "params": params, "acts": acts}

def get_complexity(cfg):
    """Total complexity of given model

    # Arguments
        cfg : Configurations
    
    # Returns
        Calculated complexity of the model
    """

    ##########################
    # Stage functions for complexity
    ##########################
    def BasicBlock(cx, w_in, w_out, s, br, gw):
        """A complexity of basick block

        # Arguments
            cx : Complexity
            w_in : Input channel
            w_out : Output channel
            s : Stride
        
        # Returns
            Calculated complexity of basick block
        """

        cx = complexity_conv2d(cx, w_in=w_in, w_out=w_out, k=3, s=s, p=1, b=False)
        cx = complexity_batchnorm2d(cx, w_in=w_out)
        cx = complexity_conv2d(cx, w_in=w_out, w_out=w_out, k=3, s=1, p=1, b=False)
        cx = complexity_batchnorm2d(cx, w_in=w_out)
        return cx

    def ResBlock(cx, w_in, w_out, s, br, gw):
        """A complexity of resblock

        # Arguments
            cx : Complexity
            w_in : Input channel
            w_out : Output channel
            s : Stride
        
        # Returns
            Calculated complexity of resblock
        """

        # shorcut
        cx = complexity_conv2d(cx, w_in=w_in, w_out=w_out, k=1, s=s, p=0, b=False)
        cx = complexity_batchnorm2d(cx, w_in=w_out)

        # main
        cx = complexity_conv2d(cx, w_in=w_in, w_out=w_out, k=3, s=s, p=1, b=False)
        cx = complexity_batchnorm2d(cx, w_in=w_out)
        cx = complexity_conv2d(cx, w_in=w_in, w_out=w_out, k=3, s=1, p=1, b=False)
        cx = complexity_batchnorm2d(cx, w_in=w_out)
        return cx

    def ResBottleneckBlock(cx, w_in, w_out, s, br, gw):
        """A complexity of resbottleneckblock

        # Arguments
            cx : Complexity
            w_in : Input channel
            w_out : Output channel
            s : Stride
            br : Bottleneck ratio
            gw : Width of group convolution
        
        # Returns
            Calculated complexity of resbottleneckblock
        """

        # shortcut
        cx = complexity_conv2d(cx, w_in=w_in, w_out=w_out, k=1, s=s, p=0, b=False)
        cx = complexity_batchnorm2d(cx, w_in=w_out)

        # main
        cx = complexity_conv2d(cx, w_in=w_in, w_out=w_out//br, k=1, s=1, p=0, b=False)
        cx = complexity_batchnorm2d(cx, w_in=w_out//br)

        cx = complexity_conv2d(cx, w_in=w_out//br, w_out=w_out//br, k=3, s=s, p=1, g=gw, b=False)
        cx = complexity_batchnorm2d(cx, w_in=w_out//br)

        cx = complexity_conv2d(cx, w_in=w_out//br, w_out=w_out, k=1, s=1, p=0, b=False)
        cx = complexity_batchnorm2d(cx, w_in=w_out)
        return cx


    cx = {"h": cfg['img_size'], "w": cfg['img_size'], "flops": 0, "params": 0, "acts": 0}

    ##########################
    # Stem
    ##########################
    if 'cifar' in cfg['dataset']:
        if cfg['stem'] == 'simple':
            cx = complexity_conv2d(cx, w_in=3, w_out=cfg['stem_out'], k=3, s=1, p=1, b=False)
            cx = complexity_batchnorm2d(cx, w_in=cfg['stem_out'])
        else:
            raise ValueError()
    else:
        if cfg['stem'] == 'simple':
            cx = complexity_conv2d(cx, w_in=3, w_out=cfg['stem_out'], k=3, s=2, p=1, b=False)
            cx = complexity_batchnorm2d(cx, w_in=cfg['stem_out'])
        elif cfg['stem'] == 'resnet':
            cx = complexity_conv2d(cx, w_in=3, w_out=cfg['stem_out'], k=7, s=2, p=1, b=False)
            cx = complexity_batchnorm2d(cx, w_in=cfg['stem_out'])
            cx = complexity_maxpool2d(cx, k=3, s=2, p=0)
        else:
            raise ValueError()
    
    ##########################
    # Stage
    ##########################
    stage_dict = {
        'basic'         : BasicBlock,
        'res'           : ResBlock,
        'resbottleneck' : ResBottleneckBlock,
    }

    for stage in range(cfg['n_stage']):
        if stage == 0:
            w_in = cfg['stem_out']

        for block in range(cfg['n_block'][stage]):
            w_out = cfg['n_channel'][stage]
            if block == 0:
                stride = 2
            else:
                stride = 1
                w_in = w_out

            cx = stage_dict[cfg['type_stage']](cx, w_in, w_out, stride, cfg['bottleneck_ratio'], cfg['group_width'])

    ##########################
    # Head
    ##########################
    # for dense
    cx['h'], cx['w'] = 1, 1
    cx = complexity_conv2d(cx, w_in=cfg['n_channel'][-1], w_out=cfg['classes'], k=1, s=1, p=0, b=True)

    return cx