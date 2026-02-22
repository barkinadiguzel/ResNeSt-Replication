class ResNeStConfig:
    radix = 2              # R
    cardinality = 1        # K (groups)
    bottleneck_width = 64  # base width inside bottleneck
    groups = 1             # grouped conv parameter

    # Network depth example 
    layers = [3, 4, 6, 3]

    # Input channels
    in_channels = 3

    # Number of output classes 
    num_classes = 1000
