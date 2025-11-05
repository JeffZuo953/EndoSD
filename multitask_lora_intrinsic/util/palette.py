def get_palette(name: str = 'inhouse'):
    """
    Returns the color palette for a given dataset.
    Currently supports: 'inhouse'
    """
    if name == 'inhouse':
        # Labels for InHouseSegDataset:
        # 0: background
        # 1: stone
        # 2: laser
        # 3: polyp
        return [
            [0, 0, 0],        # 0: background (black)
            [255, 255, 255],  # 1: stone (white)
            [128, 128, 128],  # 2: laser (gray)
            [192, 192, 192],  # 3: polyp (silver)
        ]
    else:
        raise ValueError(f"Unknown palette name: {name}")