def merge_metric_objects(a : dict, b : dict):
    """
        This utility function prepends metrics from ´b´ to ´a´. The structure has to be as follows:
        
        {
            'generator' : dict
                metric1 : list
                metric2 : list 
                
                ...
            'discriminator' : dict
                metric1 : list
                metric2 : list
                
                ...
        }
        
    """
    a_keys = a.keys()
    b_keys = b.keys()
    assert a_keys == b_keys, f'Second dictionary must have the same keys as first. Expected {a_keys} but received {b_keys}'
    
    for key in a_keys:
        as_keys = a[key].keys() # a sub key
        bs_keys = b[key].keys()
        assert as_keys == bs_keys, f'Second dictionary must have the same sub-keys as first. Expected {as_keys} but received {bs_keys}'

        for skey in as_keys:
            a[key][skey] = b[key][skey] + a[key][skey]
    
    return a