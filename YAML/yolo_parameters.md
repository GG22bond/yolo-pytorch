````
yolov5l:

                 from  n    params  module                                  arguments                     
  0                -1  1      7040  models.common.Focus                     [3, 64, 3]                    
  1                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
  2                -1  1    156928  models.common.C3                        [128, 128, 3]                 
  3                -1  1    295424  models.common.Conv                      [128, 256, 3, 2]              
  4                -1  1   1611264  models.common.C3                        [256, 256, 9]                 
  5                -1  1   1180672  models.common.Conv                      [256, 512, 3, 2]              
  6                -1  1   6433792  models.common.C3                        [512, 512, 9]                 
  7                -1  1   4720640  models.common.Conv                      [512, 1024, 3, 2]             
  8                -1  1   2624512  models.common.SPP                       [1024, 1024, [5, 9, 13]]      
  9                -1  1   9971712  models.common.C3                        [1024, 1024, 3, False]        
 10                -1  1    525312  models.common.Conv                      [1024, 512, 1, 1]             
 11                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 12           [-1, 6]  1         0  models.common.Concat                    [1]                           
 13                -1  1   2757632  models.common.C3                        [1024, 512, 3, False]         
 14                -1  1    131584  models.common.Conv                      [512, 256, 1, 1]              
 15                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 16           [-1, 4]  1         0  models.common.Concat                    [1]                           
 17                -1  1    690688  models.common.C3                        [512, 256, 3, False]          
 18                -1  1    590336  models.common.Conv                      [256, 256, 3, 2]              
 19          [-1, 14]  1         0  models.common.Concat                    [1]                           
 20                -1  1   2495488  models.common.C3                        [512, 512, 3, False]          
 21                -1  1   2360320  models.common.Conv                      [512, 512, 3, 2]              
 22          [-1, 10]  1         0  models.common.Concat                    [1]                           
 23                -1  1   9971712  models.common.C3                        [1024, 1024, 3, False]        
 24      [17, 20, 23]  1    457725  Detect                                  [80, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [256, 512, 1024]]

Model Summary: 499 layers, 47056765 parameters, 47056765 gradients, 115.9 GFLOPS
````

````
yolov9c

                 from  n    params  module                                  arguments                     
  0                -1  1      1856  models.common.Conv                      [3, 64, 3, 2]                 
  1                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
  2                -1  1    212864  models.common.RepNCSPELAN4              [128, 256, 128, 64, 1]        
  3                -1  1    164352  models.common.ADown                     [256, 256]                    
  4                -1  1    847616  models.common.RepNCSPELAN4              [256, 512, 256, 128, 1]       
  5                -1  1    656384  models.common.ADown                     [512, 512]                    
  6                -1  1   2857472  models.common.RepNCSPELAN4              [512, 512, 512, 256, 1]       
  7                -1  1    656384  models.common.ADown                     [512, 512]                    
  8                -1  1   2857472  models.common.RepNCSPELAN4              [512, 512, 512, 256, 1]       
  9                -1  1    656896  models.common.SPPELAN                   [512, 512, 256]               
 10                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 11           [-1, 6]  1         0  models.common.Concat                    [1]                           
 12                -1  1   3119616  models.common.RepNCSPELAN4              [1024, 512, 512, 256, 1]      
 13                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 14           [-1, 4]  1         0  models.common.Concat                    [1]                           
 15                -1  1    912640  models.common.RepNCSPELAN4              [1024, 256, 256, 128, 1]      
 16                -1  1    164352  models.common.ADown                     [256, 256]                    
 17          [-1, 12]  1         0  models.common.Concat                    [1]                           
 18                -1  1   2988544  models.common.RepNCSPELAN4              [768, 512, 512, 256, 1]       
 19                -1  1    656384  models.common.ADown                     [512, 512]                    
 20           [-1, 9]  1         0  models.common.Concat                    [1]                           
 21                -1  1   3119616  models.common.RepNCSPELAN4              [1024, 512, 512, 256, 1]      
 22      [15, 18, 21]  1    327165  Detect                                  [80, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [256, 512, 512]]

Model Summary: 682 layers, 20273597 parameters, 20273597 gradients, 79.1 GFLOPS
````

````
yolov8l:
                 from  n    params  module                                  arguments                     
  0                -1  1      1856  models.common.Conv                      [3, 64, 3, 2]                 
  1                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
  2                -1  1    279808  models.common.C2f                       [128, 128, 3, True]           
  3                -1  1    295424  models.common.Conv                      [128, 256, 3, 2]              
  4                -1  1   2101248  models.common.C2f                       [256, 256, 6, True]           
  5                -1  1   1180672  models.common.Conv                      [256, 512, 3, 2]              
  6                -1  1   8396800  models.common.C2f                       [512, 512, 6, True]           
  7                -1  1   4720640  models.common.Conv                      [512, 1024, 3, 2]             
  8                -1  1  17836032  models.common.C2f                       [1024, 1024, 3, True]         
  9                -1  1   2624512  models.common.SPPF                      [1024, 1024, 5]               
 10                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 11           [-1, 6]  1         0  models.common.Concat                    [1]                           
 12                -1  1   4985856  models.common.C2f                       [1536, 512, 3]                
 13                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 14           [-1, 4]  1         0  models.common.Concat                    [1]                           
 15                -1  1   1247744  models.common.C2f                       [768, 256, 3]                 
 16                -1  1    590336  models.common.Conv                      [256, 256, 3, 2]              
 17          [-1, 12]  1         0  models.common.Concat                    [1]                           
 18                -1  1   4592640  models.common.C2f                       [768, 512, 3]                 
 19                -1  1   2360320  models.common.Conv                      [512, 512, 3, 2]              
 20           [-1, 9]  1         0  models.common.Concat                    [1]                           
 21                -1  1  18360320  models.common.C2f                       [1536, 1024, 3]               
 22      [15, 18, 21]  1    457725  Detect                                  [80, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [256, 512, 1024]]
 
Model Summary: 401 layers, 70105917 parameters, 70105917 gradients, 166.8 GFLOPS
````