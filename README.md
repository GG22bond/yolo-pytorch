# yolo-pytorch
Pytorch code for Yolov5, Yolov8, and Yolo9.

```
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
```
