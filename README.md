# Arcface_implement
基于pytorch Arcface的视频实时人脸检测
**问题描述：**  
Missing key(s) in state_dict: "conv1.weight",  
Unexpected key(s) in state_dict: "module.conv1.weight",  
原因：多了module这个关键字在前面，可能是因为模型是在GPU分布式计算的。  
**改正load：**  

```python
# original saved file with DataParallel
state_dict = torch.load('myfile.pth')
# create new OrderedDict that does not contain `module.`
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
# load params
model.load_state_dict(new_state_dict)
```
solution and code 来自 [pytorch.org](https://discuss.pytorch.org/t/missing-keys-unexpected-keys-in-state-dict-when-loading-self-trained-model/22379/2)
