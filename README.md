## SVA
Code for the paper "[Speckle-Variant Attack: Toward Transferable Adversarial Attack to SAR Target Recognition](https://ieeexplore.ieee.org/abstract/document/9800917)".

## Preparation
Please install [kornia](https://kornia.readthedocs.io/en/latest/) for differentiable filtering, and arrange [model weights](https://pan.baidu.com/s/1-r-GBiH8jM-mAoAD-mJmYQ?pwd=5631) and our [test set](https://pan.baidu.com/s/1nZ1oZk3xYpDLdAEY0_gYvg?pwd=5631) to './models/' and './dataset/' respectively. Please modify your torchvision.datasets.folder with the following code to automatically load the .pt file:

```
import torch

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp", ".pt")

def default_loader(path: str) -> Any:
    from torchvision import get_image_backend
    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        if path[-3:] == '.pt':
            return torch.load(path)
        else:
            return pil_loader(path)
```

## Evaluation
Run the command below to reproduce evaluations in the paper. The results slightly differ from those reported in the paper due to some minor corrections being made when we tidy up the code for this repository. 
You can also assign another surrogate model by '--surrogate aconv/alex/vgg/dense/resnet/resnext/incres/incv4'. 
This may results in performance corruption to all the compared methods (DI/SI/SVA) due to severe overfitting that makes the transformation not loss preserving anymore, such as DenseNet121. 


```
python SVA.py --eps 48/255 --baseline FGSM/BIM/PGD --attack NA/DI/SI/SVA --batch_size 64 --beta 1.5 --s 7
```


## Citation
If you find our paper and this repository useful, please consider citing our work.
```bibtex
@ARTICLE{pengsva22,
  author={Peng, Bowen and Peng, Bo and Zhou, Jie and Xia, Jingyuan and Liu, Li},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={Speckle-Variant Attack: Toward Transferable Adversarial Attack to SAR Target Recognition}, 
  year={2022},
  volume={19},
  pages={1-5},
  doi={10.1109/LGRS.2022.3184311}}
```
