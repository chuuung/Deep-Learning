# SPECT images with Parkinson’s disease

## Introduction
Analyze single photon emission computed tomography (SPECT) images from individuals with Parkinson’s disease (PD), who are divided into three stages (1, 2, 3) according to illness severity. 

The goal is to predict patients’ PD illness stages by using their SPECT images. Since each image can only belong to one of the 3 disease stages, this is a multiclass classification problem. 

## Models
Use convolutional neural networks (CNNs) to extract features in these images. Utilize transfer learning to adopt the pre-trained model whose weights are trained from ImageNet to improve prediction accuracy
- VGG16
- ResNet50

## Create 3 channel image from gray images
For the CNN kernel, three-dimensional images are required as input, but the dataset consists of black and white images. Combine the indices with the preceding and succeeding images to create a three-channel image.

```python
for index, path in zip(self.indexs, self.paths):
    image = pydicom.dcmread(config.root+'/data'+path).pixel_array
    image = torch.tensor(image.astype(np.float32))
    image = image[index-1:index+2, :, :] # 取指定張數和前後共三個, image size = (3, 128, 128)
    if transforms:
        image = transforms(image)
    self.images.append(image)
```

## Add other variables
The dataset includes variables such as age and gender. Therefore, modify the model to incorporate these variables as input.

```python
self.classifier = nn.Sequential(
    nn.Linear(514, num_classes) # 512: vgg16特徵層結果, 2: age & gender
)
outputs = torch.cat([output, age.view(-1, 1), gender.view(-1, 1)], dim = 1) # output size = (batch_size, 512), age size = (batch_size), age.view(-1, 1) size = (batch_size, 1), dim = 1; columns concat
```

## Predict
Use K-fold and vote the result by each model's output.

