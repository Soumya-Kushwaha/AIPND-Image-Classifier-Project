import argparse
import json
import torch
import PIL
from PIL import Image
import numpy as np
from math import ceil
from torchvision import models

def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--image', type=str, required=True)

    parser.add_argument('--checkpoint',type=str,required=True)
        
    parser.add_argument('--top_k',type=int, default=5)
        
    parser.add_argument('--category_names',type=str, default= "cat_to_name.json")

    parser.add_argument('--gpu',action="store_true")

    args = parser.parse_args()
    return args
in_arg= arg_parser()    
    

    
    

with open(in_arg.category_names, 'r') as f:
    cat_to_name = json.load(f)
    
def loadcheckpoint():    
    checkpointData = torch.load(in_arg.checkpoint)
            
        
    if checkpointData['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
    else: 
        exec("model = models.{}(pretrained=True)".checkpointData['arch'])
        model.name = checkpointData['arch']
        
    for param in model.parameters(): 
        param.requires_grad = False
        
    model.classifier = checkpointData['classifier']
    model.class_to_idx = checkpointData['class_to_idx']
    model.load_state_dict(checkpointData['state_dict'])

    return model
model=loadcheckpoint()    
    

def process_image(image):
    new_img=Image.open(image)
    original_width, original_height=new_img.size
    if original_width<original_height:
        new_img.thumbnail((256,256**600))
    else:
        new_img.thumbnail((256**600,256))
    center = original_width/4, original_height/4
    left, top, right, bottom = center[0]-(244/2), center[1]-(244/2), center[0]+(244/2), center[1]+(244/2)
    new_img = new_img.crop((left, top, right, bottom)) 
    numpy_img=np.array(new_img)/255
    mean=[0.485,0.456,0.406]
    std=[0.229,0.224,0.225]
    numpy_img=(numpy_img-mean)/std
    numpy_img=numpy_img.transpose(2,0,1)
    return numpy_img



def gpu_status():
    if not in_arg.gpu:
        return torch.device("cpu")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if device == "cpu":
        print("Failed to find cuda... going with cpu")
    return device




def predict(image_path, model, topk):
    
    device = gpu_status()        
    model.to('cpu')
    img=process_image(image_path)
    image_tensor=torch.from_numpy(np.expand_dims(img,axis=0)).type(torch.FloatTensor).to("cpu")
    output=model.forward(image_tensor)
    pb=torch.exp(output)
    top_pb, top_class = pb.topk(topk)
    top_pb=top_pb.tolist()[0]
    top_class=top_class.tolist()[0]
    data = {val: key for key, val in model.class_to_idx.items()}
    top_flow = []
    for i in top_class:
        iw="{}".format(data.get(i))
        top_flow.append(cat_to_name.get(iw))
    return top_pb, top_flow
    
    
data = predict(in_arg.image, model, in_arg.top_k)

probs, flowers = data
for iterate in range(in_arg.top_k):
   
    if iterate+1 ==1:
        print("The output is :")
        print("{} is the most likely flower with {}% ".format(flowers[iterate],ceil(probs[iterate]*100)))

    else:
        print("Other outputs are :")
        print("{} flower with {}% ".format(flowers[iterate],ceil(probs[iterate]*100)))
