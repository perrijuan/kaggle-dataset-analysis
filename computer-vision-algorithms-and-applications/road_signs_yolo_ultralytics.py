# %% [code] {"_kg_hide-input":true,"_kg_hide-output":true,"execution":{"iopub.status.busy":"2025-12-09T02:41:06.362920Z","iopub.execute_input":"2025-12-09T02:41:06.363137Z","iopub.status.idle":"2025-12-09T02:41:10.457173Z","shell.execute_reply.started":"2025-12-09T02:41:06.363119Z","shell.execute_reply":"2025-12-09T02:41:10.456480Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline

#Two lines Required to Plot Plotly
import plotly.io as pio
pio.renderers.default = 'iframe'

import plotly.graph_objects as go
import plotly.express as px


import warnings
warnings.simplefilter(action='ignore', category=Warning)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [markdown]
# ## Ultralytics
# 
# Ultralytics creates cutting-edge, state-of-the-art (SOTA) YOLO models built on years of foundational research in computer vision and AI. Constantly updated for performance and flexibility, our models are fast, accurate, and easy to use. They excel at object detection, tracking, instance segmentation, image classification, and pose estimation tasks.
# 
# https://github.com/ultralytics/ultralytics

# %% [markdown]
# ## Import Libraries

# %% [code] {"execution":{"iopub.status.busy":"2025-12-09T02:41:15.774915Z","iopub.execute_input":"2025-12-09T02:41:15.775759Z","iopub.status.idle":"2025-12-09T02:41:15.783106Z","shell.execute_reply.started":"2025-12-09T02:41:15.775712Z","shell.execute_reply":"2025-12-09T02:41:15.782264Z"}}
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline

#Two lines Required to Plot Plotly
import plotly.io as pio
pio.renderers.default = 'iframe'

import plotly.graph_objects as go
import plotly.express as px


import warnings
warnings.simplefilter(action='ignore', category=Warning)

# %% [markdown]
# ## Install Ultralytics

# %% [code] {"_kg_hide-output":true,"execution":{"iopub.status.busy":"2025-12-09T02:41:21.642304Z","iopub.execute_input":"2025-12-09T02:41:21.642794Z","iopub.status.idle":"2025-12-09T02:42:40.028737Z","shell.execute_reply.started":"2025-12-09T02:41:21.642771Z","shell.execute_reply":"2025-12-09T02:42:40.027771Z"}}
%pip install  ultralytics

# %% [markdown]
# ## Import Libraries

# %% [code] {"_kg_hide-output":true,"execution":{"iopub.status.busy":"2025-12-09T02:42:54.120524Z","iopub.execute_input":"2025-12-09T02:42:54.121085Z","iopub.status.idle":"2025-12-09T02:42:54.126394Z","shell.execute_reply.started":"2025-12-09T02:42:54.121060Z","shell.execute_reply":"2025-12-09T02:42:54.125644Z"}}
import os
import random
import pandas as pd
from PIL import Image
import cv2
from ultralytics import YOLO
from IPython.display import Video
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import glob
from tqdm.notebook import trange, tqdm
import yaml
%matplotlib inline

# %% [markdown]
# ## Showing the structure


root_dir = "/kaggle/input/cabos-e-gatos"

for root, dirs, files in os.walk(root_dir):
    level = root.replace(root_dir, '').count(os.sep)
    indent = '  ' * level
    print(f"{indent}{os.path.basename(root)}/")

# %% [markdown]
# ## Showing the yaml file



yaml_path="/kaggle/input/cabos-e-gatos/data.yaml"
with open(yaml_path,"r") as file:
    data=yaml.safe_load(file)
print(data)

# %% [markdown]
# ## Labels in DataSet

# %% [code] {"execution":{"iopub.status.busy":"2025-12-09T02:43:31.597647Z","iopub.execute_input":"2025-12-09T02:43:31.598291Z","iopub.status.idle":"2025-12-09T02:43:31.603812Z","shell.execute_reply.started":"2025-12-09T02:43:31.598267Z","shell.execute_reply":"2025-12-09T02:43:31.603184Z"}}
data['names']

# %% [code] {"execution":{"iopub.status.busy":"2025-12-09T02:43:38.764589Z","iopub.execute_input":"2025-12-09T02:43:38.764871Z","iopub.status.idle":"2025-12-09T02:43:38.768888Z","shell.execute_reply.started":"2025-12-09T02:43:38.764850Z","shell.execute_reply":"2025-12-09T02:43:38.768115Z"}}
class_img={}
j=0
for i in data['names']:
    class_img[j]=i
    j=j+1

# %% [code] {"execution":{"iopub.status.busy":"2025-12-09T02:43:42.368638Z","iopub.execute_input":"2025-12-09T02:43:42.369244Z","iopub.status.idle":"2025-12-09T02:43:42.373718Z","shell.execute_reply.started":"2025-12-09T02:43:42.369219Z","shell.execute_reply":"2025-12-09T02:43:42.373027Z"}}
class_img

# %% [code] {"execution":{"iopub.status.busy":"2025-12-09T02:44:03.734549Z","iopub.execute_input":"2025-12-09T02:44:03.735266Z","iopub.status.idle":"2025-12-09T02:44:20.060199Z","shell.execute_reply.started":"2025-12-09T02:44:03.735239Z","shell.execute_reply":"2025-12-09T02:44:20.059429Z"}}

train_images_dir = "/kaggle/input/cabos-e-gatos/train/images"

# Get list of image files
image_files = [f for f in os.listdir(train_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Pick 25 random images (or less if not enough)
sample_files = random.sample(image_files, min(25, len(image_files)))

plt.figure(figsize=(20,12))

for i, img_name in enumerate(sample_files, 1):
    img_path = os.path.join(train_images_dir, img_name)
    img = Image.open(img_path)
    plt.subplot(5,5,i)
    plt.imshow(img)
    plt.axis('off')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Test Images

# %% [code] {"execution":{"iopub.status.busy":"2025-12-09T02:44:36.718789Z","iopub.execute_input":"2025-12-09T02:44:36.719078Z","iopub.status.idle":"2025-12-09T02:45:47.703676Z","shell.execute_reply.started":"2025-12-09T02:44:36.719057Z","shell.execute_reply":"2025-12-09T02:45:47.702420Z"}}

test_images_dir = "/kaggle/input/cabos-e-gatos/test/images"

# Get list of image files
image_files = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Pick 25 random images (or less if not enough)
sample_files = random.sample(image_files, min(25, len(image_files)))

plt.figure(figsize=(20,12))

for i, img_name in enumerate(sample_files, 1):
    img_path = os.path.join(test_images_dir, img_name)
    img = Image.open(img_path)
    plt.subplot(5,5,i)
    plt.imshow(img)
    plt.axis('off')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Valid images 



valid_images_dir = "/kaggle/input/cabos-e-gatos/valid/images"

# Get list of image files
image_files = [f for f in os.listdir(valid_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Pick 25 random images (or less if not enough)
sample_files = random.sample(image_files, min(25, len(image_files)))

plt.figure(figsize=(20,12))

for i, img_name in enumerate(sample_files, 1):
    img_path = os.path.join(valid_images_dir, img_name)
    img = Image.open(img_path)
    plt.subplot(5,5,i)
    plt.imshow(img)
    plt.axis('off')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Ploting the graphs of images found

# %% [code] {"execution":{"iopub.status.busy":"2025-12-09T02:46:43.470604Z","iopub.execute_input":"2025-12-09T02:46:43.471316Z","iopub.status.idle":"2025-12-09T02:46:43.614841Z","shell.execute_reply.started":"2025-12-09T02:46:43.471293Z","shell.execute_reply":"2025-12-09T02:46:43.614237Z"}}


train_dir = "/kaggle/input/cabos-e-gatos/train/images"
valid_dir = "/kaggle/input/cabos-e-gatos/valid/images"
test_dir = "/kaggle/input/cabos-e-gatos/test/images"

# Count images in each folder
train_count = len([f for f in os.listdir(train_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
valid_count = len([f for f in os.listdir(valid_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
test_count = len([f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

print(f" Train Image count :{train_count}")
print(f" Test Image count :{test_count}")
print(f" Val Image count :{valid_count}")
# Plotting
plt.figure(figsize=(15,5))
plt.bar(['Train', 'Valid', 'Test'], [train_count, valid_count, test_count], color=['blue', 'orange', 'green'])
plt.title('Number of Images in Each Dataset Split')
plt.ylabel('Number of Images')
plt.show()

# %% [markdown]
# ## cv2.putText
# 
# font=cv2.FONT_HERSHEY_SIMPLEX
# 
# fontScale=2
# 
# fontColor=(255,255,255)
# 
# lineType=cv2.line_AA
# 
# org=(443,320)
# 
# text = str(x)
# 
# cv2.putText(img, text,org,font,fontScale,fontColor,lineType)

# %% [code] {"execution":{"iopub.status.busy":"2025-12-09T02:47:04.942863Z","iopub.execute_input":"2025-12-09T02:47:04.943585Z","iopub.status.idle":"2025-12-09T02:47:04.949519Z","shell.execute_reply.started":"2025-12-09T02:47:04.943560Z","shell.execute_reply":"2025-12-09T02:47:04.948803Z"}}

def plot(image_path,label_path):
    # Load image
    # image_path = "image.jpg"
    # label_path = "image.txt"
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    
    #  Read YOLO label file
    with open(label_path, "r") as f:
        lines = f.readlines()
    
    #  Draw bounding boxes
    for line in lines:
        class_id, x_center, y_center, w, h = map(float, line.strip().split())
        
        # Convert normalized coords to pixel values
        x_center *= width
        y_center *= height
        w *= width
        h *= height
    
        # Get top-left and bottom-right
        x1 = int(x_center - w / 2)
        y1 = int(y_center - h / 2)
        x2 = int(x_center + w / 2)
        y2 = int(y_center + h / 2)
    
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  #Red (0,0,255)
        cv2.putText(image, f"{class_img}", (x1, y1 - 10),  # Original f"{class_img[class_id]}  
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) #Green (0,255,0)
    
    #  Show result
    plt.figure(figsize=(10,7))
    plt.imshow( image)
    plt.show()

# %% [markdown]
# ## Trying to draw a rectangle (wrong color and wrong text)
# 
# The issue is above cv2.putText.

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2025-12-09T02:48:29.813667Z","iopub.execute_input":"2025-12-09T02:48:29.814416Z","iopub.status.idle":"2025-12-09T02:48:31.249741Z","shell.execute_reply.started":"2025-12-09T02:48:29.814391Z","shell.execute_reply":"2025-12-09T02:48:31.248942Z"}}
plot("/kaggle/input/cabos-e-gatos/train/images/106_0034_JPG.rf.b9b6fb3db7de84c75ce6432872a2b7d2.jpg","/kaggle/input/cabos-e-gatos/train/labels/106_0034_JPG.rf.b9b6fb3db7de84c75ce6432872a2b7d2.txt")

# %% [code] {"execution":{"iopub.status.busy":"2025-12-09T02:48:41.794086Z","iopub.execute_input":"2025-12-09T02:48:41.794663Z","iopub.status.idle":"2025-12-09T02:48:41.800555Z","shell.execute_reply.started":"2025-12-09T02:48:41.794639Z","shell.execute_reply":"2025-12-09T02:48:41.799787Z"}}

def plot(image_path,label_path):
    # Load image
    # image_path = "image.jpg"
    # label_path = "image.txt"
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    
    # 📄 Read YOLO label file
    with open(label_path, "r") as f:
        lines = f.readlines()
    
    # Draw bounding boxes
    for line in lines:
        class_id, x_center, y_center, w, h = map(float, line.strip().split())
        
        # Convert normalized coords to pixel values
        x_center *= width
        y_center *= height
        w *= width
        h *= height
    
        # Get top-left and bottom-right
        x1 = int(x_center - w / 2)
        y1 = int(y_center - h / 2)
        x2 = int(x_center + w / 2)
        y2 = int(y_center + h / 2)
    
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(image, f"{class_img}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Show result
    return image

# %% [markdown]
# ## Images with rectangles (wrong text)


train_images_dir = train_dir
train_label_dir="/kaggle/input/cabos-e-gatos/train/labels"
# Get list of image files
image_files = [f for f in os.listdir(train_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Pick 25 random images (or less if not enough)
sample_files = random.sample(image_files, min(10, len(image_files)))

plt.figure(figsize=(30,90))

for i, img_name in enumerate(sample_files, 1):
    img_path = os.path.join(train_images_dir, img_name)
    labelfile= os.path.join(train_label_dir, img_name)
    # print(labelfile)
    label_path = os.path.splitext(labelfile)[0] + ".txt"
    # print(img_path,"\n",label_path)
    image_with_boxes = plot(img_path, label_path)
    plt.subplot(10,1,i)
    plt.imshow(image_with_boxes)
    # plt.axis('off')
    #plt.show()

plt.tight_layout();

# %% [code] {"execution":{"iopub.status.busy":"2025-12-09T02:50:15.114088Z","iopub.execute_input":"2025-12-09T02:50:15.114382Z","iopub.status.idle":"2025-12-09T02:50:15.118189Z","shell.execute_reply.started":"2025-12-09T02:50:15.114360Z","shell.execute_reply":"2025-12-09T02:50:15.117325Z"}}
yml_path='/kaggle/input/cabos-e-gatos/data.yaml'

# %% [code] {"execution":{"iopub.status.busy":"2025-12-09T02:50:44.223054Z","iopub.execute_input":"2025-12-09T02:50:44.223658Z","iopub.status.idle":"2025-12-09T02:50:44.227873Z","shell.execute_reply.started":"2025-12-09T02:50:44.223633Z","shell.execute_reply":"2025-12-09T02:50:44.227248Z"}}

dataset_yaml = """
# Fracture Classification Dataset Configuration
path: /kaggle/input/cabos-e-gatos
train: train/images
val: valid/images
test: test/images

# Number of classes
nc: 2

# Class names
names:
  0: normal
  1: gato  
"""

# Save the YAML to a file
with open("dataset.yaml", "w") as f:
    f.write(dataset_yaml)

# %% [code] {"execution":{"iopub.status.busy":"2025-12-09T02:50:48.695814Z","iopub.execute_input":"2025-12-09T02:50:48.696288Z","iopub.status.idle":"2025-12-09T02:50:48.699441Z","shell.execute_reply.started":"2025-12-09T02:50:48.696264Z","shell.execute_reply":"2025-12-09T02:50:48.698789Z"}}
yml_path="/kaggle/working/dataset.yaml"

# %% [markdown]
# ## Download the model

# %% [code] {"execution":{"iopub.status.busy":"2025-12-09T02:58:21.454547Z","iopub.execute_input":"2025-12-09T02:58:21.454831Z","iopub.status.idle":"2025-12-09T02:58:21.509864Z","shell.execute_reply.started":"2025-12-09T02:58:21.454811Z","shell.execute_reply":"2025-12-09T02:58:21.509232Z"}}
from ultralytics import YOLO


model = YOLO("yolo12n.pt")

# %% [markdown]
# ## Model  Epochs

# %% [code] {"_kg_hide-output":true,"execution":{"iopub.status.busy":"2025-12-09T02:58:25.299355Z","iopub.execute_input":"2025-12-09T02:58:25.299638Z","iopub.status.idle":"2025-12-09T03:07:15.097827Z","shell.execute_reply.started":"2025-12-09T02:58:25.299617Z","shell.execute_reply":"2025-12-09T03:07:15.097037Z"}}
model.train(
    data=yml_path,
    epochs=77,  #Original 100, No way, Not in this life
    batch=-1,
    optimizer="auto"
)

# %% [markdown]
# ## Output dir (Confidence, Recall curves ZEROed)


output_dir = "/kaggle/working/runs/detect/train2"

plot_files = [
    "BOXP_curve.png", "BoxPR_curve.png", "BoxF1_curve.png", "BoxR_curve.png",
    "confusion_matrix.png", "confusion_matrix_normalized.png",
    "labels.jpg", "labels_correlogram.jpg", "results.png"
]

# Filter only existing files
existing_plots = [f for f in plot_files if os.path.exists(os.path.join(output_dir, f))]

# Show each image individually
for file in existing_plots:
    img_path = os.path.join(output_dir, file)
    img = Image.open(img_path)
    plt.figure(figsize=(15, 7))
    plt.imshow(img)
    plt.title(file)
    plt.axis('off')
    plt.show()

# %% [markdown]
# ## Model result

# %% [code] {"execution":{"iopub.status.busy":"2025-12-09T03:07:46.815823Z","iopub.execute_input":"2025-12-09T03:07:46.816158Z","iopub.status.idle":"2025-12-09T03:07:46.823885Z","shell.execute_reply.started":"2025-12-09T03:07:46.816133Z","shell.execute_reply":"2025-12-09T03:07:46.823267Z"}}
model_result=pd.read_csv("/kaggle/working/runs/detect/train2/results.csv")

# %% [code] {"execution":{"iopub.status.busy":"2025-12-09T03:07:49.489325Z","iopub.execute_input":"2025-12-09T03:07:49.489902Z","iopub.status.idle":"2025-12-09T03:07:49.508836Z","shell.execute_reply.started":"2025-12-09T03:07:49.489877Z","shell.execute_reply":"2025-12-09T03:07:49.508017Z"}}
model_result

# %% [markdown]
# ## Training Metrics and Loss (more loss than metrics:)

# %% [code] {"execution":{"iopub.status.busy":"2025-12-09T03:07:58.332919Z","iopub.execute_input":"2025-12-09T03:07:58.333800Z","iopub.status.idle":"2025-12-09T03:08:00.774372Z","shell.execute_reply.started":"2025-12-09T03:07:58.333772Z","shell.execute_reply":"2025-12-09T03:08:00.773619Z"}}


model_result.columns = model_result.columns.str.strip()

fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(15, 15))

# Plot the columns using seaborn
sns.lineplot(x='epoch', y='train/box_loss', data=model_result, ax=axs[0,0])
sns.lineplot(x='epoch', y='train/cls_loss', data=model_result, ax=axs[0,1])
sns.lineplot(x='epoch', y='train/dfl_loss', data=model_result, ax=axs[1,0])
sns.lineplot(x='epoch', y='metrics/precision(B)', data=model_result, ax=axs[1,1])
sns.lineplot(x='epoch', y='metrics/recall(B)', data=model_result, ax=axs[2,0])
sns.lineplot(x='epoch', y='metrics/mAP50(B)', data=model_result, ax=axs[2,1])
sns.lineplot(x='epoch', y='metrics/mAP50-95(B)', data=model_result, ax=axs[3,0])
sns.lineplot(x='epoch', y='val/box_loss', data=model_result, ax=axs[3,1])
sns.lineplot(x='epoch', y='val/cls_loss', data=model_result, ax=axs[4,0])
sns.lineplot(x='epoch', y='val/dfl_loss', data=model_result, ax=axs[4,1])

# Set titles and axis labels for each subplot
axs[0,0].set(title='Train Box Loss')
axs[0,1].set(title='Train Class Loss')
axs[1,0].set(title='Train DFL Loss')
axs[1,1].set(title='Metrics Precision (B)')
axs[2,0].set(title='Metrics Recall (B)')
axs[2,1].set(title='Metrics mAP50 (B)')
axs[3,0].set(title='Metrics mAP50-95 (B)')
axs[3,1].set(title='Validation Box Loss')
axs[4,0].set(title='Validation Class Loss')
axs[4,1].set(title='Validation DFL Loss')


plt.suptitle('Training Metrics and Loss', fontsize=24)
plt.subplots_adjust(top=0.8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Training and validation

# %% [code] {"execution":{"iopub.status.busy":"2025-12-09T03:08:14.005339Z","iopub.execute_input":"2025-12-09T03:08:14.005676Z","iopub.status.idle":"2025-12-09T03:08:16.969415Z","shell.execute_reply.started":"2025-12-09T03:08:14.005653Z","shell.execute_reply":"2025-12-09T03:08:16.968468Z"}}

# Set Seaborn style for better aesthetics
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Define a color palette for consistency
colors = sns.color_palette("husl", 8)

# Strip whitespace from column names (just in case)
model_result.columns = model_result.columns.str.strip()

# Create figure with subplots for original plots (5 rows, 2 columns)
fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(15, 20), dpi=100)

# Original plots
sns.lineplot(x='epoch', y='train/box_loss', data=model_result, ax=axs[0, 0], color=colors[0], linewidth=2)
sns.lineplot(x='epoch', y='train/cls_loss', data=model_result, ax=axs[0, 1], color=colors[1], linewidth=2)
sns.lineplot(x='epoch', y='train/dfl_loss', data=model_result, ax=axs[1, 0], color=colors[2], linewidth=2)
sns.lineplot(x='epoch', y='metrics/precision(B)', data=model_result, ax=axs[1, 1], color=colors[3], linewidth=2)
sns.lineplot(x='epoch', y='metrics/recall(B)', data=model_result, ax=axs[2, 0], color=colors[4], linewidth=2)
sns.lineplot(x='epoch', y='metrics/mAP50(B)', data=model_result, ax=axs[2, 1], color=colors[5], linewidth=2)
sns.lineplot(x='epoch', y='metrics/mAP50-95(B)', data=model_result, ax=axs[3, 0], color=colors[6], linewidth=2)
sns.lineplot(x='epoch', y='val/box_loss', data=model_result, ax=axs[3, 1], color=colors[0], linewidth=2)
sns.lineplot(x='epoch', y='val/cls_loss', data=model_result, ax=axs[4, 0], color=colors[1], linewidth=2)
sns.lineplot(x='epoch', y='val/dfl_loss', data=model_result, ax=axs[4, 1], color=colors[2], linewidth=2)

# Set titles and labels
axs[0, 0].set(title='Train Box Loss', xlabel='Epoch', ylabel='Loss')
axs[0, 1].set(title='Train Class Loss', xlabel='Epoch', ylabel='Loss')
axs[1, 0].set(title='Train DFL Loss', xlabel='Epoch', ylabel='Loss')
axs[1, 1].set(title='Precision (B)', xlabel='Epoch', ylabel='Precision')
axs[2, 0].set(title='Recall (B)', xlabel='Epoch', ylabel='Recall')
axs[2, 1].set(title='mAP50 (B)', xlabel='Epoch', ylabel='mAP50')
axs[3, 0].set(title='mAP50-95 (B)', xlabel='Epoch', ylabel='mAP50-95')
axs[3, 1].set(title='Validation Box Loss', xlabel='Epoch', ylabel='Loss')
axs[4, 0].set(title='Validation Class Loss', xlabel='Epoch', ylabel='Loss')
axs[4, 1].set(title='Validation DFL Loss', xlabel='Epoch', ylabel='Loss')

# Main title and layout
plt.suptitle('Training and Validation Metrics\n\n')
plt.tight_layout()

# %% [markdown]
# ## Additional Plots

# %% [code] {"execution":{"iopub.status.busy":"2025-12-09T03:08:30.129229Z","iopub.execute_input":"2025-12-09T03:08:30.129775Z","iopub.status.idle":"2025-12-09T03:08:30.979218Z","shell.execute_reply.started":"2025-12-09T03:08:30.129748Z","shell.execute_reply":"2025-12-09T03:08:30.978416Z"}}

# Additional Plots
# 1. Train vs Validation Loss Comparison
fig_comp, axs_comp = plt.subplots(1, 3, figsize=(15, 5), dpi=100)
sns.lineplot(x='epoch', y='train/box_loss', data=model_result, label='Train', ax=axs_comp[0], color=colors[0], linewidth=2)
sns.lineplot(x='epoch', y='val/box_loss', data=model_result, label='Validation', ax=axs_comp[0], color=colors[1], linewidth=2)
axs_comp[0].set(title='Box Loss: Train vs Val', xlabel='Epoch', ylabel='Box Loss')
axs_comp[0].legend()

sns.lineplot(x='epoch', y='train/cls_loss', data=model_result, label='Train', ax=axs_comp[1], color=colors[0], linewidth=2)
sns.lineplot(x='epoch', y='val/cls_loss', data=model_result, label='Validation', ax=axs_comp[1], color=colors[1], linewidth=2)
axs_comp[1].set(title='Class Loss: Train vs Val', xlabel='Epoch', ylabel='Class Loss')
axs_comp[1].legend()

sns.lineplot(x='epoch', y='train/dfl_loss', data=model_result, label='Train', ax=axs_comp[2], color=colors[0], linewidth=2)
sns.lineplot(x='epoch', y='val/dfl_loss', data=model_result, label='Validation', ax=axs_comp[2], color=colors[1], linewidth=2)
axs_comp[2].set(title='DFL Loss: Train vs Val', xlabel='Epoch', ylabel='DFL Loss')
axs_comp[2].legend()

plt.tight_layout()
plt.show()

# %% [code] {"_kg_hide-output":false,"execution":{"iopub.status.busy":"2025-12-09T03:08:35.145766Z","iopub.execute_input":"2025-12-09T03:08:35.146126Z","iopub.status.idle":"2025-12-09T03:08:36.822694Z","shell.execute_reply.started":"2025-12-09T03:08:35.146086Z","shell.execute_reply":"2025-12-09T03:08:36.821882Z"}}

# 2. Precision vs Recall Scatter Plot
fig_pr, ax_pr = plt.subplots(figsize=(10, 10), dpi=100)
sns.scatterplot(x='metrics/recall(B)', y='metrics/precision(B)', hue='epoch', size='epoch',palette='viridis', data=model_result, ax=ax_pr, legend='full')
ax_pr.set(title='Precision vs Recall', xlabel='Recall (B)', ylabel='Precision (B)')
plt.legend(loc="best")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Total Loss

# %% [code] {"execution":{"iopub.status.busy":"2025-12-09T03:08:57.902065Z","iopub.execute_input":"2025-12-09T03:08:57.902672Z","iopub.status.idle":"2025-12-09T03:08:58.203330Z","shell.execute_reply.started":"2025-12-09T03:08:57.902648Z","shell.execute_reply":"2025-12-09T03:08:58.202576Z"}}

# 3. Total Loss
model_result['train/total_loss'] = model_result['train/box_loss'] + model_result['train/cls_loss'] + model_result['train/dfl_loss']
model_result['val/total_loss'] = model_result['val/box_loss'] + model_result['val/cls_loss'] + model_result['val/dfl_loss']

fig_total, ax_total = plt.subplots(figsize=(10, 5), dpi=100)
sns.lineplot(x='epoch', y='train/total_loss', data=model_result, label='Train', ax=ax_total, color=colors[0], linewidth=2)
sns.lineplot(x='epoch', y='val/total_loss', data=model_result, label='Validation', ax=ax_total, color=colors[1], linewidth=2)
ax_total.set(title='Total Loss: Train vs Val', xlabel='Epoch', ylabel='Total Loss')
ax_total.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## mAP50 vs mAP50-95

# %% [code] {"execution":{"iopub.status.busy":"2025-12-09T03:09:06.108967Z","iopub.execute_input":"2025-12-09T03:09:06.109579Z","iopub.status.idle":"2025-12-09T03:09:06.409962Z","shell.execute_reply.started":"2025-12-09T03:09:06.109555Z","shell.execute_reply":"2025-12-09T03:09:06.409242Z"}}

# 4. mAP50 vs mAP50-95
fig_map, ax_map = plt.subplots(figsize=(10, 5), dpi=100)
sns.lineplot(x='epoch', y='metrics/mAP50(B)', data=model_result, label='mAP50', ax=ax_map, color=colors[5], linewidth=2)
sns.lineplot(x='epoch', y='metrics/mAP50-95(B)', data=model_result, label='mAP50-95', ax=ax_map, color=colors[6], linewidth=2)
ax_map.set(title='mAP50 vs mAP50-95', xlabel='Epoch', ylabel='mAP')
ax_map.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Stacked Area plot

# %% [code] {"execution":{"iopub.status.busy":"2025-12-09T03:09:11.800328Z","iopub.execute_input":"2025-12-09T03:09:11.800993Z","iopub.status.idle":"2025-12-09T03:09:12.332173Z","shell.execute_reply.started":"2025-12-09T03:09:11.800954Z","shell.execute_reply":"2025-12-09T03:09:12.331409Z"}}

# 5. Loss Components Contribution (Stacked Area Plot)
fig_stack, axs_stack = plt.subplots(1, 2, figsize=(15, 5), dpi=100)
axs_stack[0].stackplot(model_result['epoch'],
                       model_result['train/box_loss'],
                       model_result['train/cls_loss'],
                       model_result['train/dfl_loss'],
                       labels=['Box Loss', 'Class Loss', 'DFL Loss'],
                       colors=colors[:3])
axs_stack[0].set(title='Train Loss Components', xlabel='Epoch', ylabel='Loss')
axs_stack[0].legend(loc='upper right')

axs_stack[1].stackplot(model_result['epoch'],
                       model_result['val/box_loss'],
                       model_result['val/cls_loss'],
                       model_result['val/dfl_loss'],
                       labels=['Box Loss', 'Class Loss', 'DFL Loss'],
                       colors=colors[:3])
axs_stack[1].set(title='Validation Loss Components', xlabel='Epoch', ylabel='Loss')
axs_stack[1].legend(loc='upper right')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Smoothed metrics

# %% [code] {"_kg_hide-output":true,"execution":{"iopub.status.busy":"2025-12-09T03:09:21.567277Z","iopub.execute_input":"2025-12-09T03:09:21.567893Z","iopub.status.idle":"2025-12-09T03:09:21.874384Z","shell.execute_reply.started":"2025-12-09T03:09:21.567865Z","shell.execute_reply":"2025-12-09T03:09:21.873621Z"}}

# 6. Smoothed Metrics (e.g., mAP50)
model_result_smoothed = model_result.rolling(window=3, center=True).mean()
fig_smooth, ax_smooth = plt.subplots(figsize=(10, 5), dpi=100)
sns.lineplot(x='epoch', y='metrics/mAP50(B)', data=model_result, label='Original', alpha=0.3, ax=ax_smooth, color=colors[5])
sns.lineplot(x='epoch', y='metrics/mAP50(B)', data=model_result_smoothed, label='Smoothed', ax=ax_smooth, color=colors[5], linewidth=2)
ax_smooth.set(title='Smoothed mAP50 (B)', xlabel='Epoch', ylabel='mAP50 (B)')
ax_smooth.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Heatmap

# %% [code] {"execution":{"iopub.status.busy":"2025-12-09T03:09:30.743826Z","iopub.execute_input":"2025-12-09T03:09:30.744501Z","iopub.status.idle":"2025-12-09T03:09:31.296273Z","shell.execute_reply.started":"2025-12-09T03:09:30.744479Z","shell.execute_reply":"2025-12-09T03:09:31.295475Z"}}

# 7. Metrics Correlation Heatmap
metrics_cols = ['train/box_loss', 'train/cls_loss', 'train/dfl_loss', 
                'val/box_loss', 'val/cls_loss', 'val/dfl_loss', 
                'metrics/precision(B)', 'metrics/recall(B)', 
                'metrics/mAP50(B)', 'metrics/mAP50-95(B)']
correlation_matrix = model_result[metrics_cols].corr()

fig_corr, ax_corr = plt.subplots(figsize=(15, 8), dpi=100)
sns.heatmap(correlation_matrix, annot=True, cmap='gnuplot', fmt='.2f', ax=ax_corr, cbar_kws={'label': 'Correlation'})
ax_corr.set(title='Metrics Correlation Heatmap')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Training time plot

# %% [code] {"execution":{"iopub.status.busy":"2025-12-09T03:09:37.901346Z","iopub.execute_input":"2025-12-09T03:09:37.901915Z","iopub.status.idle":"2025-12-09T03:09:39.948726Z","shell.execute_reply.started":"2025-12-09T03:09:37.901888Z","shell.execute_reply":"2025-12-09T03:09:39.947784Z"}}

# 8. Training Time Plot (using 'time' column)
fig_time, ax_time = plt.subplots(figsize=(10, 5), dpi=100)
sns.lineplot(x='epoch', y='time', data=model_result, ax=ax_time, color=colors[7], linewidth=2)
ax_time.set(title='Training Time per Epoch', xlabel='Epoch', ylabel='Time (seconds)')
plt.tight_layout()
plt.show()

# Display the original plot
plt.figure(fig)
plt.show()

# %% [markdown]
# ## Best Model

# %% [code] {"_kg_hide-output":true,"execution":{"iopub.status.busy":"2025-12-09T03:09:49.390804Z","iopub.execute_input":"2025-12-09T03:09:49.391390Z","iopub.status.idle":"2025-12-09T03:10:08.755468Z","shell.execute_reply.started":"2025-12-09T03:09:49.391362Z","shell.execute_reply":"2025-12-09T03:10:08.754638Z"}}

best_model="/kaggle/working/runs/detect/train2/weights/best.pt"
test_model=YOLO(best_model)
metrics=test_model.val(split="test")


for metric_name, value in metrics.results_dict.items():
    print(f"{metric_name}: {value}")

# %% [code] {"_kg_hide-output":true,"execution":{"iopub.status.busy":"2025-12-09T03:10:14.891173Z","iopub.execute_input":"2025-12-09T03:10:14.891494Z","iopub.status.idle":"2025-12-09T03:11:38.815722Z","shell.execute_reply.started":"2025-12-09T03:10:14.891469Z","shell.execute_reply":"2025-12-09T03:11:38.814793Z"}}

from ultralytics import YOLO

original_images_path = '/kaggle/input/cabos-e-gatos/test/images'
labels_path = '/kaggle/input/cabos-e-gatos/test/labels'

class_names = ['normal', 'gato']


image_files = [f for f in os.listdir(original_images_path) if f.lower().endswith('.jpg')]


step = max(1, len(image_files) // 10)
selected_images = image_files[::step][:10]

rows, cols = 10, 2
fig, axes = plt.subplots(rows, cols, figsize=(16, 40))
fig.suptitle('Original (with Label Boxes) and Prediction Images', fontsize=24)

test_model = YOLO("/kaggle/working/runs/detect/train/weights/best.pt")

def draw_boxes_from_yolo_label(img, label_file, class_names):
    h, w = img.shape[:2]
    with open(label_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        cls, x_center, y_center, width, height = map(float, line.strip().split())
        cls = int(cls)
        
        
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)
        
        color = (255, 0, 0)  # Red color
        label = f"{class_names}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

for i, img_name in enumerate(selected_images):
    img_path = os.path.join(original_images_path, img_name)
    img = cv2.imread(img_path)
    img_orig_with_boxes = img.copy()

    label_file = os.path.join(labels_path, img_name.replace('.jpg', '.txt'))


    if os.path.exists(label_file):
        draw_boxes_from_yolo_label(img_orig_with_boxes, label_file, class_names)
    else:
        print(f"Label file missing for {img_name}")

    img_orig_rgb = cv2.cvtColor(img_orig_with_boxes, cv2.COLOR_BGR2RGB)

    
    results = test_model.predict(source=img_path, imgsz=640, conf=0.01)  # কম confidence থ্রেশোল্ড
    print(f"{img_name}: Detected boxes count: {len(results[0].boxes)}")  # বক্স সংখ্যা দেখাও

    pred_annotated = results[0].plot(line_width=3)
    pred_annotated_rgb = cv2.cvtColor(pred_annotated, cv2.COLOR_BGR2RGB)

    axes[i, 0].imshow(img_orig_rgb)
    axes[i, 0].set_title("Original with Label Boxes", fontsize=12)
    axes[i, 0].axis('off')

    axes[i, 1].imshow(pred_annotated_rgb)
    axes[i, 1].set_title("Prediction", fontsize=12)
    axes[i, 1].axis('off')

plt.tight_layout()
plt.show()
